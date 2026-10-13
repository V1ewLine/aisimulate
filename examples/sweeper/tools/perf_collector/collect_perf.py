#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model-scoped SGLang collection campaigns; run in the intended conda environment."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal

import yaml
from packaging.requirements import Requirement
from pydantic import BaseModel, ConfigDict, Field, model_validator

ROOT = Path(__file__).resolve().parents[4]
PACKAGE = ROOT / "python/aisimulate"
COLLECTOR = PACKAGE / "collector"
MODEL = "nvidia/GLM-5.2-NVFP4"
ARCH = "GlmMoeDsaForCausalLM"
CATEGORIES = {
    "gemm": ["gemm"],
    "moe": ["moe"],
    "attention": [
        "dsa_context_module",
        "dsa_generation_module",
        "dsa_context_module_skip_indexer",
        "dsa_generation_module_skip_indexer",
        "glm5_mqa_logits_module",
        "glm5_topk_module",
        "glm5_dsa_attn_module",
    ],
    "communication": [],
}
Positive = Annotated[int, Field(strict=True, gt=0)]
Nonnegative = Annotated[int, Field(strict=True, ge=0)]
Sizes = Annotated[list[Positive], Field(min_length=1)]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Model(Strict):
    id: Literal["nvidia/GLM-5.2-NVFP4"] = MODEL
    config: Path
    kv_cache_dtype: Literal["fp8", "bfloat16"] = "fp8"


class Parallelism(Strict):
    num_gpus: Positive = 8
    attention_tp_sizes: Sizes = [1, 2, 4, 8]
    moe_ep_sizes: Sizes = [1, 2, 4, 8]

    @model_validator(mode="after")
    def check_sizes(self):
        for size in self.attention_tp_sizes + self.moe_ep_sizes:
            if self.num_gpus % size:
                raise ValueError("Every TP/EP size must divide parallelism.num_gpus")
        return self


class Attention(Strict):
    context_batch_sizes: Sizes = [1, 2, 4, 8]
    context_sequence_lengths: Sizes = [128, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    prefix_lengths: Annotated[list[Nonnegative], Field(min_length=1)] = [0, 19968, 32768]
    generation_batch_sizes: Sizes = [1, 2, 4, 8, 16, 32, 64, 96, 128]
    generation_sequence_lengths: Sizes = [1024, 8192, 16384, 32768, 65536]
    context_max_tokens: Positive = 131072
    generation_max_tokens: Positive = 8388608

    @model_validator(mode="after")
    def nonempty(self):
        for phase in ("context", "generation"):
            if min(getattr(self, f"{phase}_batch_sizes")) * min(getattr(self, f"{phase}_sequence_lengths")) > getattr(
                self, f"{phase}_max_tokens"
            ):
                raise ValueError(f"{phase}: token budget excludes every requested shape")
        return self


class Sweeps(Strict):
    gemm_tokens: Sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    moe_tokens: Sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    moe_power_law_alphas: Annotated[list[Annotated[float, Field(gt=0, allow_inf_nan=False)]], Field(min_length=1)] = [
        1.01
    ]
    moe_balanced: bool = True
    attention: Attention = Field(default_factory=Attention)


class MessageRange(Strict):
    min_bytes: Positive = 512
    max_bytes_exclusive: Positive = 536870913
    factor: Annotated[int, Field(strict=True, gt=1)] = 2

    @model_validator(mode="after")
    def check_range(self):
        if self.max_bytes_exclusive <= self.min_bytes:
            raise ValueError("max_bytes_exclusive must exceed min_bytes")
        return self

    def argument(self):
        return f"{self.min_bytes},{self.max_bytes_exclusive},{self.factor}"

    def count(self):
        n, value = 0, self.min_bytes
        while value < self.max_bytes_exclusive:
            n, value = n + 1, value * self.factor
        return n


class Communication(Strict):
    collectors: Annotated[list[Literal["nccl", "custom_allreduce"]], Field(min_length=1)] = ["nccl", "custom_allreduce"]
    nccl_ops: Annotated[list[Literal["all_reduce", "all_gather", "reduce_scatter"]], Field(min_length=1)] = [
        "all_reduce",
        "all_gather",
        "reduce_scatter",
    ]
    nccl_tests_dir: Path | None = None
    message_sizes: MessageRange = Field(default_factory=MessageRange)


class Collection(Strict):
    smoke: bool = False
    limit: Positive | None = None
    shuffle: bool = False
    measure_power: bool = False
    power_test_duration_sec: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 1.0
    profile: bool = False
    debug: bool = False
    keep_csv: bool = False
    timeout_seconds: Positive = 86400
    continue_on_error: bool = True


class Runtime(Strict):
    expected_sglang_version: str | None = None
    gpu_ids: Annotated[list[Nonnegative], Field(min_length=1)] = list(range(8))


class Settings(Strict):
    model: Model
    hardware: Literal["b300_sxm"] = "b300_sxm"
    output_dir: Path
    categories: (
        Literal["all"] | Annotated[list[Literal["gemm", "moe", "attention", "communication"]], Field(min_length=1)]
    ) = "all"
    parallelism: Parallelism = Field(default_factory=Parallelism)
    sweeps: Sweeps = Field(default_factory=Sweeps)
    communication: Communication = Field(default_factory=Communication)
    collection: Collection = Field(default_factory=Collection)
    runtime: Runtime = Field(default_factory=Runtime)

    @model_validator(mode="after")
    def gpu_count(self):
        if len(set(self.runtime.gpu_ids)) != len(self.runtime.gpu_ids):
            raise ValueError("runtime.gpu_ids must be unique")
        if len(self.runtime.gpu_ids) != self.parallelism.num_gpus:
            raise ValueError("gpu_ids length must equal parallelism.num_gpus")
        return self

    def selected_categories(self):
        return list(CATEGORIES) if self.categories == "all" else list(dict.fromkeys(self.categories))


def load_settings(path: Path) -> Settings:
    settings = Settings.model_validate(yaml.safe_load(path.read_text()))
    for obj, name in ((settings, "output_dir"), (settings.model, "config"), (settings.communication, "nccl_tests_dir")):
        value = getattr(obj, name)
        if value is not None:
            setattr(obj, name, (path.parent / value.expanduser()).resolve())
    return settings


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def save_yaml(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False, allow_unicode=True))


def read_model(settings: Settings):
    local = json.loads(settings.model.config.read_text())
    cached_path = PACKAGE / "src/aisimulate_core/model_configs/nvidia--GLM-5.2-NVFP4_config.json"
    cached = json.loads(cached_path.read_text())
    # These collectors instantiate dummy-weight modules from the repository's cached config.
    # Refuse geometry / quantization drift instead of labelling a different model local GLM-5.2.
    keys = (
        "architectures",
        "hidden_size",
        "intermediate_size",
        "moe_intermediate_size",
        "n_routed_experts",
        "n_shared_experts",
        "num_experts_per_tok",
        "num_attention_heads",
        "q_lora_rank",
        "kv_lora_rank",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "v_head_dim",
        "index_n_heads",
        "index_head_dim",
        "index_topk",
        "index_topk_freq",
        "vocab_size",
        "first_k_dense_replace",
        "quantization_config",
        "max_position_embeddings",
        "num_hidden_layers",
        "num_nextn_predict_layers",
        "routed_scaling_factor",
        "scoring_func",
        "norm_topk_prob",
        "n_group",
        "topk_group",
        "rope_theta",
        "rope_scaling",
    )
    different = [key for key in keys if local.get(key) != cached.get(key)]
    if different:
        raise ValueError(
            f"Local model differs from the collector's cached model in {different}; "
            "update its model adapter before collecting"
        )
    if local["architectures"] != [ARCH]:
        raise ValueError(f"Expected {ARCH}")
    return local, cached_path


def gemm_shapes(settings: Settings, model: dict) -> list[dict]:
    """Exact independent GEMMs consumed by sdk/models/deepseek_v32.py; DSA includes its projections."""
    h, inter, moe_inter = model["hidden_size"], model["intermediate_size"], model["moe_intermediate_size"]
    shapes: dict[tuple[int, int], list[str]] = {}

    def add(n, k, role):
        shapes.setdefault((n, k), []).append(role)

    add(model["n_routed_experts"], h, "router")
    for tp in sorted(set(settings.parallelism.attention_tp_sizes)):
        if model["vocab_size"] % tp or inter % tp or model["num_attention_heads"] % tp:
            raise ValueError(f"Model dimensions are not divisible by attention TP={tp}")
        add(model["vocab_size"] // tp, h, f"lm_head_tp{tp}")
        if model.get("first_k_dense_replace", 0):
            add(2 * inter // tp, h, f"dense_gate_up_tp{tp}")
            add(h, inter // tp, f"dense_down_tp{tp}")
    for ep in sorted(set(settings.parallelism.moe_ep_sizes)):
        tp = settings.parallelism.num_gpus // ep
        if moe_inter % tp or model["n_routed_experts"] % ep:
            raise ValueError(f"Model dimensions are not divisible by MoE TP={tp}, EP={ep}")
        add(2 * moe_inter // tp, h, f"shared_gate_up_tp{tp}")
        add(h, moe_inter // tp, f"shared_down_tp{tp}")
    return [{"n": n, "k": k, "roles": roles} for (n, k), roles in sorted(shapes.items())]


def prepare_cases(settings: Settings, model: dict, root: Path) -> list[dict]:
    import shutil

    shutil.copytree(COLLECTOR / "cases/base_ops", root / "base_ops")
    # Some shared generator constants load other architecture declarations at import
    # time. Keep those declarations; COLLECTOR_MODEL_PATH scopes actual expansion.
    shutil.copytree(COLLECTOR / "cases/models", root / "models")
    source = COLLECTOR / f"cases/models/{ARCH}_cases.yaml"
    spec = yaml.safe_load(source.read_text())
    spec["model_path"], spec["model_paths"] = MODEL, [MODEL]
    rows = spec["model_case_values"]["moe"]
    moe = next(row for row in rows if row["model_path"] == MODEL).copy()
    moe.pop("model_aliases", None)
    shapes = gemm_shapes(settings, model)
    spec["model_case_values"] = {
        "moe": [moe],
        "mla_module": [
            {"model_path": MODEL, "attention_type": "dsa", "native_num_heads": model["num_attention_heads"]}
        ],
        "gemm": [
            {"model_path": MODEL, "input_feature_sizes": [row["k"]], "output_feature_sizes": [row["n"]]}
            for row in shapes
        ],
    }
    spec.pop("framework_specific_model_case_values", None)
    spec["framework_specific_op_cases"] = {"sglang": spec["framework_specific_op_cases"]["sglang"]}
    save_yaml(root / f"models/{ARCH}_cases.yaml", spec)
    # Base YAML owns sweep density; model YAML owns the correlated exact GEMM widths.
    save_yaml(
        root / "base_ops/gemm.yaml",
        {
            "schema_version": 1,
            "op": "gemm",
            "model_ops": ["gemm"],
            "all_frameworks_op_cases": {
                "gemm": {
                    "cases": [
                        {"id": "campaign_tokens", "token_counts": settings.sweeps.gemm_tokens, "feature_sizes": []}
                    ]
                }
            },
        },
    )
    moe_path = root / "base_ops/moe.yaml"
    data = yaml.safe_load(moe_path.read_text())
    common = data["common_case_values"]["moe"]
    common.update(
        token_counts=settings.sweeps.moe_tokens,
        tensor_parallel_sizes=sorted({settings.parallelism.num_gpus // ep for ep in settings.parallelism.moe_ep_sizes}),
        expert_parallel_sizes=sorted(set(settings.parallelism.moe_ep_sizes)),
        gpu_counts=[settings.parallelism.num_gpus],
        token_expert_distributions=(
            [{"name": "balanced", "power_law_alpha": 0.0}] if settings.sweeps.moe_balanced else []
        )
        + [
            {"name": "power_law", "power_law_alpha": alpha}
            for alpha in sorted(set(settings.sweeps.moe_power_law_alphas))
        ],
    )
    save_yaml(moe_path, data)
    mla_path = root / "base_ops/mla_module.yaml"
    data = yaml.safe_load(mla_path.read_text())
    attention = settings.sweeps.attention
    data["common_case_values"]["mla_module_sglang"] = {
        "module_tp_sizes": sorted(set(settings.parallelism.attention_tp_sizes)),
        "inner_sweep_head_counts": sorted(
            {model["num_attention_heads"] // tp for tp in settings.parallelism.attention_tp_sizes}
        ),
        "top_level_head_counts": [model["num_attention_heads"]],
        "module_precision_combos": [
            {"compute_dtype": "bfloat16", "kv_cache_dtype": settings.model.kv_cache_dtype, "gemm_type": "bfloat16"}
        ],
        "context_batch_sizes": attention.context_batch_sizes,
        "context_sequence_lengths": attention.context_sequence_lengths,
        "generation_batch_sizes": attention.generation_batch_sizes,
        "generation_sequence_lengths": attention.generation_sequence_lengths,
        "context": {
            "max_tokens": attention.context_max_tokens,
            "prefix_lengths": attention.prefix_lengths,
            "large_sequence_min": 8192,
            "large_sequence_max_batch_size": max(attention.context_batch_sizes),
        },
        "generation": {
            "max_tokens": attention.generation_max_tokens,
            "large_sequence_min": 8192,
            "large_sequence_max_batch_size": max(attention.generation_batch_sizes),
        },
    }
    save_yaml(mla_path, data)
    return shapes


def metadata(settings: Settings, *, gpu: bool):
    packages = {}
    for name in (
        "aisimulate",
        "sglang",
        "torch",
        "triton",
        "flashinfer-python",
        "sgl-kernel",
        "sglang-kernel",
        "transformers",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    result = {
        "python": sys.executable,
        "python_version": sys.version,
        "packages": packages,
        "gpu_ids": settings.runtime.gpu_ids,
        "gpu_probe": "not_requested",
    }
    module = importlib.util.find_spec("sglang")
    if module and module.origin:
        source = Path(module.origin)
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=source.parent, capture_output=True, text=True, check=False
        )
        result["sglang_source"] = {
            "path": str(source),
            "init_sha256": digest(source),
            "commit": commit.stdout.strip() if commit.returncode == 0 else "unknown",
        }
        if commit.returncode == 0:
            dirty = subprocess.run(
                ["git", "status", "--porcelain", "--", "."],
                cwd=source.parent,
                capture_output=True,
                text=True,
                check=False,
            )
            result["sglang_source"]["worktree_changes"] = dirty.stdout.splitlines()
    if gpu:
        import torch

        result.update(cuda=torch.version.cuda, nccl=torch.cuda.nccl.version(), gpu_probe="completed")
        result["gpus"] = [
            {
                "name": torch.cuda.get_device_name(i),
                "sm": list(torch.cuda.get_device_capability(i)),
                "memory_bytes": torch.cuda.get_device_properties(i).total_memory,
            }
            for i in range(torch.cuda.device_count())
        ]
        if len(result["gpus"]) != settings.parallelism.num_gpus:
            raise ValueError("Visible GPU count does not match YAML")
        if any("B300" not in row["name"] or row["sm"] != [10, 3] for row in result["gpus"]):
            raise ValueError("This campaign requires B300 (SM103) on every selected device")
        probe = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=uuid,name,driver_version,pci.bus_id,memory.used,utilization.gpu,clocks.sm,clocks.mem",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        result["nvidia_smi"] = probe.stdout.strip()
    return result


def inspect_case_plan(env: dict) -> dict:
    """Expand the real CPU-only generators in a fresh process, as collection workers do."""
    script = """
import dataclasses, json, os
from collector.case_generator import (
    get_gemm_case_specs, get_common_moe_test_cases,
    get_mla_module_model_specs, get_mla_module_sweep_spec,
)
from collector.model_cases import build_collection_case_plan
model = os.environ['COLLECTOR_MODEL_PATH']
plan = build_collection_case_plan(backend='sglang', model_path=model, gpu_type='b300_sxm', sm_version=103)
gemm = get_gemm_case_specs()
moe = get_common_moe_test_cases()
attention = get_mla_module_model_specs(attention_type='dsa')
assert gemm and moe and attention, 'Empty model case expansion'
assert {row.model_name for row in moe} == {model}, 'Unrelated MoE model'
assert {row.model_path for row in attention} == {model}, 'Unrelated attention model'
print(json.dumps(dict(plan=plan.to_log_dict(), gemm_cases=len(gemm),
    gemm_shapes=sorted({(row.n, row.k) for row in gemm}),
    moe_recipes=len(moe), moe_topologies=sorted({(row.tp, row.ep) for row in moe}),
    attention=dataclasses.asdict(get_mla_module_sweep_spec('sglang')))))
"""
    result = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=60, check=False
    )
    if result.returncode:
        raise ValueError(f"Case expansion failed: {result.stderr.strip()}")
    return json.loads(result.stdout)


def op_compatibility(op: str, version: str | None) -> str | None:
    from collector.framework_manifest import require_collector_runtime
    from collector.sglang.registry import REGISTRY
    from collector.version_resolver import resolve_module

    if version is None:
        return "sglang is not installed in the selected Python environment"
    try:
        require_collector_runtime("sglang", version, requested_ops={op}, model_path=MODEL)
        entry = next(entry for entry in REGISTRY if entry.op == op)
        module = resolve_module(entry, version)
        if module is None:
            return f"No collector route for {op} at {version}"
        tree = ast.parse((PACKAGE / (module.replace(".", "/") + ".py")).read_text())
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "__compat__" for target in node.targets
            ):
                requirement = Requirement(ast.literal_eval(node.value))
                if version not in requirement.specifier:
                    return f"{module} requires {requirement}; installed {version}"
    except (ValueError, RuntimeError, KeyError) as error:
        return str(error)
    return None


def build_jobs(settings: Settings, run: Path, version: str | None):
    jobs = []
    common = settings.collection
    power = (
        ["--measure_power", "--power_test_duration_sec", str(common.power_test_duration_sec)]
        if common.measure_power
        else []
    )
    for category in settings.selected_categories():
        for op in CATEGORIES[category]:
            command = [
                sys.executable,
                str(COLLECTOR / "collect.py"),
                "--backend",
                "sglang",
                "--model-path",
                MODEL,
                "--gpu",
                settings.hardware,
                "--sm",
                "103",
                "--ops",
                op,
                "--checkpoint-dir",
                "checkpoints",
                *power,
            ]
            for field, flag in (
                ("smoke", "--smoke"),
                ("shuffle", "--shuffle"),
                ("profile", "--profile"),
                ("debug", "--debug"),
                ("keep_csv", "--keep-csv"),
            ):
                if getattr(common, field):
                    command.append(flag)
            if common.limit is not None:
                command += ["--limit", str(common.limit)]
            jobs.append({"id": op, "category": category, "command": command, "blocker": op_compatibility(op, version)})
    if "communication" in settings.selected_categories():
        groups = sorted(set(settings.parallelism.attention_tp_sizes + [settings.parallelism.num_gpus]))
        for count in groups:
            if count == 1:
                continue
            if "nccl" in settings.communication.collectors:
                for op in dict.fromkeys(settings.communication.nccl_ops):
                    command = [
                        sys.executable,
                        str(COLLECTOR / "network/collect_nccl.py"),
                        "--nccl_op",
                        op,
                        "--dtype",
                        "half",
                        "--num_gpus",
                        str(count),
                        "--range",
                        settings.communication.message_sizes.argument(),
                        *power,
                    ]
                    jobs.append(
                        {
                            "id": f"nccl_{op}_{count}gpu",
                            "category": "communication",
                            "command": command,
                            "expected_min_rows": settings.communication.message_sizes.count(),
                            "blocker": None,
                        }
                    )
            if "custom_allreduce" in settings.communication.collectors:
                command = [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    f"--nproc_per_node={count}",
                    str(COLLECTOR / "network/collect_all_reduce.py"),
                    "--backend",
                    "sglang",
                    "--dtype",
                    "bfloat16",
                    "--world-size",
                    str(count),
                    "--range",
                    settings.communication.message_sizes.argument(),
                    *power,
                ]
                jobs.append(
                    {
                        "id": f"custom_allreduce_{count}gpu",
                        "category": "communication",
                        "command": command,
                        "expected_min_rows": 2 * settings.communication.message_sizes.count(),
                        "blocker": None,
                    }
                )
    for job in jobs:
        job.update(cwd=str(run / "jobs" / job["id"]), status="planned")
    return jobs


def run_command(command: list[str], cwd: Path, env: dict, timeout: int) -> int:
    """Kill the entire distributed process group on timeout or interruption."""
    with (cwd / "console.log").open("w") as log:
        process = subprocess.Popen(
            command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            return process.wait(timeout=timeout)
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            raise


def inspect_outputs(job: dict) -> tuple[list[dict], int]:
    cwd = Path(job["cwd"])
    artifacts, rows = [], 0
    for path in sorted(cwd.rglob("*_perf.*")):
        if path.suffix not in {".parquet", ".txt"}:
            continue
        if path.suffix == ".txt" and path.with_suffix(".parquet").exists():
            continue
        if path.suffix == ".parquet":
            import pyarrow.parquet as pq

            table = pq.read_table(path, columns=["latency"])
            latencies = table.column("latency").to_pylist()
        else:
            with path.open() as stream:
                latencies = [float(row["latency"]) for row in csv.DictReader(stream)]
        # The top-k calibration contract uses zero for select-all (KV <=
        # index_topk): its flat/top_last delta is exactly zero. Other kernels
        # must report a strictly positive measured latency.
        allow_zero = job.get("id") == "glm5_topk_module" and path.stem == "glm5_topk_module_perf"
        if any(
            value is None
            or not math.isfinite(float(value))
            or float(value) < 0
            or (float(value) == 0 and not allow_zero)
            for value in latencies
        ):
            raise ValueError(f"Invalid latency in {path}")
        rows += len(latencies)
        artifacts.append({"path": str(path.relative_to(cwd)), "rows": len(latencies), "sha256": digest(path)})
    if rows < job.get("expected_min_rows", 1):
        raise ValueError(
            f"Missing performance rows: observed {rows}, expected at least {job.get('expected_min_rows', 1)}"
        )
    if job["category"] != "communication":
        summaries = list(cwd.rglob("collection_summary_sglang.json"))
        if not summaries:
            raise ValueError("Collector did not publish its error summary")
        errors = sum(json.loads(path.read_text())["summary"]["total_errors"] for path in summaries)
        if errors:
            raise ValueError(f"Collector recorded {errors} case errors; see collection_summary_sglang.json")
    return artifacts, rows


def execute_jobs(jobs, settings, env, run):
    partial_grid = settings.collection.smoke or settings.collection.limit is not None
    for job in jobs:
        cwd = Path(job["cwd"])
        cwd.mkdir(parents=True, exist_ok=True)
        save_json(cwd / "command.json", job["command"])
        started = time.monotonic()
        try:
            if job["blocker"]:
                job.update(status="unsupported", error=job["blocker"])
            else:
                print(f"Collecting {job['id']} -> {cwd / 'console.log'}", flush=True)
                job["returncode"] = run_command(job["command"], cwd, env, settings.collection.timeout_seconds)
                if job["returncode"]:
                    raise RuntimeError(f"Collector exited {job['returncode']}; see console.log")
                job["artifacts"], job["rows"] = inspect_outputs(job)
                job["status"] = "success"
        except KeyboardInterrupt:
            job.update(status="interrupted", error="Interrupted by user")
            save_json(run / "summary.json", {"complete": False, "partial_grid": partial_grid, "jobs": jobs})
            return 130
        except (OSError, ValueError, RuntimeError, KeyError, subprocess.TimeoutExpired) as error:
            job.update(status="failed", error=str(error))
        job["elapsed_seconds"] = time.monotonic() - started
        save_json(cwd / "result.json", job)
        save_json(
            run / "summary.json",
            {
                "complete": all(item["status"] == "success" for item in jobs),
                "partial_grid": partial_grid,
                "jobs": jobs,
            },
        )
        if job["status"] != "success" and not settings.collection.continue_on_error:
            break
    complete = all(job["status"] == "success" for job in jobs)
    print(f"{'Completed' if complete else 'Incomplete'}: {run / 'summary.json'}")
    return 0 if complete else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", type=Path, required=True)
    parser.add_argument(
        "--plan-only", action="store_true", help="Save scoped cases and commands; no CUDA initialization or benchmarks"
    )
    args = parser.parse_args(argv)
    run = None
    try:
        settings = load_settings(args.config.resolve())
        model, cached_path = read_model(settings)
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, settings.runtime.gpu_ids))
        # Do not inherit ad-hoc filters or another campaign's identity/runtime override.
        for key in os.environ:
            if key.startswith(("AISIM_COLLECTOR_", "COLLECTOR_", "AIC_")) and os.environ.get(key):
                raise ValueError(f"Unset {key} before starting this campaign")
        runtime = metadata(settings, gpu=False)
        version = runtime["packages"]["sglang"]
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
        run = settings.output_dir / "sglang" / (version or "not-installed") / settings.hardware / stamp
        run.mkdir(parents=True, exist_ok=False)
        save_yaml(run / "config.resolved.yaml", settings.model_dump(mode="json"))
        (run / "config.input.yaml").write_bytes(args.config.read_bytes())
        save_json(run / "runtime.json", runtime)
        expected = settings.runtime.expected_sglang_version
        if expected is not None and expected != version:
            raise ValueError(f"Expected SGLang {expected}, installed {version}")
        shapes = prepare_cases(settings, model, run / "cases")
        env = os.environ.copy()
        env.update(
            AISIM_COLLECTOR_CASES_ROOT=str(run / "cases"),
            COLLECTOR_MODEL_PATH=MODEL,
            AIC_COLLECT_GEMM_TYPES="bfloat16",
            PYTHONPATH=str(PACKAGE) + os.pathsep + env.get("PYTHONPATH", ""),
        )
        if settings.communication.nccl_tests_dir:
            env["PATH"] = str(settings.communication.nccl_tests_dir) + os.pathsep + env.get("PATH", "")
        sys.path.insert(0, str(PACKAGE))
        jobs = build_jobs(settings, run, version)
        expanded = inspect_case_plan(env)
        expected_shapes = sorted((row["n"], row["k"]) for row in shapes)
        if [tuple(row) for row in expanded["gemm_shapes"]] != expected_shapes:
            raise ValueError("Expanded GEMM cases differ from GLM-5.2's exact shapes")
        sources = (
            list(COLLECTOR.rglob("*.py"))
            + list(COLLECTOR.rglob("*.yaml"))
            + list(Path(__file__).parent.glob("*.py"))
            + [cached_path]
        )
        save_json(run / "source_sha256.json", {str(path.relative_to(ROOT)): digest(path) for path in sorted(sources)})
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False)
        save_json(
            run / "provenance.json",
            {
                "aisimulate_commit": revision.stdout.strip() if revision.returncode == 0 else "unknown",
                "model_config_path": str(settings.model.config),
                "model_config_sha256": digest(settings.model.config),
                "cases_sha256": {
                    str(path.relative_to(run)): digest(path) for path in sorted((run / "cases").rglob("*.yaml"))
                },
                "environment": {
                    key: env[key]
                    for key in (
                        "CUDA_VISIBLE_DEVICES",
                        "AISIM_COLLECTOR_CASES_ROOT",
                        "COLLECTOR_MODEL_PATH",
                        "AIC_COLLECT_GEMM_TYPES",
                    )
                },
            },
        )
        save_json(
            run / "plan.json",
            {
                "model": MODEL,
                "gemm_shapes": shapes,
                "expanded": expanded,
                "jobs": jobs,
                "scope": "GLM-5.2 collector operations within configured grids; excludes serving/scheduler timing",
                "partial_grid": settings.collection.smoke or settings.collection.limit is not None,
            },
        )
        print(f"Plan: {run / 'plan.json'} ({len(jobs)} jobs)")
        for job in jobs:
            if job["blocker"]:
                print(f"UNSUPPORTED {job['id']}: {job['blocker']}")
        if args.plan_only:
            return 0  # Planning success does not mean the runtime is supported.
        runtime = metadata(settings, gpu=True)
        save_json(run / "runtime.json", runtime)
        return execute_jobs(jobs, settings, env, run)
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired, yaml.YAMLError) as error:
        if run is not None:
            save_json(run / "summary.json", {"complete": False, "error": str(error)})
        print(f"Collection error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
