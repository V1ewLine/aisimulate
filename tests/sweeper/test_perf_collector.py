# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Campaign scope, failure accounting, and process execution without GPU access."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "collect_perf", ROOT / "examples/sweeper/tools/perf_collector/collect_perf.py"
)
TOOL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TOOL
SPEC.loader.exec_module(TOOL)
pytestmark = pytest.mark.unit


@pytest.fixture
def settings(tmp_path):
    return TOOL.Settings.model_validate(
        {
            "model": {
                "config": str(TOOL.PACKAGE / "src/aisimulate_core/model_configs/nvidia--GLM-5.2-NVFP4_config.json")
            },
            "output_dir": str(tmp_path),
            "parallelism": {"num_gpus": 8, "attention_tp_sizes": [2, 4], "moe_ep_sizes": [1, 4]},
            "sweeps": {
                "gemm_tokens": [1, 32],
                "moe_tokens": [8, 64],
                "moe_balanced": False,
                "attention": {
                    "context_batch_sizes": [1, 2],
                    "context_sequence_lengths": [512],
                    "generation_batch_sizes": [4, 32],
                    "generation_sequence_lengths": [32768],
                    "prefix_lengths": [0, 19968],
                },
            },
        }
    )


def test_real_generators_use_only_target_shapes(settings, tmp_path):
    model, _ = TOOL.read_model(settings)
    root = tmp_path / "case snapshot with spaces"
    shapes = TOOL.prepare_cases(settings, model, root)
    env = {
        **os.environ,
        "PYTHONPATH": str(TOOL.PACKAGE),
        "AISIM_COLLECTOR_CASES_ROOT": str(root),
        "COLLECTOR_MODEL_PATH": TOOL.MODEL,
    }
    expanded = TOOL.inspect_case_plan(env)
    assert expanded["gemm_cases"] == len(shapes) * 2
    assert expanded["gemm_shapes"] == [[row["n"], row["k"]] for row in shapes]
    assert expanded["moe_topologies"] == [[2, 4], [8, 1]]
    assert expanded["moe_recipes"] == 2
    assert expanded["attention"]["module_tp_sizes"] == [2, 4]
    assert expanded["attention"]["context_prefix_lengths"] == [0, 19968]
    assert expanded["attention"]["generation_batch_sizes"] == [4, 32]
    assert expanded["attention"]["module_precision_combos"] == [["bfloat16", "fp8", "bfloat16"]]
    assert set(expanded["plan"]["ops"]) == {op for ops in TOOL.CATEGORIES.values() for op in ops}
    # Exercise the actual CLI with the same case-root override, not just the wrapper.
    process = subprocess.run(
        [
            sys.executable,
            str(TOOL.COLLECTOR / "collect.py"),
            "--backend",
            "sglang",
            "--model-path",
            TOOL.MODEL,
            "--gpu",
            "b300_sxm",
            "--ops",
            "gemm",
            "--plan-only",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert process.returncode == 0, process.stderr
    assert str(root) in process.stdout


@pytest.mark.parametrize(
    "updates",
    [
        {"categories": ["kda"]},
        {"parallelism": {"attention_tp_sizes": [3]}},
        {"runtime": {"gpu_ids": [0, 0]}},
        {"collection": {"limit": 0}},
        {"communication": {"message_sizes": {"factor": 1}}},
        {"communication": {"message_sizes": {"min_bytes": 512, "max_bytes_exclusive": 512}}},
        {"sweeps": {"gemm_tokens": []}},
        {"collection": {"limti": 5}},
    ],
)
def test_invalid_config_rejected(settings, updates):
    with pytest.raises(ValidationError):
        TOOL.Settings.model_validate({**settings.model_dump(), **updates})


def test_paths_resolved_relative_to_yaml(settings, tmp_path):
    data = settings.model_dump(mode="json")
    data.update(output_dir="relative output")
    data["model"]["config"] = "model/config.json"
    file = tmp_path / "test.yaml"
    file.write_text(yaml.safe_dump(data))
    resolved = TOOL.load_settings(file)
    assert resolved.output_dir == tmp_path / "relative output"
    assert resolved.model.config == tmp_path / "model/config.json"


def test_model_drift_rejected(settings, tmp_path):
    data = json.loads(settings.model.config.read_text())
    data["index_topk_freq"] = 1
    path = tmp_path / "model.json"
    path.write_text(json.dumps(data))
    settings.model.config = path
    with pytest.raises(ValueError, match="index_topk_freq"):
        TOOL.read_model(settings)


def test_job_arguments_and_category_isolation(settings, monkeypatch, tmp_path):
    monkeypatch.setattr(TOOL, "op_compatibility", lambda *args: None)
    settings.categories = ["moe"]
    settings.collection.limit = 3
    jobs = TOOL.build_jobs(settings, tmp_path, "0.5.19")
    assert [job["id"] for job in jobs] == ["moe"]
    assert jobs[0]["command"][-2:] == ["--limit", "3"]
    settings.categories = ["communication"]
    jobs = TOOL.build_jobs(settings, tmp_path, "0.5.19")
    assert len(jobs) == 12
    assert all(job["category"] == "communication" for job in jobs)
    assert all("--standalone" in job["command"] for job in jobs if job["id"].startswith("custom"))
    assert all("--limit" not in job["command"] for job in jobs)


def make_job(tmp_path, *, errors=0, latency=0.25):
    cwd = tmp_path / "job"
    cwd.mkdir()
    (cwd / "gemm_perf.txt").write_text(f"latency,m,n,k\n{latency},1,32,32\n")
    (cwd / "collection_summary_sglang.json").write_text(json.dumps({"summary": {"total_errors": errors}}))
    return {
        "id": "gemm",
        "cwd": str(cwd),
        "category": "gemm",
        "blocker": None,
        "status": "planned",
        "command": [sys.executable, "-c", "pass"],
    }


def test_zero_exit_with_case_errors_is_not_success(settings, tmp_path):
    job = make_job(tmp_path, errors=2)
    assert TOOL.execute_jobs([job], settings, os.environ.copy(), tmp_path) == 1
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["complete"] is False
    assert summary["jobs"][0]["status"] == "failed"
    assert "2 case errors" in summary["jobs"][0]["error"]
    assert (Path(job["cwd"]) / "gemm_perf.txt").exists()


def test_success_records_rows_and_hashes(settings, tmp_path):
    job = make_job(tmp_path)
    assert TOOL.execute_jobs([job], settings, os.environ.copy(), tmp_path) == 0
    assert job["rows"] == 1
    assert len(job["artifacts"][0]["sha256"]) == 64


@pytest.mark.parametrize("latency", ["nan", "inf", "0", "-1"])
def test_invalid_timings_fail(tmp_path, latency):
    job = make_job(tmp_path, latency=latency)
    with pytest.raises(ValueError, match="Invalid latency"):
        TOOL.inspect_outputs(job)


def test_empty_or_missing_summary_fails(tmp_path):
    job = make_job(tmp_path)
    path = Path(job["cwd"])
    (path / "collection_summary_sglang.json").unlink()
    with pytest.raises(ValueError, match="error summary"):
        TOOL.inspect_outputs(job)
    (path / "gemm_perf.txt").write_text("latency,m,n,k\n")
    with pytest.raises(ValueError, match="Missing performance rows"):
        TOOL.inspect_outputs(job)


def test_unsupported_is_recorded_without_launch(settings, tmp_path, monkeypatch):
    job = make_job(tmp_path)
    job["blocker"] = "requires another SGLang version"
    monkeypatch.setattr(TOOL, "run_command", lambda *args: pytest.fail("unsupported job was launched"))
    assert TOOL.execute_jobs([job], settings, os.environ.copy(), tmp_path) == 1
    assert job["status"] == "unsupported"


def test_timeout_is_failure(settings, tmp_path):
    job = make_job(tmp_path)
    job["command"] = [sys.executable, "-c", "import time; time.sleep(30)"]
    settings.collection.timeout_seconds = 1
    assert TOOL.execute_jobs([job], settings, os.environ.copy(), tmp_path) == 1
    assert job["status"] == "failed"
    assert "timed out" in job["error"]


def test_communication_partial_grid_fails(tmp_path):
    job = make_job(tmp_path)
    job.update(category="communication", expected_min_rows=4)
    with pytest.raises(ValueError, match="expected at least 4"):
        TOOL.inspect_outputs(job)


def test_existing_version_gate_is_preserved(monkeypatch):
    monkeypatch.syspath_prepend(str(TOOL.PACKAGE))
    assert TOOL.op_compatibility("gemm", "0.5.19") is None
    assert "requires exactly 0.5.19" in TOOL.op_compatibility("gemm", "0.5.14")
    assert "requires exactly 0.5.19" in TOOL.op_compatibility("gemm", "0.5.20")


def test_topk_select_all_zero_calibration_is_accepted(tmp_path):
    job = make_job(tmp_path, latency=0)
    cwd = Path(job["cwd"])
    (cwd / "gemm_perf.txt").rename(cwd / "glm5_topk_module_perf.txt")
    job.update(id="glm5_topk_module", category="attention")
    _, rows = TOOL.inspect_outputs(job)
    assert rows == 1


def test_parquet_output_checked_without_double_counting_csv(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    job = make_job(tmp_path)
    pq.write_table(pa.table({"latency": [0.1, 0.2]}), Path(job["cwd"]) / "gemm_perf.parquet")
    artifacts, rows = TOOL.inspect_outputs(job)
    assert rows == 2
    assert [row["path"] for row in artifacts] == ["gemm_perf.parquet"]


def test_invalid_case_root_never_falls_back(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(TOOL.PACKAGE))
    from collector.case_paths import collection_case_root

    monkeypatch.delenv("AISIM_COLLECTOR_CASES_ROOT", raising=False)
    assert collection_case_root() == TOOL.COLLECTOR / "cases"
    monkeypatch.setenv("AISIM_COLLECTOR_CASES_ROOT", str(tmp_path))
    with pytest.raises(ValueError, match="missing base_ops"):
        collection_case_root()
    (tmp_path / "base_ops").mkdir()
    with pytest.raises(ValueError, match="missing models"):
        collection_case_root()
