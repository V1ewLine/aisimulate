# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Search aggregated SGLang configurations by running the public predict CLI.

Run with the Python interpreter from the environment containing AISimulate.
This is a finite grid search; its winner is the best observed prediction under
strict request-level P50 SLOs, not a claim about measured serving performance.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import itertools
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from copy import deepcopy
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import Field, field_validator

from aisimulate.config.cli import CorePredictionConfig
from aisimulate.config.common import PositiveFiniteFloat, PositiveStrictInt, StrictModel, load_yaml

METRICS = (
    "median_ttft_ms",
    "median_tpot_ms",
    "input_throughput_tok_s",
    "output_throughput_tok_s",
    "total_throughput_tok_s",
)
COLUMNS = (
    "candidate",
    "status",
    "concurrency",
    "attention_data",
    "moe_expert",
    "chunked_prefill_size",
    "max_prefill_tokens",
    "max_batched_tokens",
    "max_sequences",
    "memory_fraction",
    *METRICS,
    "all_requests_median_ttft_ms",
    "ttft_scope",
    "ttft_samples",
    "tpot_samples",
    "completed_requests",
    "eligible",
    "elapsed_seconds",
    "error",
)


class SearchGrid(StrictModel):
    concurrency: list[PositiveStrictInt] | None = Field(default=None, min_length=1)
    dp_ep: list[tuple[PositiveStrictInt, PositiveStrictInt]] | None = Field(default=None, min_length=1)
    chunked_prefill_size: list[PositiveStrictInt] | None = Field(default=None, min_length=1)
    max_prefill_tokens: list[PositiveStrictInt] | None = Field(default=None, min_length=1)
    max_sequences: list[PositiveStrictInt] | None = Field(default=None, min_length=1)
    memory_fraction: list[Annotated[float, Field(strict=True, gt=0, le=1, allow_inf_nan=False)]] | None = Field(
        default=None, min_length=1
    )

    @field_validator("*")
    @classmethod
    def unique_choices(cls, value):
        if value is not None and len(set(value)) != len(value):
            raise ValueError("search choices must be unique")
        return value


class SLO(StrictModel):
    ttft_p50_ms: PositiveFiniteFloat = 15000.0
    tpot_p50_ms: PositiveFiniteFloat = 33.0


class RunOptions(StrictModel):
    jobs: PositiveStrictInt = 1
    timeout_seconds: PositiveFiniteFloat = 300.0


class MetricOptions(StrictModel):
    ttft_scope: Literal["first_batch", "all"] = "first_batch"


class WorkloadOptions(StrictModel):
    requests_per_concurrency: PositiveStrictInt | None = None


class TuningConfig(StrictModel):
    prediction: CorePredictionConfig
    output_dir: Annotated[str, Field(strict=True, min_length=1)]
    search: SearchGrid = Field(default_factory=SearchGrid)
    slo: SLO = Field(default_factory=SLO)
    run: RunOptions = Field(default_factory=RunOptions)
    metrics: MetricOptions = Field(default_factory=MetricOptions)
    workload: WorkloadOptions = Field(default_factory=WorkloadOptions)


def relative_path(value, directory):
    return str((directory / Path(value).expanduser()).resolve())


def load_settings(path):
    """Resolve file references against this YAML, independent of the shell cwd."""
    path = path.expanduser().resolve()
    raw = load_yaml(path)

    def mapping(value):
        return value if isinstance(value, dict) else {}

    prediction = raw.get("prediction")
    if isinstance(prediction, dict):
        source = mapping(mapping(prediction.get("traffic")).get("source"))
        if source.get("type") == "trace" and isinstance(source.get("paths"), list):
            source["paths"] = [relative_path(p, path.parent) if isinstance(p, str) else p for p in source["paths"]]
        engine = mapping(prediction.get("engine"))
        model = engine.get("model")
        if isinstance(model, str) and (model.startswith((".", "~", "/")) or (path.parent / model).is_dir()):
            engine["model"] = relative_path(model, path.parent)
        settings = [engine]
        for worker in mapping(engine.get("workers")).values():
            if isinstance(worker, dict):
                settings.append(mapping(worker.get("timing")))
        for setting in settings:
            if isinstance(setting.get("systems_paths"), list):
                setting["systems_paths"] = [
                    relative_path(p, path.parent) if isinstance(p, str) and p.lower() != "default" else p
                    for p in setting["systems_paths"]
                ]
            for field in ("systems_path", "fpm_parquet_path"):
                if isinstance(setting.get(field), str):
                    setting[field] = relative_path(setting[field], path.parent)
    config = TuningConfig.model_validate(raw)
    config.output_dir = relative_path(config.output_dir, path.parent)
    return config


def configure_args(args, config):
    grid = config.search
    args.prediction = config.prediction
    args.output_dir = Path(config.output_dir)
    args.concurrencies = grid.concurrency or [config.prediction.traffic.load.concurrency]
    args.dp_ep = grid.dp_ep
    args.chunk_sizes = grid.chunked_prefill_size
    args.max_prefill_tokens = grid.max_prefill_tokens
    args.max_sequences = grid.max_sequences
    args.memory_fractions = grid.memory_fraction
    args.ttft_p50_ms, args.tpot_p50_ms = config.slo.ttft_p50_ms, config.slo.tpot_p50_ms
    args.jobs, args.timeout = config.run.jobs, config.run.timeout_seconds
    args.ttft_scope = config.metrics.ttft_scope
    args.requests_per_concurrency = config.workload.requests_per_concurrency
    return args


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("-c", "--config", type=Path, required=True, help="search YAML (prediction, search, output_dir)")
    p.add_argument(
        "--resume", action="store_true", help="reuse successful results; retry failed/interrupted candidates"
    )
    p.add_argument("--dry-run", action="store_true", help="validate grid and print counts without running predictions")
    return p


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_write(path, text):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_json(path, value):
    atomic_write(path, json.dumps(value, indent=2, allow_nan=False) + "\n")


def prepare_base(prediction, prefix_counts=()):
    if isinstance(prediction, (str, Path)):
        prediction = CorePredictionConfig.from_yaml(prediction)
    base = prediction.model_dump(mode="json", exclude_none=True)
    engine, traffic = base["engine"], base["traffic"]
    if engine["backend"] != "sglang" or engine["mode"] != "aggregated":
        raise ValueError("this script requires aggregated SGLang predict YAML")
    if traffic["load"]["type"] != "concurrency" or traffic.get("stop", {}).get("max_virtual_time_seconds"):
        raise ValueError("use concurrency traffic without a virtual-time cutoff")
    parallel = engine["workers"]["aggregated"]["parallelism"]
    if any(parallel.get(key, 1) != 1 for key in ("replicas", "pipeline", "prefill_context", "decode_context")):
        raise ValueError("this script requires one replica, PP=1, and CP/DCP=1")
    if not engine.get("backend_version"):
        raise ValueError("pin engine.backend_version in the base YAML")
    if Path(engine["model"]).is_dir():
        engine["model"] = str(Path(engine["model"]).resolve())
    source = traffic["source"]
    if source["type"] == "trace":
        if source["format"] != "mooncake":
            raise ValueError("only flat Mooncake traces and fixed-count synthetic traffic are supported")
        trace = Path(source["paths"][0]).resolve()
        source["paths"] = [str(trace)]
        totals = {"requests": 0, "input_tokens": 0, "output_tokens": 0}
        prefixes = {}
        prefix_counts = set(prefix_counts)
        digest = hashlib.sha256()
        with trace.open("rb") as stream:
            for line in stream:
                digest.update(line)
                if not line.strip():
                    continue
                row = json.loads(line)
                for field in ("input_length", "output_length"):
                    if type(row.get(field)) is not int or row[field] < (2 if field == "output_length" else 1):
                        raise ValueError("trace needs flat input_length/output_length records with output_length >= 2")
                if "session_id" in row:
                    raise ValueError("session traces are not supported by this script")
                totals["requests"] += 1
                totals["input_tokens"] += row["input_length"]
                totals["output_tokens"] += row["output_length"]
                if totals["requests"] in prefix_counts:
                    prefixes[totals["requests"]] = {**totals, "trace_sha256": digest.hexdigest()}
        totals["trace_sha256"] = digest.hexdigest()
        if prefix_counts:
            totals["prefixes"] = prefixes
    elif source["type"] == "synthetic" and traffic.get("stop", {}).get("requests"):
        count = traffic["stop"]["requests"]
        if source["output_tokens"] < 2:
            raise ValueError("output_tokens must be >= 2 to measure request TPOT")
        totals = {
            "requests": count,
            "input_tokens": count * source["input_tokens"],
            "output_tokens": count * source["output_tokens"],
        }
    else:
        raise ValueError("use a flat Mooncake trace or synthetic traffic with stop.requests")
    if not totals["requests"]:
        raise ValueError("trace is empty")
    return base, totals


def search_space(base, args):
    worker = base["engine"]["workers"]["aggregated"]
    parallel, scheduler, capacity = worker["parallelism"], worker["scheduler"], worker["kv_cache"]["capacity"]
    if args.memory_fractions and capacity["type"] != "default":
        raise ValueError("search.memory_fraction requires default KV capacity estimation")
    if args.memory_fractions and any(value > 1 for value in args.memory_fractions):
        raise ValueError("memory fractions must be <= 1")
    layouts = args.dp_ep or [(parallel["attention_data"], parallel["moe_expert"])]
    total = parallel["tensor"] * parallel["attention_data"]
    if any(total % dp or total % ep for dp, ep in layouts):
        raise ValueError(f"each DP and EP must divide the base GPU count ({total})")
    if parallel["moe_tensor"] * parallel["moe_expert"] != total:
        raise ValueError("base Attention and MoE layouts must use the same GPU count")
    space = {
        "concurrency": args.concurrencies,
        "layout": layouts,
        "chunked_prefill_size": args.chunk_sizes or [scheduler.get("chunked_prefill_size")],
        "max_prefill_tokens": args.max_prefill_tokens or [scheduler.get("max_prefill_tokens")],
        "max_sequences": args.max_sequences or [scheduler["max_sequences"]],
        "memory_fraction": args.memory_fractions or [capacity.get("memory_fraction")],
    }
    if any(space[key] == [None] for key in ("chunked_prefill_size", "max_prefill_tokens")):
        raise ValueError("set chunked_prefill_size and max_prefill_tokens in search or prediction scheduler")
    return space


def prepare_workloads(base, totals, space, args):
    workloads = {}
    for concurrency in space["concurrency"]:
        traffic = deepcopy(base["traffic"])
        expected = {k: v for k, v in totals.items() if k != "prefixes"}
        if args.requests_per_concurrency is not None:
            count = concurrency * args.requests_per_concurrency
            if traffic["source"]["type"] == "trace":
                if count > totals["requests"]:
                    raise ValueError(
                        f"C={concurrency} requires {count} requests but the trace has {totals['requests']}"
                    )
                expected = totals["prefixes"][count]
                traffic["source"]["paths"] = [str(args.output_dir / "workloads" / f"requests-{count}.jsonl")]
            else:
                traffic["stop"] = {"requests": count}
                expected = {
                    "requests": count,
                    "input_tokens": count * traffic["source"]["input_tokens"],
                    "output_tokens": count * traffic["source"]["output_tokens"],
                }
        if args.ttft_scope == "first_batch" and expected["requests"] < concurrency:
            raise ValueError(f"first_batch TTFT needs at least C={concurrency} requests")
        workloads[concurrency] = {"traffic": traffic, "totals": expected}
    return workloads


def materialize_workloads(base, workloads):
    """Stream each selected trace prefix once per campaign, shared by its candidates."""
    source = base["traffic"]["source"]
    if source["type"] != "trace":
        return
    destinations = {Path(w["traffic"]["source"]["paths"][0]): w["totals"] for w in workloads.values()}
    destinations.pop(Path(source["paths"][0]), None)
    if not destinations:
        return
    largest = max(t["requests"] for t in destinations.values())
    from contextlib import ExitStack

    with ExitStack() as stack:
        streams = []
        for path, totals in destinations.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            streams.append((path, totals, stack.enter_context(path.with_suffix(".tmp").open("wb"))))
        count = 0
        digest = hashlib.sha256()
        with Path(source["paths"][0]).open("rb") as source_stream:
            for line in source_stream:
                digest.update(line)
                for _, totals, stream in streams:
                    if count < totals["requests"]:
                        stream.write(line)
                if line.strip():
                    count += 1
                    for _, totals, _ in streams:
                        if count == totals["requests"] and digest.hexdigest() != totals["trace_sha256"]:
                            raise ValueError("trace changed during workload preparation")
                if count == largest:
                    break
        if count < largest:
            raise ValueError("trace became shorter during workload preparation")
    for path in destinations:
        path.with_suffix(".tmp").replace(path)


def candidates(base, space, workloads=None):
    for values in itertools.product(*space.values()):
        knobs = dict(zip(space, values, strict=True))
        config = deepcopy(base)
        if workloads is not None:
            config["traffic"] = deepcopy(workloads[knobs["concurrency"]]["traffic"])
        worker = config["engine"]["workers"]["aggregated"]
        parallel, scheduler = worker["parallelism"], worker["scheduler"]
        total = parallel["tensor"] * parallel["attention_data"]
        dp, ep = knobs.pop("layout")
        knobs.update(attention_data=dp, moe_expert=ep, max_batched_tokens=knobs["chunked_prefill_size"] // dp)
        parallel.update(tensor=total // dp, attention_data=dp, moe_tensor=total // ep, moe_expert=ep)
        config["traffic"]["load"]["concurrency"] = knobs["concurrency"]
        for key in ("chunked_prefill_size", "max_prefill_tokens", "max_batched_tokens", "max_sequences"):
            scheduler[key] = knobs[key]
        if knobs["memory_fraction"] is not None:
            worker["kv_cache"]["capacity"]["memory_fraction"] = knobs["memory_fraction"]
        error = ""
        page = worker["kv_cache"].get("block_size", 1)
        if knobs["chunked_prefill_size"] % (dp * page):
            error = "chunk must be divisible by DP * KV page size"
        elif knobs["max_prefill_tokens"] < knobs["max_batched_tokens"]:
            error = "prefill budget is smaller than the per-rank chunk"
        candidate = hashlib.sha256(canonical(config).encode()).hexdigest()[:16]
        yield candidate, config, knobs, error


def score(report, totals, ttft_limit, tpot_limit, ttft_scope="all", concurrency=None):
    if not isinstance(report, dict):
        raise ValueError("prediction.json must contain a report object")
    for field in (
        *METRICS,
        "num_requests",
        "completed_requests",
        "num_ttft_samples",
        "num_tpot_samples",
        "total_input_tokens",
        "total_output_tokens",
    ):
        value = report.get(field)
        if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"invalid or missing report metric: {field}")
    for field in ("num_requests", "completed_requests", "num_ttft_samples", "num_tpot_samples"):
        if report[field] != totals["requests"]:
            raise ValueError(
                f"incomplete workload or latency samples: {field}={report[field]}, expected {totals['requests']}"
            )
    for kind in ("input", "output"):
        if report[f"total_{kind}_tokens"] != totals[f"{kind}_tokens"]:
            raise ValueError(f"{kind} token total differs from the full workload")
    requests = report.get("per_request")
    if not isinstance(requests, list) or len(requests) != totals["requests"]:
        raise ValueError("complete per_request capture is required")
    ttfts, tpots, first_ttfts = [], [], []
    input_tokens = output_tokens = 0
    for request in requests:
        if not isinstance(request, dict) or request.get("terminal_status") != "completed":
            raise ValueError("per_request contains an incomplete request")
        for field in ("arrival_time_ms", "first_token_ms", "last_token_ms", "ttft_ms", "input_length", "output_length"):
            value = request.get(field)
            if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid per-request field: {field}")
        if request["output_length"] < 2 or request["last_token_ms"] < request["first_token_ms"]:
            raise ValueError("per_request has invalid decode duration or output length")
        ttfts.append(request["ttft_ms"])
        tpots.append((request["last_token_ms"] - request["first_token_ms"]) / (request["output_length"] - 1))
        if request["arrival_time_ms"] == 0:
            first_ttfts.append(request["ttft_ms"])
        input_tokens += request["input_length"]
        output_tokens += request["output_length"]
    if input_tokens != totals["input_tokens"] or output_tokens != totals["output_tokens"]:
        raise ValueError("per-request token totals differ from the workload")
    if ttft_scope == "first_batch" and len(first_ttfts) != concurrency:
        raise ValueError(f"expected {concurrency} initial arrivals, got {len(first_ttfts)}")
    selected = first_ttfts if ttft_scope == "first_batch" else ttfts
    ttft, tpot = statistics.median(selected), statistics.median(tpots)
    if not math.isclose(
        report["input_throughput_tok_s"] + report["output_throughput_tok_s"], report["total_throughput_tok_s"]
    ):
        raise ValueError("total throughput differs from input + output throughput")
    return {
        **{key: report[key] for key in (*METRICS, "completed_requests")},
        "median_ttft_ms": ttft,
        "median_tpot_ms": tpot,
        "all_requests_median_ttft_ms": statistics.median(ttfts),
        "ttft_scope": ttft_scope,
        "ttft_samples": len(selected),
        "tpot_samples": len(tpots),
        "eligible": ttft < ttft_limit and tpot < tpot_limit,
    }


def rank_key(row):
    return (-row["output_throughput_tok_s"], row["median_tpot_ms"], row["median_ttft_ms"], row["candidate"])


def runtime_identity(base):
    # Hash code once per campaign; data metadata detects normal profile updates
    # without reading every parquet file for every candidate.
    digest = hashlib.sha256()
    for module in ("aisimulate", "aisimulate_core"):
        root = Path(importlib.util.find_spec(module).origin).parent
        for path in sorted(root.rglob("*.py")):
            digest.update(str(path).encode())
            digest.update(path.read_bytes())
    native = Path(importlib.util.find_spec("aisimulate._runtime").origin)
    core = Path(importlib.util.find_spec("aisimulate_core").origin).parent
    roots = [core / "systems"]
    engine = base["engine"]
    timing = engine["workers"]["aggregated"]["timing"]
    roots.extend(
        Path(value)
        for settings in (engine, timing)
        for value in settings.get("systems_paths", [])
        if value != "default"
    )
    if Path(engine["model"]).is_dir():
        roots.append(Path(engine["model"]))
    data = hashlib.sha256()
    for root in roots:
        for path in sorted(root.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                stat = path.stat()
                data.update(canonical([str(path), stat.st_size, stat.st_mtime_ns]).encode())
    return {
        "python": sys.executable,
        "version": importlib.metadata.version("aisimulate"),
        "native_sha256": file_hash(native),
        "python_code_sha256": digest.hexdigest(),
        "performance_data_metadata_sha256": data.hexdigest(),
        "script_sha256": file_hash(__file__),
    }


def run_command(command, log, timeout, stop):
    with subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True) as process:
        deadline = time.monotonic() + timeout
        while process.poll() is None:
            if stop.is_set() or time.monotonic() >= deadline:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise RuntimeError("interrupted" if stop.is_set() else f"timeout after {timeout:g}s")
            try:
                process.wait(timeout=0.2)
            except subprocess.TimeoutExpired:
                pass
        return process.returncode


def run_candidate(item, output, totals, args, stop):
    candidate, config, knobs, error = item
    directory = output / "candidates" / candidate
    directory.mkdir(parents=True, exist_ok=True)
    row = {"candidate": candidate, **knobs, "status": "invalid", "eligible": False, "error": error}
    config_path = directory / "config.yaml"
    atomic_write(config_path, yaml.safe_dump(config, sort_keys=False))
    if error:
        write_json(directory / "result.json", row)
        return row
    report_path = directory / "prediction" / "prediction.json"
    result_path = directory / "result.json"
    if args.resume and result_path.exists() and report_path.exists():
        try:
            cached = json.loads(result_path.read_text())
            if isinstance(cached, dict) and cached.get("status") == "ok":
                metrics = score(
                    json.loads(report_path.read_text()),
                    totals,
                    args.ttft_p50_ms,
                    args.tpot_p50_ms,
                    args.ttft_scope,
                    knobs["concurrency"],
                )
                return {**cached, **metrics}
        except (ValueError, OSError, TypeError):
            pass  # A partial/corrupt result is retried, never ranked as successful.
    start = time.monotonic()
    try:
        CorePredictionConfig.model_validate(config)
        command = [
            sys.executable,
            "-m",
            "aisimulate",
            "predict",
            "--stack",
            "engine",
            "-c",
            str(config_path),
            "--output-dir",
            str(report_path.parent),
            "--overwrite",
            "--format",
            "json",
            "--capture-per-request",
        ]
        write_json(directory / "command.json", command)
        with (directory / "run.log").open("w") as log:
            code = run_command(command, log, args.timeout, stop)
        if code:
            raise RuntimeError(f"predict exited {code}; see {directory / 'run.log'}")
        metrics = score(
            json.loads(report_path.read_text()),
            totals,
            args.ttft_p50_ms,
            args.tpot_p50_ms,
            args.ttft_scope,
            knobs["concurrency"],
        )
        row.update(metrics, status="ok", error="")
    except (ValueError, OSError, RuntimeError) as failure:
        row.update(status="error", error=str(failure))
    row["elapsed_seconds"] = round(time.monotonic() - start, 3)
    write_json(result_path, row)
    return row


def prepare_output(output, manifest, resume):
    path = output / "manifest.json"
    if resume:
        if not path.exists() or json.loads(path.read_text()) != json.loads(canonical(manifest)):
            raise ValueError(
                "resume requires the same config, workload, grid, SLOs, code and data; use a new output directory"
            )
    else:
        if any(p.name != ".lock" for p in output.iterdir()):
            raise ValueError("output directory is not empty; use --resume or a new directory")
        write_json(path, manifest)


def search(base, workloads, space, args, output):
    count = math.prod(len(values) for values in space.values())
    rows, best = [], None
    stop = threading.Event()
    iterator = iter(candidates(base, space, workloads))
    for name in ("best.yaml", "best.json", "summary.json", "ranking.csv"):
        (output / name).unlink(missing_ok=True)
    executor = ThreadPoolExecutor(max_workers=args.jobs)
    pending = {}
    try:
        with (output / "metrics.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=COLUMNS)
            writer.writeheader()
            while pending or iterator is not None:
                while iterator is not None and len(pending) < args.jobs:
                    item = next(iterator, None)
                    if item is None:
                        iterator = None
                    else:
                        totals = workloads[item[2]["concurrency"]]["totals"]
                        future = executor.submit(run_candidate, item, output, totals, args, stop)
                        pending[future] = item[1]
                if not pending:
                    break
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    config = pending.pop(future)
                    row = future.result()
                    rows.append(row)
                    writer.writerow(row)
                    stream.flush()
                    if row["eligible"] and (best is None or rank_key(row) < rank_key(best)):
                        best = row
                        atomic_write(output / "best.yaml", yaml.safe_dump(config, sort_keys=False))
                        write_json(output / "best.json", best)
                    detail = row["error"] or (
                        f"TTFT={row['median_ttft_ms']:.2f}ms TPOT={row['median_tpot_ms']:.2f}ms "
                        f"output={row['output_throughput_tok_s']:.2f}tok/s "
                        f"SLO={'PASS' if row['eligible'] else 'FAIL'}"
                    )
                    print(f"[{len(rows)}/{count}] {row['candidate']} {row['status']} {detail}", flush=True)
    finally:
        stop.set()
        executor.shutdown(wait=True, cancel_futures=True)
    ranked = sorted((row for row in rows if row["eligible"]), key=rank_key)
    with (output / "ranking.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(ranked)
    summary = {
        "candidates": count,
        "successful": sum(row["status"] == "ok" for row in rows),
        "invalid": sum(row["status"] == "invalid" for row in rows),
        "failed": sum(row["status"] == "error" for row in rows),
        "eligible": len(ranked),
        "ttft_scope": args.ttft_scope,
        "requests_per_concurrency": args.requests_per_concurrency,
        "best": best,
    }
    write_json(output / "summary.json", summary)
    if best:
        print(f"Best observed SLO-compliant prediction: {output / 'best.yaml'}", flush=True)
        return 0
    print("No complete prediction satisfies both strict P50 limits; no best.yaml was produced.", flush=True)
    return 2


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        args = configure_args(args, load_settings(args.config))
        prefix_counts = (
            [c * args.requests_per_concurrency for c in args.concurrencies] if args.requests_per_concurrency else []
        )
        base, totals = prepare_base(args.prediction, prefix_counts)
        space = search_space(base, args)
        workloads = prepare_workloads(base, totals, space, args)
        counts = {"candidates": 0, "valid": 0, "invalid": 0}
        for _, _, _, error in candidates(base, space, workloads):
            counts["candidates"] += 1
            counts["invalid" if error else "valid"] += 1
        request_counts = {c: w["totals"]["requests"] for c, w in workloads.items()}
        print(
            f"Grid: {counts}; jobs={args.jobs}; requests by concurrency={request_counts}; TTFT={args.ttft_scope}",
            flush=True,
        )
        print(f"Output: {args.output_dir}", flush=True)
        if args.dry_run:
            return 0
        manifest = {
            "base": base,
            "workload": totals,
            "workloads_by_concurrency": workloads,
            "search_space": space,
            "ttft_p50_ms": args.ttft_p50_ms,
            "tpot_p50_ms": args.tpot_p50_ms,
            "ttft_scope": args.ttft_scope,
            "requests_per_concurrency": args.requests_per_concurrency,
            "runtime": runtime_identity(base),
        }
        output = args.output_dir.resolve()
        output.mkdir(parents=True, exist_ok=True)
        with (output / ".lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError("another search is writing this output directory") from error
            prepare_output(output, manifest, args.resume)
            materialize_workloads(base, workloads)
            return search(base, workloads, space, args, output)
    except KeyboardInterrupt:
        print("Interrupted; completed candidates are saved. Re-run with --resume.", file=sys.stderr)
        return 130
    except (ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
