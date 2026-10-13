# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract checks for the example CLI grid search, without performance data."""

import importlib.util
import json
import sys
import threading
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "tune_predict", ROOT / "examples/sweeper/tools/predict_tuner/tune_predict.py"
)
TUNE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TUNE
SPEC.loader.exec_module(TUNE)


@pytest.fixture
def setup(tmp_path):
    config = {
        "traffic": {
            "source": {"type": "synthetic", "input_tokens": 100, "output_tokens": 3},
            "load": {"type": "concurrency", "concurrency": 2},
            "stop": {"requests": 2},
        },
        "engine": {
            "model": "nvidia/GLM-5.2-NVFP4",
            "hardware": "b300_sxm",
            "backend": "sglang",
            "backend_version": "0.5.17",
            "nextn": 3,
            "nextn_accepted": 1.11,
            "workers": {
                "aggregated": {
                    "parallelism": {"tensor": 4, "attention_data": 2, "moe_tensor": 4, "moe_expert": 2},
                    "kv_cache": {"block_size": 64},
                }
            },
        },
    }
    path = tmp_path / "search.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "prediction": config,
                "output_dir": "out",
                "metrics": {"ttft_scope": "all"},
                "search": {"concurrency": [16, 32], "chunked_prefill_size": [32768], "max_prefill_tokens": [458752]},
                "run": {"jobs": 2},
            }
        )
    )
    args = TUNE.configure_args(TUNE.parser().parse_args(["-c", str(path)]), TUNE.load_settings(path))
    base, totals = TUNE.prepare_base(args.prediction)
    return base, totals, args


def report(**changes):
    ttft, tpot = changes.get("median_ttft_ms", 100), changes.get("median_tpot_ms", 20)
    output_tps = changes.get("output_throughput_tok_s", 100)
    return {
        "num_requests": 2,
        "completed_requests": 2,
        "num_ttft_samples": 2,
        "num_tpot_samples": 2,
        "total_input_tokens": 200,
        "total_output_tokens": 6,
        "median_ttft_ms": 100,
        "median_tpot_ms": 20,
        "output_throughput_tok_s": 100,
        "input_throughput_tok_s": output_tps * 200 / 6,
        "total_throughput_tok_s": output_tps * 206 / 6,
        "median_itl_ms": 0,
        "per_request": [
            {
                "arrival_time_ms": 0,
                "first_token_ms": ttft,
                "last_token_ms": ttft + 2 * (tpot or 0),
                "ttft_ms": ttft,
                "input_length": 100,
                "output_length": 3,
                "terminal_status": "completed",
            }
            for _ in range(2)
        ],
        **changes,
    }


def test_layout_preserves_identity_and_passes_undivided_chunk(setup):
    base, _, args = setup
    args.dp_ep = [(2, 2), (8, 1)]
    args.concurrencies = [32]
    items = list(TUNE.candidates(base, TUNE.search_space(base, args)))
    assert len(items) == 2
    for _, config, _, error in items:
        assert not error
        engine = config["engine"]
        assert engine["backend_version"] == "0.5.17"
        assert engine["nextn"] == 3 and engine["nextn_accepted"] == 1.11
        worker = engine["workers"]["aggregated"]
        dp = worker["parallelism"]["attention_data"]
        assert worker["parallelism"]["tensor"] * dp == 8
        assert worker["parallelism"]["moe_tensor"] * worker["parallelism"]["moe_expert"] == 8
        assert worker["scheduler"]["chunked_prefill_size"] == 32768
        assert worker["scheduler"]["max_batched_tokens"] == 32768 // dp
        assert worker["scheduler"]["max_prefill_tokens"] == 458752
        TUNE.CorePredictionConfig.model_validate(config)


@pytest.mark.parametrize("chunk,prefill,reason", [(129, 458752, "divisible"), (32768, 8192, "smaller")])
def test_invalid_budgets_are_excluded_before_predict(setup, chunk, prefill, reason):
    base, _, args = setup
    args.chunk_sizes, args.max_prefill_tokens = [chunk], [prefill]
    assert all(reason in item[3] for item in TUNE.candidates(base, TUNE.search_space(base, args)))


@pytest.mark.parametrize(
    "changes",
    [
        {"median_tpot_ms": float("nan")},
        {"median_ttft_ms": float("inf")},
        {"median_tpot_ms": None},
        {"output_throughput_tok_s": True},
        {"completed_requests": 1},
        {"num_requests": 1},
        {"num_tpot_samples": 1},
        {"total_output_tokens": 5},
        {"total_input_tokens": 199},
    ],
)
def test_bad_or_partial_reports_cannot_win(setup, changes):
    _, totals, _ = setup
    with pytest.raises(ValueError):
        TUNE.score(report(**changes), totals, 15000, 33)


@pytest.mark.parametrize(
    "changes,eligible",
    [
        ({}, True),
        ({"median_ttft_ms": 15000}, False),
        ({"median_tpot_ms": 33}, False),
        ({"median_tpot_ms": 34, "median_itl_ms": 0}, False),
    ],
)
def test_slo_is_strict_and_uses_request_tpot(setup, changes, eligible):
    _, totals, _ = setup
    assert TUNE.score(report(**changes), totals, 15000, 33)["eligible"] is eligible


def test_resume_rejects_changed_inputs_without_overwriting(tmp_path):
    manifest = {"base": {"nextn_accepted": 1.11}, "workload": {"trace_sha256": "a"}}
    TUNE.prepare_output(tmp_path, manifest, False)
    original = (tmp_path / "manifest.json").read_bytes()
    TUNE.prepare_output(tmp_path, manifest, True)
    with pytest.raises(ValueError, match="resume requires"):
        TUNE.prepare_output(tmp_path, {**manifest, "workload": {"trace_sha256": "b"}}, True)
    assert (tmp_path / "manifest.json").read_bytes() == original
    with pytest.raises(ValueError, match="not empty"):
        TUNE.prepare_output(tmp_path, manifest, False)


def test_resume_revalidates_report_and_retries_partial_output(setup, tmp_path, monkeypatch):
    base, totals, args = setup
    args.resume = True
    item = next(TUNE.candidates(base, TUNE.search_space(base, args)))
    calls = []

    def predict(command, log, timeout, stop):
        calls.append(command)
        out = Path(command[command.index("--output-dir") + 1])
        out.mkdir(exist_ok=True)
        (out / "prediction.json").write_text(json.dumps(report()))
        return 0

    monkeypatch.setattr(TUNE, "run_command", predict)
    first = TUNE.run_candidate(item, tmp_path, totals, args, threading.Event())
    cached = TUNE.run_candidate(item, tmp_path, totals, args, threading.Event())
    assert first == cached and len(calls) == 1
    path = tmp_path / "candidates" / item[0] / "prediction" / "prediction.json"
    path.write_text(json.dumps(report(total_output_tokens=1)))
    retried = TUNE.run_candidate(item, tmp_path, totals, args, threading.Event())
    assert retried["status"] == "ok" and len(calls) == 2


@pytest.mark.parametrize("all_fail", [False, True])
def test_search_saves_only_slo_compliant_winner(setup, tmp_path, monkeypatch, all_fail):
    base, totals, args = setup

    def predict(command, log, timeout, stop):
        config = yaml.safe_load(Path(command[command.index("-c") + 1]).read_text())
        high_throughput = config["traffic"]["load"]["concurrency"] == 32
        result = report(
            output_throughput_tok_s=1000 if high_throughput else 100,
            median_tpot_ms=34 if high_throughput or all_fail else 20,
        )
        out = Path(command[command.index("--output-dir") + 1])
        out.mkdir(exist_ok=True)
        (out / "prediction.json").write_text(json.dumps(result))
        return 0

    monkeypatch.setattr(TUNE, "run_command", predict)
    space = TUNE.search_space(base, args)
    workloads = TUNE.prepare_workloads(base, totals, space, args)
    code = TUNE.search(base, workloads, space, args, tmp_path)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["successful"] == 2
    assert code == (2 if all_fail else 0)
    if all_fail:
        assert summary["best"] is None
        assert not (tmp_path / "best.yaml").exists()
    else:
        best = yaml.safe_load((tmp_path / "best.yaml").read_text())
        assert best["traffic"]["load"]["concurrency"] == 16


def test_timeout_terminates_child(tmp_path):
    with (tmp_path / "run.log").open("w") as log, pytest.raises(RuntimeError, match="timeout"):
        TUNE.run_command([sys.executable, "-c", "import time; time.sleep(60)"], log, 0.01, threading.Event())


def test_trace_is_fingerprinted_and_token_totals_verified(setup, tmp_path):
    base, _, args = setup
    trace = tmp_path / "trace.jsonl"
    trace.write_text('{"input_length":100,"output_length":3}\n{"input_length":200,"output_length":5}\n')
    base["traffic"]["source"] = {"type": "trace", "format": "mooncake", "paths": [str(trace)], "block_size": 1}
    base["traffic"].pop("stop")
    args.config.write_text(yaml.safe_dump(base))
    _, totals = TUNE.prepare_base(args.config)
    assert totals == {"requests": 2, "input_tokens": 300, "output_tokens": 8, "trace_sha256": TUNE.file_hash(trace)}


def test_yaml_paths_are_relative_to_config_not_cwd(setup, tmp_path, monkeypatch):
    base, _, args = setup
    trace = tmp_path / "trace.jsonl"
    trace.write_text('{"input_length":100,"output_length":3}\n')
    base["traffic"] = {
        "source": {"type": "trace", "format": "mooncake", "paths": ["trace.jsonl"]},
        "load": {"type": "concurrency", "concurrency": 1},
    }
    base["engine"]["model"] = "./local-model"
    base["engine"]["systems_paths"] = ["default", "./profiles"]
    args.config.write_text(yaml.safe_dump({"prediction": base, "output_dir": "results"}))
    monkeypatch.chdir(ROOT)
    settings = TUNE.load_settings(args.config)
    assert settings.output_dir == str(tmp_path / "results")
    assert settings.prediction.traffic.source.paths == [str(trace)]
    assert settings.prediction.engine.model == str(tmp_path / "local-model")
    assert settings.prediction.engine.systems_paths == ["default", str(tmp_path / "profiles")]


@pytest.mark.parametrize(
    "section,value",
    [
        ("search", {"concurency": [2]}),
        ("search", {"concurrency": []}),
        ("search", {"concurrency": [True]}),
        ("search", {"concurrency": [2, 2]}),
        ("search", {"dp_ep": [[2, 2, 2]]}),
        ("search", {"memory_fraction": [1.1]}),
        ("run", {"jobs": 0}),
        ("slo", {"tpot_p50_ms": float("inf")}),
        ("metrics", {"ttft_scope": "batch"}),
        ("workload", {"requests_per_concurrency": 0}),
        ("prediction", {"engine": []}),
    ],
)
def test_invalid_yaml_is_rejected_before_output(setup, section, value):
    _, _, args = setup
    config = yaml.safe_load(args.config.read_text())
    config[section] = value
    args.config.write_text(yaml.safe_dump(config))
    assert TUNE.main(["-c", str(args.config)]) == 2
    assert not args.output_dir.exists()


def test_first_batch_ttft_and_full_tpot_use_different_samples():
    requests = []
    for index in range(8):
        arrival = 0 if index < 2 else 20000 + index * 100
        ttft = 20000 if index < 2 else 10
        tpot = 5 if index < 2 else 30
        requests.append(
            {
                "arrival_time_ms": arrival,
                "first_token_ms": arrival + ttft,
                "last_token_ms": arrival + ttft + tpot * 2,
                "ttft_ms": ttft,
                "input_length": 100,
                "output_length": 3,
                "terminal_status": "completed",
            }
        )
    result = report(
        num_requests=8,
        completed_requests=8,
        num_ttft_samples=8,
        num_tpot_samples=8,
        total_input_tokens=800,
        total_output_tokens=24,
        per_request=list(reversed(requests)),
    )
    totals = {"requests": 8, "input_tokens": 800, "output_tokens": 24}
    first = TUNE.score(result, totals, 15000, 33, "first_batch", 2)
    all_requests = TUNE.score(result, totals, 15000, 33, "all", 2)
    assert first["median_ttft_ms"] == 20000 and first["ttft_samples"] == 2
    assert first["all_requests_median_ttft_ms"] == all_requests["median_ttft_ms"] == 10
    assert first["median_tpot_ms"] == 30 and first["tpot_samples"] == 8
    assert not first["eligible"] and all_requests["eligible"]
    with pytest.raises(ValueError, match="initial arrivals"):
        TUNE.score(result, totals, 15000, 33, "first_batch", 3)


def test_trace_prefixes_are_shared_and_preserve_records(setup, tmp_path):
    base, _, args = setup
    args.concurrencies, args.requests_per_concurrency, args.ttft_scope = [1, 2], 4, "first_batch"
    trace = tmp_path / "trace.jsonl"
    records = [
        json.dumps({"input_length": 100 + i, "output_length": 3 + i, "hash_ids": [1, 2, 100 + i]}) + "\n"
        for i in range(10)
    ]
    trace.write_text("\n" + "\n".join(records))
    base["traffic"] = {
        "source": {"type": "trace", "format": "mooncake", "paths": [str(trace)]},
        "load": {"type": "concurrency", "concurrency": 1},
    }
    base, totals = TUNE.prepare_base(TUNE.CorePredictionConfig.model_validate(base), [4, 8])
    space = TUNE.search_space(base, args)
    workloads = TUNE.prepare_workloads(base, totals, space, args)
    assert not args.output_dir.exists()
    TUNE.materialize_workloads(base, workloads)
    for concurrency, workload in workloads.items():
        count = concurrency * 4
        path = Path(workload["traffic"]["source"]["paths"][0])
        saved = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        assert saved == [json.loads(line) for line in records[:count]]
        assert workload["totals"]["input_tokens"] == sum(100 + i for i in range(count))
        assert TUNE.file_hash(path) == workload["totals"]["trace_sha256"]
    items = list(TUNE.candidates(base, space, workloads))
    assert {item[1]["traffic"]["source"]["paths"][0] for item in items} == {
        str(args.output_dir / "workloads/requests-4.jsonl"),
        str(args.output_dir / "workloads/requests-8.jsonl"),
    }
    args.requests_per_concurrency = 8
    with pytest.raises(ValueError, match="trace has 10"):
        TUNE.prepare_workloads(base, totals, space, args)


def test_yaml_entrypoint_runs_and_resumes_with_metric_scope(setup, monkeypatch):
    _, _, args = setup
    config = yaml.safe_load(args.config.read_text())
    config.update(
        search={"concurrency": [2], "chunked_prefill_size": [32768], "max_prefill_tokens": [458752]},
        metrics={"ttft_scope": "first_batch"},
        workload={"requests_per_concurrency": 4},
    )
    args.config.write_text(yaml.safe_dump(config))
    calls = []

    def predict(command, log, timeout, stop):
        calls.append(command)
        generated = yaml.safe_load(Path(command[command.index("-c") + 1]).read_text())
        assert generated["traffic"]["stop"]["requests"] == 8
        assert "--capture-per-request" in command
        result = report(
            num_requests=8,
            completed_requests=8,
            num_ttft_samples=8,
            num_tpot_samples=8,
            total_input_tokens=800,
            total_output_tokens=24,
        )
        initial = result["per_request"]
        result["per_request"] = initial + [{**initial[0], "arrival_time_ms": 1} for _ in range(6)]
        out = Path(command[command.index("--output-dir") + 1])
        out.mkdir(exist_ok=True)
        (out / "prediction.json").write_text(json.dumps(result))
        return 0

    monkeypatch.setattr(TUNE, "run_command", predict)
    monkeypatch.setattr(TUNE, "runtime_identity", lambda _: {"test": "fixed"})
    assert TUNE.main(["-c", str(args.config), "--dry-run"]) == 0
    assert not args.output_dir.exists()
    assert TUNE.main(["-c", str(args.config)]) == 0
    config["run"]["jobs"] = 1
    args.config.write_text(yaml.safe_dump(config))
    assert TUNE.main(["-c", str(args.config), "--resume"]) == 0
    assert len(calls) == 1
    best = json.loads((args.output_dir / "best.json").read_text())
    assert best["ttft_samples"] == 2 and best["tpot_samples"] == 8
    assert all(name in best for name in TUNE.METRICS)
    config["metrics"]["ttft_scope"] = "all"
    args.config.write_text(yaml.safe_dump(config))
    assert TUNE.main(["-c", str(args.config), "--resume"]) == 2
    assert len(calls) == 1
