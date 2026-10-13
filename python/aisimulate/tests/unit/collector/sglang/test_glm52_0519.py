# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Version selection and the 0.5.19 request/config API contracts, without CUDA."""

import ast
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from collector.framework_manifest import require_collector_runtime, validate_manifest
from collector.sglang.registry import REGISTRY
from collector.version_resolver import _check_compat, resolve_module

pytestmark = pytest.mark.unit
MODEL = "nvidia/GLM-5.2-NVFP4"
SOURCE = Path(__file__).resolve().parents[4] / "collector/sglang/collect_mla_module.py"


@pytest.fixture
def collector_module(monkeypatch):
    # collect.py is also a standalone script and imports its sibling helper.
    monkeypatch.syspath_prepend(str(SOURCE.parents[1]))
    from collector import collect

    return collect


def helper(name, version="0.5.19"):
    tree = ast.parse(SOURCE.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {"get_version": lambda _: version}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace[name]


def test_model_scoped_runtime_and_routes_keep_legacy_pins():
    runtime = require_collector_runtime("sglang", "0.5.19", requested_ops={"gemm", "moe"}, model_path=MODEL)
    assert runtime.execution == "local_python"
    assert runtime.images == {}
    with pytest.raises(ValueError, match="local Python"):
        runtime.image()
    legacy = require_collector_runtime("sglang", "0.5.14", requested_ops={"gemm"})
    assert legacy.execution == "container"
    with pytest.raises(RuntimeError, match="requires exactly"):
        require_collector_runtime("sglang", "0.5.20", requested_ops={"gemm"}, model_path=MODEL)
    entry = next(e for e in REGISTRY if e.op == "dsa_context_module")
    assert resolve_module(entry, "0.5.14").endswith("_v1")
    assert resolve_module(entry, "0.5.19").endswith("_v2")
    assert _check_compat("sglang==0.5.19", "0.5.19")
    assert not _check_compat("sglang==0.5.19", "0.5.20")


def test_local_sidecar_keeps_version_without_inventing_image(collector_module):
    runtime = require_collector_runtime("sglang", "0.5.19", requested_ops={"gemm"}, model_path=MODEL)
    assert collector_module._runtime_metadata({"runtime": runtime}) == {"framework": "sglang", "version": "0.5.19"}


@pytest.mark.parametrize("prefix,end", [(0, 128), (4096, 4224)])
def test_request_range_preserves_cached_prefix_and_full_input(prefix, end):
    req = SimpleNamespace(origin_input_ids=list(range(end)), set_extend_range=Mock())
    helper("_set_request_extend_range")(req, prefix, end)
    req.set_extend_range.assert_called_once_with(prefix, end)
    assert list(req.full_untruncated_fill_ids) == req.origin_input_ids
    assert not hasattr(req, "fill_len")


def test_legacy_request_constructor_is_preserved():
    req = SimpleNamespace(origin_input_ids=list(range(20)), set_extend_input_len=Mock())
    helper("_set_request_extend_range", "0.5.14")(req, 12, 20)
    req.set_extend_input_len.assert_called_once_with(8)
    assert req.fill_len == 20


def test_forward_batch_supplies_new_explicit_hidden_state_policy(monkeypatch):
    init = Mock(return_value="batch")
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.model_executor.forward_batch_info",
        SimpleNamespace(ForwardBatch=SimpleNamespace(init_new=init)),
    )
    assert helper("_make_forward_batch")("schedule", "runner") == "batch"
    init.assert_called_once_with("schedule", "runner", return_hidden_states_before_norm=False)


def test_graph_coverage_reads_resolved_runtime_instead_of_raw_server_args(monkeypatch):
    config = SimpleNamespace(decode=SimpleNamespace(backend="cudagraphs", bs=[1, 4], max_bs=4))
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.runtime_context",
        SimpleNamespace(get_exec=lambda: SimpleNamespace(graph=SimpleNamespace(cuda_graph_config=config))),
    )
    enabled = helper("_generation_cuda_graph_enabled_for_tokens")
    assert enabled(SimpleNamespace(), 4)
    assert not enabled(SimpleNamespace(), 3)


@pytest.mark.parametrize("execution,images", [("local_python", {"default": "fake"}), ("unknown", {})])
def test_runtime_cannot_mislabel_a_local_environment(execution, images):
    with pytest.raises(ValueError):
        validate_manifest(
            {
                "schema_version": 2,
                "frameworks": {"sglang": {"default": {"version": "0.5.19", "execution": execution, "images": images}}},
            }
        )


def test_skip_indexer_enum_filename_reaches_worker_on_python311(monkeypatch, collector_module):
    from collector.version_resolver import build_collections

    op = "dsa_context_module_skip_indexer"
    entry = next(e for e in REGISTRY if e.op == op)
    module = SimpleNamespace(
        __compat__="sglang==0.5.19",
        get_dsa_context_module_skip_indexer_test_cases=lambda: [],
        run_mla_module_worker=lambda **kwargs: kwargs,
    )
    monkeypatch.setitem(sys.modules, resolve_module(entry, "0.5.19"), module)
    called = []

    def run(_backend, _op, _get, worker, _n, **kwargs):
        called.append(worker())
        return []

    monkeypatch.setattr(collector_module, "collect_module_safe", run)
    collections = build_collections([entry], "sglang", "0.5.19")
    assert collector_module.collect_ops(1, collections, runtime_version="0.5.19", sm_version=103) == []
    assert called == [{"perf_filename": str(entry.worker_perf_filename)}]
