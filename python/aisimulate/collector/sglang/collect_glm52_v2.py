# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact SGLang 0.5.19 entry points for the local GLM-5.2 NVFP4 campaign.

Uses the existing timing/data contract. Version-specific API adaptations live
at the shared call sites, while this route limits the new release to its audited
model/precision scope. See benchmarks/evidence/sglang-0.5.19-collector/.
"""

import importlib
import os

__compat__ = "sglang==0.5.19"

_MODEL = "nvidia/GLM-5.2-NVFP4"
_ENTRY_POINTS = {
    "get_moe_test_cases": "collect_moe",
    "run_moe_torch": "collect_moe",
    "run_gemm": "collect_gemm",
    "get_dsa_context_module_test_cases": "collect_mla_module",
    "get_dsa_generation_module_test_cases": "collect_mla_module",
    "get_dsa_context_module_skip_indexer_test_cases": "collect_mla_module",
    "get_dsa_generation_module_skip_indexer_test_cases": "collect_mla_module",
    "run_mla_module_worker": "collect_mla_module",
    "get_glm5_mqa_test_cases": "glm5_dsa_sparse_modules",
    "get_glm5_topk_test_cases": "glm5_dsa_sparse_modules",
    "get_glm5_dsa_attn_test_cases": "glm5_dsa_sparse_modules",
    "run_glm5_dsa_sparse_kernel_worker": "glm5_dsa_sparse_modules",
}


def _require_model():
    if os.environ.get("COLLECTOR_MODEL_PATH") != _MODEL:
        raise ValueError(f"SGLang 0.5.19 collector requires --model-path {_MODEL}")
    from collector.helper import get_sm_version

    if get_sm_version() != 103:
        raise ValueError("The local SGLang 0.5.19 GLM-5.2 collector is validated for B300 (SM103)")


def get_gemm_test_cases():
    _require_model()
    from collector.case_generator import get_gemm_case_specs

    requested = os.environ.get("AIC_COLLECT_GEMM_TYPES", "bfloat16")
    if requested != "bfloat16":
        raise ValueError("GLM-5.2 NVFP4 uses BF16 independent GEMMs; routed NVFP4 GEMMs are collected by moe")
    return [["bfloat16", case.x, case.n, case.k] for case in get_gemm_case_specs()]


def _call(name, *args, **kwargs):
    _require_model()
    module = importlib.import_module(f"collector.sglang.{_ENTRY_POINTS[name]}")
    return getattr(module, name)(*args, **kwargs)


def get_moe_test_cases():
    return _call("get_moe_test_cases")


def run_moe_torch(*args, **kwargs):
    return _call("run_moe_torch", *args, **kwargs)


def run_gemm(*args, **kwargs):
    return _call("run_gemm", *args, **kwargs)


def get_dsa_context_module_test_cases():
    return _call("get_dsa_context_module_test_cases")


def get_dsa_generation_module_test_cases():
    return _call("get_dsa_generation_module_test_cases")


def get_dsa_context_module_skip_indexer_test_cases():
    return _call("get_dsa_context_module_skip_indexer_test_cases")


def get_dsa_generation_module_skip_indexer_test_cases():
    return _call("get_dsa_generation_module_skip_indexer_test_cases")


def run_mla_module_worker(*args, **kwargs):
    return _call("run_mla_module_worker", *args, **kwargs)


def get_glm5_mqa_test_cases():
    return _call("get_glm5_mqa_test_cases")


def get_glm5_topk_test_cases():
    return _call("get_glm5_topk_test_cases")


def get_glm5_dsa_attn_test_cases():
    return _call("get_glm5_dsa_attn_test_cases")


def run_glm5_dsa_sparse_kernel_worker(*args, **kwargs):
    return _call("run_glm5_dsa_sparse_kernel_worker", *args, **kwargs)
