# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Legacy collector routes, retaining each implementation's compatibility guard."""

import importlib

__compat__ = "sglang>=0.5.14,<=0.5.17,!=0.5.15,!=0.5.16"

_ENTRY_POINTS = {
    "get_moe_test_cases": "collect_moe",
    "run_moe_torch": "collect_moe",
    "run_gemm": "collect_gemm",
    "get_gemm_test_cases": "collect_gemm",
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


def _call(name, *args, **kwargs):
    module = importlib.import_module(f"collector.sglang.{_ENTRY_POINTS[name]}")
    from importlib.metadata import version

    from collector.version_resolver import _check_compat

    if not _check_compat(module.__compat__, version("sglang")):
        raise RuntimeError(f"{module.__name__} requires {module.__compat__}")
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


def get_gemm_test_cases():
    return _call("get_gemm_test_cases")
