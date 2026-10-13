# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve a collection campaign's case snapshot before generators are imported."""

import os
from pathlib import Path


def collection_case_root() -> Path:
    override = os.environ.get("AISIM_COLLECTOR_CASES_ROOT")
    root = Path(override).expanduser().resolve() if override else Path(__file__).resolve().parent / "cases"
    for name in ("base_ops", "models"):
        if not (root / name).is_dir():
            raise ValueError(f"Collector case root is missing {name}/: {root}")
    return root
