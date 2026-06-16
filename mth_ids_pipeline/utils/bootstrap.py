"""Garante FCBF_module no sys.path."""

import sys

from mth_ids_pipeline.config import REPO_ROOT


def ensure_repo_on_path():
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return REPO_ROOT
