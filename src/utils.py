"""
src/utils.py
------------
Shared utility functions: configuration loading, reproducibility seeding,
structured logging, and path helpers.
"""

from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a module-level logger with consistent formatting.

    Usage
    -----
    >>> log = get_logger(__name__)
    >>> log.info("Starting experiment")
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


# Module-level logger for utils itself
log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML configuration file.

    Searches for ``config/default.yaml`` relative to the repository root when
    *path* is not provided.  The repo root is inferred as the directory two
    levels above this file (``src/utils.py`` → ``src/`` → repo root).

    Parameters
    ----------
    path:
        Explicit path to a YAML config file.  When ``None`` the default
        ``config/default.yaml`` is used.

    Returns
    -------
    dict
        Parsed YAML contents.
    """
    if path is None:
        repo_root = Path(__file__).resolve().parent.parent
        path = repo_root / "config" / "default.yaml"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r") as fh:
        cfg = yaml.safe_load(fh)
    log.info("Loaded config from %s", path)
    return cfg


def get_nested(cfg: dict, *keys: str, default: Any = None) -> Any:
    """Safe nested key access for config dicts.

    Example
    -------
    >>> get_nested(cfg, "tda", "min_persistence", default=0.05)
    """
    node = cfg
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA/MPS) for reproducibility.

    Notes
    -----
    Full determinism on GPU also requires setting
    ``torch.backends.cudnn.deterministic = True``, which can slow training.
    This function does *not* do that automatically; callers may opt in.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if torch.backends.mps.is_available():
            # MPS does not expose a per-device seed API; manual_seed is sufficient
            pass
    except ImportError:
        pass

    log.debug("Global seed set to %d", seed)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def repo_root() -> Path:
    """Return the absolute path to the repository root."""
    return Path(__file__).resolve().parent.parent


def results_dir(cfg: dict | None = None, model_alias: str | None = None) -> Path:
    """Return (and create) the results directory for a given model alias.

    Parameters
    ----------
    cfg:
        Parsed config dict.  When ``None`` defaults to ``results/``.
    model_alias:
        Optional subdirectory name (e.g. ``"llama3"``).  When provided the
        returned path is ``<results_root>/<model_alias>/``.

    Returns
    -------
    Path
        Guaranteed to exist (created with ``mkdir`` if necessary).
    """
    if cfg is not None:
        root = repo_root() / get_nested(cfg, "paths", "results_dir", default="results/")
    else:
        root = repo_root() / "results"
    if model_alias:
        root = root / model_alias
    root.mkdir(parents=True, exist_ok=True)
    return root


def figures_dir(cfg: dict | None = None) -> Path:
    """Return (and create) the static figures directory."""
    if cfg is not None:
        root = repo_root() / get_nested(cfg, "paths", "figures_dir", default="static/")
    else:
        root = repo_root() / "static"
    root.mkdir(parents=True, exist_ok=True)
    return root


def prompts_path(cfg: dict | None = None) -> Path:
    """Return the path to the prompts JSON file."""
    if cfg is not None:
        rel = get_nested(cfg, "paths", "prompts", default="data/prompts.json")
    else:
        rel = "data/prompts.json"
    return repo_root() / rel


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------


def load_prompts(
    cfg: dict | None = None,
    categories: list[str] | None = None,
) -> list[dict]:
    """Load the prompt suite from ``data/prompts.json``.

    Parameters
    ----------
    cfg:
        Parsed config dict (used to locate the file).
    categories:
        Optional list of category strings to filter by (e.g.
        ``["repetition", "adversarial"]``).  When ``None`` all prompts are
        returned.

    Returns
    -------
    list of dict
        Each dict has keys: ``id``, ``category``, ``text``, ``notes``.
    """
    import json

    path = prompts_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"Prompts file not found: {path}")
    with path.open("r") as fh:
        data = json.load(fh)
    prompts = data["prompts"]
    if categories is not None:
        prompts = [p for p in prompts if p["category"] in categories]
    log.info(
        "Loaded %d prompts%s",
        len(prompts),
        f" (categories: {categories})" if categories else "",
    )
    return prompts


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------


def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Flatten a nested dict into a single-level dict with dotted keys.

    Example
    -------
    >>> flatten_dict({"a": {"b": 1, "c": 2}, "d": 3})
    {'a.b': 1, 'a.c': 2, 'd': 3}
    """
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Division that returns *default* instead of raising ZeroDivisionError."""
    return numerator / denominator if denominator != 0.0 else default


def to_numpy(tensor) -> np.ndarray:
    """Safely convert a PyTorch tensor (any device, any dtype) to a NumPy array."""
    try:
        import torch

        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().float().numpy()
    except ImportError:
        pass
    return np.asarray(tensor)
