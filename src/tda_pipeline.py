"""
src/tda_pipeline.py
-------------------
TDA core pipeline.  Replaces the original topo_scanner.py with:

  * Explicit documentation of the symmetrization step and its consequences.
  * Per-prompt aggregation returning mean ΔH1, std, and role-stability scores.
  * Full persistence-diagram output (not just lifetime scalars) so that
    topo_reg.py can consume the diagrams for Wasserstein-based regularization.

WARNING — Symmetrization
------------------------
``attention_to_distance`` symmetrizes the attention matrix via
``D_sym = 0.5 * (D + D^T)``.  This is required because Vietoris-Rips filtration
operates on an *undirected* metric space.  The consequence is that causal
directionality is lost: a loop (H1 cycle) in the resulting complex reflects
structural closure among tokens, NOT a directed causal flow cycle.  Any
interpretation about causal routing via the sink should be stated cautiously.
See LIMITATIONS.md §1 for a full discussion.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Low-level TDA primitives
# ---------------------------------------------------------------------------


def attention_to_distance(attn: np.ndarray) -> np.ndarray:
    """
    Convert an attention weight matrix to a symmetric distance matrix suitable
    for Vietoris-Rips filtration.

    Transformation
    --------------
    1.  D_ij = 1 - A_ij          (high attention ↔ small distance)
    2.  np.fill_diagonal(D, 0)   (self-distance = 0)
    3.  D_sym = 0.5 * (D + D^T)  (symmetrize — see module-level WARNING)

    Parameters
    ----------
    attn : np.ndarray, shape (seq_len, seq_len)
        Post-softmax attention weights.  Values should lie in [0, 1].

    Returns
    -------
    np.ndarray, shape (seq_len, seq_len)
        Symmetric distance matrix.  Diagonal is zero.
    """
    if attn.ndim != 2 or attn.shape[0] != attn.shape[1]:
        raise ValueError(f"Expected a square 2-D array, got shape {attn.shape}")
    D = 1.0 - attn.astype(np.float64)
    np.fill_diagonal(D, 0.0)
    D_sym = 0.5 * (D + D.T)
    return D_sym


def compute_persistence(
    distance_matrix: np.ndarray,
    max_filtration: float = 1.0,
    homology_dim: int = 1,
    min_persistence: float = 0.05,
) -> np.ndarray:
    """
    Run Vietoris-Rips persistent homology on a pre-computed distance matrix.

    Uses ``ripser`` (``pip install ripser``) under the hood — a lean, fast
    C++-backed persistent homology library that installs easily on macOS,
    Linux, and Windows without any additional build steps.

    Parameters
    ----------
    distance_matrix : np.ndarray, shape (n, n)
        Symmetric distance matrix (output of ``attention_to_distance``).
    max_filtration : float
        Upper bound on the filtration parameter.  Edges with distance >
        max_filtration are never added.  Default: 1.0.
    homology_dim : int
        Homology dimension to inspect.  1 → H1 loops.  Default: 1.
    min_persistence : float
        Minimum (death - birth) threshold.  Cycles with persistence below this
        value are filtered out as topological noise.  Default: 0.05.

    Returns
    -------
    np.ndarray, shape (n_cycles, 2)
        Persistence diagram rows [birth, death] for cycles that survive the
        ``min_persistence`` filter.  Returns shape (0, 2) when no cycles survive.

    Notes
    -----
    ripser returns ``np.inf`` for the death value of the single essential class
    that never dies (H0 always has one, H1 may have one for non-contractible
    spaces).  These infinite-death entries are filtered out before returning.
    """
    try:
        from ripser import ripser
    except ImportError as exc:
        raise ImportError(
            "ripser is required for TDA computations.  Install with: pip install ripser"
        ) from exc

    result = ripser(
        distance_matrix,
        maxdim=homology_dim,
        distance_matrix=True,
        thresh=max_filtration,
    )

    # result['dgms'] is a list indexed by dimension.
    # Each entry is an ndarray of shape (n_cycles, 2) with [birth, death].
    if homology_dim >= len(result["dgms"]):
        return np.empty((0, 2), dtype=np.float64)

    diagram_dim = result["dgms"][homology_dim]  # (k, 2)

    if len(diagram_dim) == 0:
        return np.empty((0, 2), dtype=np.float64)

    # Remove entries where death is infinite (essential classes)
    finite_mask = np.isfinite(diagram_dim[:, 1])
    diagram_dim = diagram_dim[finite_mask]

    if len(diagram_dim) == 0:
        return np.empty((0, 2), dtype=np.float64)

    # Filter by minimum persistence
    persistence = diagram_dim[:, 1] - diagram_dim[:, 0]
    keep = persistence >= min_persistence
    return diagram_dim[keep].astype(np.float64)


def _mean_lifetime(diagram: np.ndarray) -> float:
    """Return the mean lifetime of cycles in a persistence diagram.

    Parameters
    ----------
    diagram : np.ndarray, shape (n_cycles, 2)
        Rows are [birth, death].

    Returns
    -------
    float
        Mean (death − birth).  Returns 0.0 when the diagram is empty.
    """
    if len(diagram) == 0:
        return 0.0
    lifetimes = diagram[:, 1] - diagram[:, 0]
    return float(np.mean(lifetimes))


def compute_delta_lifetime(
    attn_unmasked: np.ndarray,
    attn_masked: np.ndarray,
    **persistence_kwargs: Any,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Compute ΔH1 = mean_lifetime(unmasked) − mean_lifetime(masked).

    Sign convention
    ---------------
    * ΔH1 < 0  →  removing the sink *destroys* loops → sink is a **bridge**
                  (the sink is necessary for the loops to close).
    * ΔH1 > 0  →  removing the sink *creates* loops → sink is a **cone**
                  (the sink was collapsing/absorbing cycles; without it they emerge).
    * ΔH1 ≈ 0  →  **neutral** (sink has little topological effect).

    Note: the original topo_scanner.py used (masked − unmasked), which flips
    the sign.  This implementation uses (unmasked − masked) to match the SPEC
    and to make the bridge/cone labels more intuitive.

    Parameters
    ----------
    attn_unmasked : np.ndarray, shape (seq_len, seq_len)
    attn_masked   : np.ndarray, shape (seq_len, seq_len)
        Output of ``src.models.mask_sink``.
    **persistence_kwargs
        Forwarded to ``compute_persistence``.

    Returns
    -------
    delta_h1 : float
    diagram_unmasked : np.ndarray, shape (n_cycles, 2)
    diagram_masked   : np.ndarray, shape (n_cycles, 2)
    """
    D_unmasked = attention_to_distance(attn_unmasked)
    D_masked = attention_to_distance(attn_masked)

    diag_unmasked = compute_persistence(D_unmasked, **persistence_kwargs)
    diag_masked = compute_persistence(D_masked, **persistence_kwargs)

    delta_h1 = _mean_lifetime(diag_unmasked) - _mean_lifetime(diag_masked)
    return delta_h1, diag_unmasked, diag_masked


# ---------------------------------------------------------------------------
# Role classification
# ---------------------------------------------------------------------------

_DEFAULT_BRIDGE_THRESHOLD = -0.03
_DEFAULT_CONE_THRESHOLD = 0.03


def _classify_role(
    delta_h1: float,
    bridge_threshold: float = _DEFAULT_BRIDGE_THRESHOLD,
    cone_threshold: float = _DEFAULT_CONE_THRESHOLD,
) -> str:
    if delta_h1 < bridge_threshold:
        return "bridge"
    if delta_h1 > cone_threshold:
        return "cone"
    return "neutral"


# ---------------------------------------------------------------------------
# Atlas builders
# ---------------------------------------------------------------------------


def build_topological_atlas(
    model,
    tokenizer,
    prompts: list[dict],
    layers: Optional[list[int]] = None,
    heads: Optional[list[int]] = None,
    sink_idx: int = 0,
    bridge_threshold: float = _DEFAULT_BRIDGE_THRESHOLD,
    cone_threshold: float = _DEFAULT_CONE_THRESHOLD,
    max_filtration: float = 1.0,
    homology_dim: int = 1,
    min_persistence: float = 0.05,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run TDA over every (layer, head) pair for every prompt and return a tidy
    DataFrame of per-prompt ΔH1 values plus an aggregated summary.

    Parameters
    ----------
    model, tokenizer
        Loaded HuggingFace model and tokenizer (from ``src.models.load_model``).
    prompts : list[dict]
        List of prompt dicts with keys ``id``, ``category``, ``text``
        (output of ``src.utils.load_prompts``).
    layers, heads : list[int] or None
        Restrict the scan to a subset of layers / heads.  None → all.
    sink_idx : int
        Position of the sink token (default 0).
    bridge_threshold, cone_threshold : float
        ΔH1 thresholds for classifying a head as bridge / cone / neutral.
    max_filtration, homology_dim, min_persistence
        Forwarded to ``compute_persistence``.
    verbose : bool
        Log progress at the layer level.

    Returns
    -------
    pd.DataFrame
        Columns: ``prompt_id``, ``category``, ``layer``, ``head``,
                 ``delta_h1``, ``role``.

        A separate aggregated DataFrame is **not** returned here; use
        ``aggregate_atlas`` to compute mean / std / stability scores.
    """
    from src.models import get_attention_matrices, mask_sink

    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    layer_range = list(layers) if layers is not None else list(range(n_layers))
    head_range = list(heads) if heads is not None else list(range(n_heads))

    persistence_kwargs = dict(
        max_filtration=max_filtration,
        homology_dim=homology_dim,
        min_persistence=min_persistence,
    )

    records = []

    for prompt_dict in prompts:
        prompt_id = prompt_dict["id"]
        category = prompt_dict.get("category", "unknown")
        text = prompt_dict["text"]

        if verbose:
            logger.info("Atlas scan — prompt: %s", prompt_id)

        # Single forward pass for all (layer, head) pairs
        try:
            all_attns = get_attention_matrices(
                model, tokenizer, text, layers=layer_range, heads=head_range
            )
        except Exception as exc:
            logger.warning("Forward pass failed for prompt %s: %s", prompt_id, exc)
            continue

        for layer_idx in layer_range:
            if verbose:
                logger.debug("  Layer %d / %d", layer_idx, n_layers - 1)
            for head_idx in head_range:
                key = (layer_idx, head_idx)
                if key not in all_attns:
                    continue
                attn_u = all_attns[key]
                attn_m = mask_sink(attn_u, sink_idx=sink_idx)

                try:
                    delta_h1, _, _ = compute_delta_lifetime(
                        attn_u, attn_m, **persistence_kwargs
                    )
                except Exception as exc:
                    logger.warning(
                        "TDA failed for prompt=%s layer=%d head=%d: %s",
                        prompt_id,
                        layer_idx,
                        head_idx,
                        exc,
                    )
                    delta_h1 = float("nan")

                role = _classify_role(delta_h1, bridge_threshold, cone_threshold)
                records.append(
                    {
                        "prompt_id": prompt_id,
                        "category": category,
                        "layer": layer_idx,
                        "head": head_idx,
                        "delta_h1": delta_h1,
                        "role": role,
                    }
                )

    df = pd.DataFrame(records)
    logger.info(
        "Atlas complete: %d rows, %d prompts, %d (layer,head) pairs",
        len(df),
        df["prompt_id"].nunique() if len(df) else 0,
        df.groupby(["layer", "head"]).ngroups if len(df) else 0,
    )
    return df


def build_control_atlas(
    model,
    tokenizer,
    prompts: list[dict],
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Build a topological atlas on a copy of *model* with randomized attention
    projection weights.

    The control atlas should display no consistent bridge/cone structure.
    Comparing it to the trained atlas validates that the observed topology is
    *learned* rather than a structural artefact of the architecture.

    Implementation
    --------------
    A deep copy of the model is made; then for each attention layer, the
    query/key/value/output projection weights (and biases, if present) are
    replaced with Gaussian noise matching the original parameter statistics
    (mean 0, same std-dev).  The copy is discarded after the scan.

    Parameters
    ----------
    model, tokenizer
        Original trained model and its tokenizer.
    prompts : list[dict]
        Same prompt suite used for the trained atlas.
    **kwargs
        Forwarded to ``build_topological_atlas``.

    Returns
    -------
    pd.DataFrame
        Same schema as ``build_topological_atlas``.
    """
    import torch

    logger.info("Building control atlas — randomizing attention weights ...")

    control_model = copy.deepcopy(model)

    with torch.no_grad():
        for name, param in control_model.named_parameters():
            # Target attention projection weights/biases only
            is_attn = any(
                kw in name
                for kw in (
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "query",
                    "key",
                    "value",
                    "out_proj",
                    "W_Q",
                    "W_K",
                    "W_V",
                    "W_O",
                )
            )
            if not is_attn:
                continue

            # Skip quantized storage tensors (Byte/uint8/int8).
            # 4-bit quantized models store weights as Byte; torch.randn_like on
            # these raises "normal_kernel_cuda not implemented for Byte".
            # We randomize only float/half parameters — these are the compute
            # buffers that actually produce the attention outputs TDA measures.
            if param.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32):
                logger.debug("Skipping quantized param %s (dtype=%s)", name, param.dtype)
                continue

            std = param.float().std().item()
            std = std if std > 0 else 0.02  # fallback for near-zero params
            noise = torch.randn(param.shape, dtype=param.dtype, device=param.device) * std
            param.copy_(noise)

    logger.info("Attention weights randomized.  Running TDA scan ...")

    control_df = build_topological_atlas(control_model, tokenizer, prompts, **kwargs)

    # Free GPU memory from the temporary copy
    del control_model
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return control_df


# ---------------------------------------------------------------------------
# Post-hoc aggregation
# ---------------------------------------------------------------------------


def aggregate_atlas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-prompt ΔH1 values into per-(layer, head) summary statistics.

    Computed statistics
    -------------------
    * ``mean_delta_h1``  : mean ΔH1 across all prompts
    * ``std_delta_h1``   : standard deviation (missing in original codebase)
    * ``n_prompts``      : number of prompts that produced a valid (non-NaN) result
    * ``role``           : majority role across prompts
    * ``stability``      : fraction of prompts where the head's role matches the
                           majority role — a value of 1.0 means the head is
                           consistently bridge / cone across all prompts.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``build_topological_atlas``.

    Returns
    -------
    pd.DataFrame
        Indexed by (layer, head); reset index is applied for compatibility.
        Columns: ``layer``, ``head``, ``mean_delta_h1``, ``std_delta_h1``,
                 ``n_prompts``, ``role``, ``stability``.
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "layer",
                "head",
                "mean_delta_h1",
                "std_delta_h1",
                "n_prompts",
                "role",
                "stability",
            ]
        )

    def _majority_role(series: pd.Series) -> str:
        return series.mode().iloc[0] if len(series) > 0 else "neutral"

    def _stability(series: pd.Series) -> float:
        if len(series) == 0:
            return 0.0
        majority = series.mode().iloc[0]
        return float((series == majority).mean())

    agg = (
        df.dropna(subset=["delta_h1"])
        .groupby(["layer", "head"])
        .agg(
            mean_delta_h1=("delta_h1", "mean"),
            std_delta_h1=("delta_h1", "std"),
            n_prompts=("delta_h1", "count"),
            role=("role", _majority_role),
            stability=("role", _stability),
        )
        .reset_index()
    )

    # Fill NaN std (occurs when only one prompt) with 0
    agg["std_delta_h1"] = agg["std_delta_h1"].fillna(0.0)

    return agg


def get_hub_heads(
    agg_df: pd.DataFrame,
    hub_threshold: float = 0.3,
    roles: Optional[list[str]] = None,
) -> list[tuple[int, int]]:
    """
    Return (layer, head) pairs whose |mean_delta_h1| exceeds *hub_threshold*.

    Parameters
    ----------
    agg_df : pd.DataFrame
        Output of ``aggregate_atlas``.
    hub_threshold : float
        Minimum |mean_delta_h1| to be considered a hub.
    roles : list[str] or None
        Restrict to specific roles, e.g. ``["bridge", "cone"]``.  None → all.

    Returns
    -------
    list of (layer, head) tuples, sorted by |mean_delta_h1| descending.
    """
    mask = agg_df["mean_delta_h1"].abs() >= hub_threshold
    if roles is not None:
        mask &= agg_df["role"].isin(roles)
    subset = agg_df[mask].copy()
    subset["abs_delta"] = subset["mean_delta_h1"].abs()
    subset = subset.sort_values("abs_delta", ascending=False)
    return list(zip(subset["layer"].tolist(), subset["head"].tolist()))
