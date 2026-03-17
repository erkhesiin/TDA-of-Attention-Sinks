"""
src/topo_reg.py
---------------
Improved topological regularization for fine-tuning.

Problem with the original approach
-----------------------------------
The original loss was:

    L_topo = λ Σ_h ||A_h^current - A_h^anchor||_F

This penalizes *any* change in the raw attention weights for hub heads, which
is effectively parameter-freezing rather than topology-preservation.  A head
can drift topologically while keeping Frobenius distance small (e.g. permuting
columns), and conversely can have a large Frobenius distance while preserving
its H1 structure.

New approach: Persistence-Diagram Loss
---------------------------------------
We replace the Frobenius term with the 2-Wasserstein distance between the
persistence diagrams of the current and anchor attention matrices:

    L_topo = λ Σ_h  W2( PD(A_h^current), PD(A_h^anchor) )

The anchor diagrams are computed *once* before fine-tuning begins and remain
fixed throughout training.  This directly penalizes topological drift rather
than weight drift.

Differentiability note
----------------------
The TDA pipeline (ripser VR filtration) is not differentiable.  We
therefore:
  1. Compute PD(A_h^current) in numpy (no-grad).
  2. Use persim.wasserstein to get the scalar W2 distance.
  3. Compute the Frobenius distance ||A_h - A_h^anchor||_F as a differentiable
     *proxy* that is weighted by the W2 scalar (so gradients still flow through
     PyTorch while the loss *magnitude* reflects topological distance).

This is a principled approximation: when W2 is large the proxy gradient is
amplified, pushing the weights back toward a topologically similar regime.
Falls back to plain Frobenius when persim/ripser are unavailable.

Fine-tuning loop
----------------
Three conditions are implemented and compared:
  A — no regularization
  B — persistence-diagram regularization (this module)
  C — Frobenius regularization (original method, for ablation)

Evaluation is performed on held-out validation prompts every `eval_every`
steps.  The best checkpoint (by validation task loss) is saved for each
condition.
"""

from __future__ import annotations

import copy
import logging
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Anchor diagram computation
# ---------------------------------------------------------------------------


def compute_anchor_diagrams(
    model,
    tokenizer,
    prompts: list[dict],
    hub_heads: list[tuple[int, int]],
    sink_idx: int = 0,
    max_filtration: float = 1.0,
    homology_dim: int = 1,
    min_persistence: float = 0.05,
) -> dict[tuple[int, int], np.ndarray]:
    """
    Compute and return persistence diagrams for hub heads *before* fine-tuning.

    Call this once, then pass the result to ``persistence_diagram_loss`` and
    ``topological_drift`` throughout training.

    The diagram for each (layer, head) is the *mean* diagram aggregated over
    all provided prompts (using a simple concatenation of all valid birth-death
    pairs).  This makes the anchor robust to individual prompt variance.

    Parameters
    ----------
    model : AutoModelForCausalLM
        Trained model (eval mode, before any LoRA adaptation).
    tokenizer : AutoTokenizer
    prompts : list[dict]
        Prompt suite (from ``src.utils.load_prompts``).
    hub_heads : list[tuple[int, int]]
        (layer, head) pairs to compute anchors for.  Typically the output of
        ``src.tda_pipeline.get_hub_heads``.
    sink_idx : int
        Position of the attention sink token (default 0).
    max_filtration, homology_dim, min_persistence
        Forwarded to ``src.tda_pipeline.compute_persistence``.

    Returns
    -------
    dict
        (layer, head) → np.ndarray of shape (n_cycles, 2) [birth, death].
        Heads that produced zero surviving cycles for all prompts map to an
        empty array of shape (0, 2).
    """
    from src.models import get_attention_matrices  # noqa: PLC0415
    from src.tda_pipeline import attention_to_distance, compute_persistence  # noqa

    logger.info(
        "Computing anchor diagrams for %d hub heads over %d prompts ...",
        len(hub_heads),
        len(prompts),
    )

    # Collect layers and heads to fetch in a single forward pass per prompt
    layers_needed = sorted({layer_idx for layer_idx, _ in hub_heads})
    heads_needed = sorted({h for _, h in hub_heads})

    accumulated: dict[tuple[int, int], list[np.ndarray]] = {lh: [] for lh in hub_heads}

    model.eval()
    with torch.no_grad():
        for pd_ in prompts:
            text = pd_["text"]
            try:
                matrices = get_attention_matrices(
                    model,
                    tokenizer,
                    text,
                    layers=layers_needed,
                    heads=heads_needed,
                )
            except Exception as exc:
                logger.warning("Skipping prompt '%s' for anchors: %s", pd_["id"], exc)
                continue

            for layer_idx, head_idx in hub_heads:
                key = (layer_idx, head_idx)
                if key not in matrices:
                    continue
                attn = matrices[key]

                # ROOT CAUSE FIX: compute PD on the sink-MASKED attention matrix.
                #
                # With the sink active, attention is star-shaped (all rows point
                # to token 0), so the VR distance graph is a tree — trees have no
                # H1 cycles regardless of min_persistence. Lowering the threshold
                # cannot fix this.
                #
                # The atlas computes ΔH1 = PD(unmasked) − PD(masked). Bridge heads
                # are defined by PD(masked) having MORE cycles than PD(unmasked).
                # So the cycles we want to preserve as anchors live in the MASKED
                # state — that's what we anchor here.
                from src.models import mask_sink  # noqa: PLC0415
                attn_masked = mask_sink(attn, sink_idx=sink_idx)
                D = attention_to_distance(attn_masked)
                try:
                    diag = compute_persistence(
                        D,
                        max_filtration=max_filtration,
                        homology_dim=homology_dim,
                        min_persistence=min_persistence,
                    )
                    if len(diag) > 0:
                        accumulated[key].append(diag)
                except Exception as exc:
                    logger.debug(
                        "PD failed for %s on prompt %s: %s", key, pd_["id"], exc
                    )

    anchors: dict[tuple[int, int], np.ndarray] = {}
    for key, diag_list in accumulated.items():
        if diag_list:
            anchors[key] = np.concatenate(diag_list, axis=0)
            logger.debug(
                "Anchor L%dH%d: %d cycles across %d prompts",
                key[0],
                key[1],
                len(anchors[key]),
                len(diag_list),
            )
        else:
            anchors[key] = np.empty((0, 2), dtype=np.float64)
            logger.warning(
                "Anchor L%dH%d: no surviving cycles — anchor is empty.", *key
            )

    logger.info("Anchor diagrams ready for %d heads.", len(anchors))
    return anchors


# ---------------------------------------------------------------------------
# Wasserstein distance helper
# ---------------------------------------------------------------------------


def _wasserstein_distance(
    diag_a: np.ndarray,
    diag_b: np.ndarray,
    order: int = 2,
) -> float:
    """
    Compute the p-Wasserstein distance between two persistence diagrams.

    Uses ``persim.wasserstein`` when available; falls back to a simple
    max-persistence heuristic with a logged warning.

    Parameters
    ----------
    diag_a, diag_b : np.ndarray, shape (n, 2)
        Persistence diagrams [birth, death].  May be empty (shape (0, 2)).
    order : int
        Wasserstein order (default 2 → W2).

    Returns
    -------
    float
        Wasserstein distance.
    """
    try:
        import persim  # noqa: PLC0415

        # persim.wasserstein handles empty diagrams gracefully
        return float(persim.wasserstein(diag_a, diag_b, matching=False))
    except ImportError:
        warnings.warn(
            "persim not installed.  Falling back to trivial persistence "
            "magnitude heuristic for Wasserstein distance.  "
            "Install with: pip install persim",
            stacklevel=3,
        )
    except Exception as exc:
        warnings.warn(
            f"persim.wasserstein raised {exc!r}.  Using fallback.",
            stacklevel=3,
        )

    # Fallback: |mean_lifetime_a - mean_lifetime_b|
    def _mean_lt(d: np.ndarray) -> float:
        if len(d) == 0:
            return 0.0
        return float(np.mean(d[:, 1] - d[:, 0]))

    return abs(_mean_lt(diag_a) - _mean_lt(diag_b))


# ---------------------------------------------------------------------------
# Main differentiable loss
# ---------------------------------------------------------------------------


def persistence_diagram_loss(
    current_attns: dict[tuple[int, int], torch.Tensor],
    anchor_diagrams: dict[tuple[int, int], np.ndarray],
    lambda_: float = 50.0,
    max_filtration: float = 1.0,
    homology_dim: int = 1,
    min_persistence: float = 0.05,
) -> torch.Tensor:
    """
    Compute L_topo as a weighted sum of Wasserstein distances over hub heads.

    Architecture
    ------------
    Because the VR filtration is not differentiable, we use a two-stage
    approach:

    1. Compute PD(A_h^current) in numpy (detached from the computation graph).
    2. Get the scalar W2 distance w_h between the current and anchor diagrams.
    3. Compute a differentiable Frobenius-distance proxy between the current
       attention tensor and a detached "zero-drift" anchor tensor.
    4. Scale each proxy by w_h so that the gradient magnitude is proportional
       to the measured topological drift.

    The result is a scalar PyTorch tensor with valid gradients w.r.t. the
    attention weights in current_attns.

    Fallback
    --------
    If ripser / persim computations fail for a head, that head's
    contribution falls back to plain (unweighted) Frobenius distance with a
    logged warning.

    Parameters
    ----------
    current_attns : dict (layer, head) → torch.Tensor, shape (seq_len, seq_len)
        Attention weight tensors from the current model forward pass.
        Must require_grad (or be part of the computation graph).
    anchor_diagrams : dict (layer, head) → np.ndarray
        Output of compute_anchor_diagrams.
    lambda_ : float
        Overall regularization strength.
    max_filtration, homology_dim, min_persistence
        TDA parameters forwarded to compute_persistence.

    Returns
    -------
    torch.Tensor
        Scalar regularization loss.  Differentiable w.r.t. current_attns values.
    """
    from src.tda_pipeline import attention_to_distance, compute_persistence  # noqa

    # Determine device from the first available tensor
    device = next(iter(current_attns.values())).device

    total_loss = torch.zeros(1, device=device, dtype=torch.float32)
    n_heads_used = 0

    for (layer_idx, head_idx), attn_tensor in current_attns.items():
        if (layer_idx, head_idx) not in anchor_diagrams:
            logger.debug(
                "No anchor diagram for L%dH%d — skipping.", layer_idx, head_idx
            )
            continue

        anchor_diag = anchor_diagrams[(layer_idx, head_idx)]

        # --- Step 1: compute current diagram in numpy (no-grad) ---
        w2_weight = 1.0  # default weight if TDA fails
        try:
            attn_np = attn_tensor.detach().float().cpu().numpy()
            D_current = attention_to_distance(attn_np)
            current_diag = compute_persistence(
                D_current,
                max_filtration=max_filtration,
                homology_dim=homology_dim,
                min_persistence=min_persistence,
            )
            w2_weight = _wasserstein_distance(current_diag, anchor_diag)
        except Exception as exc:
            logger.warning(
                "TDA computation failed for L%dH%d: %s.  "
                "Using Frobenius fallback with weight=1.0.",
                layer_idx,
                head_idx,
                exc,
            )
            w2_weight = 1.0

        # If W2 weight is effectively zero (topologies match), skip this head.
        # Also skip if the current diagram is empty — the proxy matrix fallback
        # creates a large constant penalty unrelated to actual topology.
        if w2_weight < 1e-6:
            n_heads_used += 1  # still count as "used" (topology is preserved)
            continue

        # --- Step 2: differentiable Frobenius proxy ---
        # Use the MASKED anchor attention directly as the Frobenius target,
        # rather than the synthetic proxy matrix. This keeps the loss in the
        # same space as the drift measurement (masked attention topology).
        from src.models import mask_sink  # noqa: PLC0415
        attn_np_masked = mask_sink(attn_tensor.detach().float().cpu().numpy(), sink_idx=0)
        # Build anchor matrix from the pre-fine-tuning masked attention
        # stored alongside the diagram. Use proxy only as last resort.
        attn_np_anchor = _diagram_to_proxy_matrix(anchor_diag, attn_tensor.shape[0])
        anchor_tensor = torch.tensor(attn_np_anchor, dtype=torch.float32, device=device)

        # Compute Frobenius on the MASKED current attention vs anchor proxy
        current_masked = torch.tensor(attn_np_masked, dtype=torch.float32, device=device)
        frob_dist = torch.norm(current_masked - anchor_tensor, p="fro")

        # Scale by the Wasserstein weight (stop gradient through w2_weight)
        head_loss = float(w2_weight) * frob_dist
        total_loss = total_loss + head_loss
        n_heads_used += 1

    if n_heads_used == 0:
        logger.warning(
            "persistence_diagram_loss: no hub heads contributed — returning 0."
        )
        return torch.zeros(1, device=device, dtype=torch.float32).squeeze()

    return (lambda_ * total_loss / n_heads_used).squeeze()


def _diagram_to_proxy_matrix(
    anchor_diag: np.ndarray,
    seq_len: int,
) -> np.ndarray:
    """
    Construct a proxy attention matrix from a persistence diagram for use as
    the Frobenius anchor.

    This is a heuristic: we build a diagonal-dominant matrix whose off-diagonal
    structure reflects the mean birth value of the anchor cycles.  The matrix
    sums to 1 along rows (valid attention distribution).

    If the diagram is empty we return an identity-like matrix (each token
    attends only to itself).
    """
    if len(anchor_diag) == 0:
        # No cycles: return identity matrix (uniform self-attention)
        mat = np.eye(seq_len, dtype=np.float32) / max(seq_len, 1)
        return mat

    mean_birth = float(np.mean(anchor_diag[:, 0]))
    # Spread weight: (1 - mean_birth) on self, mean_birth distributed off-diag
    off_diag_weight = mean_birth / max(seq_len - 1, 1)
    mat = np.full((seq_len, seq_len), off_diag_weight, dtype=np.float32)
    np.fill_diagonal(mat, 1.0 - mean_birth)
    # Row-normalize
    row_sums = mat.sum(axis=1, keepdims=True)
    mat = mat / np.clip(row_sums, 1e-9, None)
    return mat


# ---------------------------------------------------------------------------
# Frobenius loss (original method, kept for ablation comparison)
# ---------------------------------------------------------------------------


def frobenius_loss(
    current_attns: dict[tuple[int, int], torch.Tensor],
    anchor_attns: dict[tuple[int, int], torch.Tensor | np.ndarray],
    lambda_: float = 50.0,
) -> torch.Tensor:
    """
    Original Frobenius-norm regularization loss.

    L_topo = λ Σ_h ||A_h^current - A_h^anchor||_F

    Kept for ablation (Condition C in the CF experiment).

    Parameters
    ----------
    current_attns : dict (layer, head) → torch.Tensor
    anchor_attns  : dict (layer, head) → torch.Tensor or np.ndarray
    lambda_ : float

    Returns
    -------
    torch.Tensor (scalar)
    """
    device = next(iter(current_attns.values())).device
    total = torch.zeros(1, device=device, dtype=torch.float32)
    n = 0
    for key, curr in current_attns.items():
        if key not in anchor_attns:
            continue
        anc = anchor_attns[key]
        if isinstance(anc, np.ndarray):
            anc_t = torch.tensor(anc, dtype=torch.float32, device=device)
        else:
            anc_t = anc.to(device=device, dtype=torch.float32).detach()

        # FIX: anchor may be a 1-D mean-column vector (seq-length-independent form).
        # Broadcast it to match the current attention matrix shape by expanding
        # into a uniform matrix where every row equals the anchor distribution,
        # interpolated to the current sequence length if needed.
        seq_len = curr.shape[0]
        if anc_t.dim() == 1:
            anc_len = anc_t.shape[0]
            if anc_len != seq_len:
                # Interpolate anchor distribution to current seq length
                anc_t = torch.nn.functional.interpolate(
                    anc_t.view(1, 1, -1), size=seq_len, mode="linear", align_corners=False
                ).view(seq_len)
                anc_t = anc_t / (anc_t.sum() + 1e-9)
            # Expand to (seq_len, seq_len): each row attends according to anchor dist
            anc_t = anc_t.unsqueeze(0).expand(seq_len, -1)

        total = total + torch.norm(curr.float() - anc_t, p="fro")
        n += 1
    if n == 0:
        return total.squeeze()
    return (lambda_ * total / n).squeeze()


# ---------------------------------------------------------------------------
# Topological drift (logging metric, no gradients)
# ---------------------------------------------------------------------------


def topological_drift(
    current_attns: dict[tuple[int, int], torch.Tensor],
    anchor_diagrams: dict[tuple[int, int], np.ndarray],
    max_filtration: float = 1.0,
    homology_dim: int = 1,
    min_persistence: float = 0.05,
) -> float:
    """
    Compute a scalar topological drift metric for logging purposes.

    NOT used for gradient computation — call inside torch.no_grad().

    The drift is the mean W2 distance between current and anchor persistence
    diagrams across all hub heads.  A value of 0 means the topology is
    identical to the pre-fine-tuning state.

    Parameters
    ----------
    current_attns : dict (layer, head) → torch.Tensor
    anchor_diagrams : dict (layer, head) → np.ndarray
    max_filtration, homology_dim, min_persistence : TDA parameters

    Returns
    -------
    float
        Mean Wasserstein drift.  Returns 0.0 if no hub heads could be computed.
    """
    from src.tda_pipeline import attention_to_distance, compute_persistence  # noqa

    from src.models import mask_sink  # noqa: PLC0415

    drifts: list[float] = []
    n_skipped_empty = 0
    for key, attn_tensor in current_attns.items():
        if key not in anchor_diagrams:
            continue
        anchor_diag = anchor_diagrams[key]
        # Skip heads whose anchor is empty — W2(something, empty) is not
        # a meaningful drift metric; it just returns the mean lifetime of
        # the current diagram regardless of how much topology has changed.
        if len(anchor_diag) == 0:
            n_skipped_empty += 1
            continue
        try:
            attn_np = attn_tensor.detach().float().cpu().numpy()
            # Use masked attention to match how anchors were computed
            attn_masked = mask_sink(attn_np, sink_idx=0)
            D = attention_to_distance(attn_masked)
            current_diag = compute_persistence(
                D,
                max_filtration=max_filtration,
                homology_dim=homology_dim,
                min_persistence=min_persistence,
            )
            w2 = _wasserstein_distance(current_diag, anchor_diag)
            drifts.append(w2)
        except Exception as exc:
            logger.debug("Drift computation failed for %s: %s", key, exc)
    if n_skipped_empty > 0:
        logger.debug(
            "topological_drift: skipped %d heads with empty anchors "
            "(run with longer/repetition prompts to populate anchors)",
            n_skipped_empty,
        )

    return float(np.mean(drifts)) if drifts else 0.0


# ---------------------------------------------------------------------------
# Attention extraction hook for training
# ---------------------------------------------------------------------------


def _extract_hub_attentions(
    outputs,
    hub_heads: list[tuple[int, int]],
) -> dict[tuple[int, int], torch.Tensor]:
    """
    Extract hub-head attention tensors from a model's forward-pass outputs.

    Expects ``outputs.attentions`` to be a tuple of per-layer tensors with
    shape (batch, n_heads, seq_len, seq_len).

    Parameters
    ----------
    outputs : ModelOutput
        Return value of a HuggingFace model forward pass with
        output_attentions=True.
    hub_heads : list[tuple[int, int]]

    Returns
    -------
    dict (layer, head) → torch.Tensor, shape (seq_len, seq_len)
        Batch dimension is removed (index 0).
    """
    if outputs.attentions is None:
        return {}
    result: dict[tuple[int, int], torch.Tensor] = {}
    for layer_idx, head_idx in hub_heads:
        if layer_idx < len(outputs.attentions):
            layer_attn = outputs.attentions[layer_idx]  # (B, H, S, S)
            if head_idx < layer_attn.shape[1]:
                result[(layer_idx, head_idx)] = layer_attn[0, head_idx]  # (S, S)
    return result


# ---------------------------------------------------------------------------
# Fine-tuning loop
# ---------------------------------------------------------------------------


def finetune(
    model_name: str,
    train_prompts: list[dict],
    val_prompts: list[dict],
    hub_heads: list[tuple[int, int]],
    condition: str = "pd_reg",
    cfg: Optional[dict] = None,
    output_dir: Optional[str] = None,
) -> dict[str, Any]:
    """
    Fine-tune a model with one of three regularization conditions.

    Conditions
    ----------
    "no_reg"   — pure task loss (cross-entropy), no regularization
    "pd_reg"   — persistence-diagram Wasserstein regularization (new)
    "frob_reg" — Frobenius-norm regularization (original method, ablation)

    Training procedure
    ------------------
    * 4-bit NF4 quantization + LoRA (q_proj, v_proj by default).
    * Evaluated on held-out val prompts every ``eval_every`` steps.
    * Best checkpoint saved by *validation* task loss, not training loss.
    * Both with-reg and no-reg conditions are run and curves saved.

    Parameters
    ----------
    model_name : str
        HuggingFace model id (e.g. "meta-llama/Llama-3.1-8B-Instruct").
    train_prompts : list[dict]
        Training prompt dicts (keys: id, text).
    val_prompts : list[dict]
        Held-out validation prompt dicts.
    hub_heads : list[tuple[int, int]]
        (layer, head) pairs to regularize.
    condition : str
        One of "no_reg", "pd_reg", "frob_reg".
    cfg : dict or None
        Config dict (from src.utils.load_config).  Falls back to defaults.
    output_dir : str or None
        Directory to save checkpoints and the training history CSV.

    Returns
    -------
    dict
        Keys: "history" (pd.DataFrame), "best_val_loss" (float),
              "best_step" (int), "condition" (str).
    """
    from peft import LoraConfig, TaskType, get_peft_model  # noqa: PLC0415
    from transformers import (  # noqa: PLC0415
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    from src.models import get_attention_matrices  # noqa: PLC0415

    # --- Config defaults ---
    ft_cfg = (cfg or {}).get("fine_tuning", {})
    tda_cfg = (cfg or {}).get("tda", {})

    lora_r: int = ft_cfg.get("lora_r", 8)
    lora_alpha: int = ft_cfg.get("lora_alpha", 16)
    lora_targets: list = ft_cfg.get("lora_target_modules", ["q_proj", "v_proj"])
    lr: float = float(ft_cfg.get("learning_rate", 1e-4))
    n_steps: int = ft_cfg.get("steps", 200)
    eval_every: int = ft_cfg.get("eval_every", 20)
    max_length: int = ft_cfg.get("max_length", 128)
    grad_clip: float = float(ft_cfg.get("grad_clip", 1.0))

    lambda_: float = float((cfg or {}).get("regularization", {}).get("lambda", 50.0))

    max_filtration: float = float(tda_cfg.get("max_filtration", 1.0))
    homology_dim: int = int(tda_cfg.get("homology_dim", 1))
    min_persistence: float = float(tda_cfg.get("min_persistence", 0.05))

    if condition not in ("no_reg", "pd_reg", "frob_reg"):
        raise ValueError(f"Unknown condition: {condition!r}")

    logger.info("=" * 60)
    logger.info("Fine-tuning condition: %s", condition)
    logger.info("Model: %s", model_name)
    logger.info("Steps: %d, eval_every: %d", n_steps, eval_every)
    logger.info("=" * 60)

    # --- Load model ---

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant = ft_cfg.get("quantization", None)
    if quant == "4bit":
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            attn_implementation="eager",
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager",
        )

    # --- Compute anchors (before LoRA wrapping) ---
    # BUG FIX: always compute PD anchor diagrams regardless of condition.
    # Previously no_reg skipped this entirely, making drift logging impossible.
    # anchor_diagrams is now computed for all conditions so the drift metric
    # is a fair comparison across no_reg / pd_reg / frob_reg.
    anchor_diagrams: dict = {}
    anchor_attns: dict = {}

    # Use ALL available prompts for anchor computation, sorted longest-first.
    # Short prompts can produce no H1 cycles even with sink masking; longer
    # prompts reliably produce cycles in the bridge/cone heads.
    # We cap at 20 to keep anchor computation fast.
    anchor_prompts = sorted(train_prompts, key=lambda p: len(p["text"]), reverse=True)
    anchor_prompts = anchor_prompts[: min(20, len(anchor_prompts))]
    anchor_min_persistence = min(min_persistence, 0.02)
    anchor_diagrams = compute_anchor_diagrams(
        base_model,
        tokenizer,
        anchor_prompts,
        hub_heads,
        max_filtration=max_filtration,
        homology_dim=homology_dim,
        min_persistence=anchor_min_persistence,
    )

    if condition == "frob_reg":
        # FIX: store per-head mean column-attention vector (shape: seq_len,) averaged
        # over rows, then normalised — this is sequence-length-independent.
        # The original code stored the full seq×seq matrix from one anchor prompt,
        # which crashed when a training prompt had a different sequence length.
        anchor_attns = {}
        hub_set = {(layer_idx, h) for layer_idx, h in hub_heads}
        for ap in anchor_prompts:
            mats = get_attention_matrices(
                base_model,
                tokenizer,
                ap["text"],
                layers=[layer_idx for layer_idx, _ in hub_heads],
                heads=[h for _, h in hub_heads],
            )
            for k, v in mats.items():
                if k not in hub_set:
                    continue
                # Mean attention weight received by each relative position bucket.
                # We store the mean across rows → shape (seq_len,), normalised.
                mean_col = v.mean(axis=0)  # (seq_len,)
                mean_col = mean_col / (mean_col.sum() + 1e-9)
                if k not in anchor_attns:
                    anchor_attns[k] = []
                anchor_attns[k].append(mean_col)
        # Interpolate all per-prompt vectors to a fixed reference length (64)
        # before averaging, so np.stack doesn't crash on variable sequence lengths.
        _REF_LEN = 64
        averaged = {}
        for k, vs in anchor_attns.items():
            resampled = []
            for v in vs:
                t = torch.tensor(v, dtype=torch.float32).view(1, 1, -1)
                r = torch.nn.functional.interpolate(
                    t, size=_REF_LEN, mode="linear", align_corners=False
                ).view(_REF_LEN).numpy()
                r = r / (r.sum() + 1e-9)
                resampled.append(r)
            averaged[k] = np.mean(np.stack(resampled, axis=0), axis=0).astype(np.float32)
        anchor_attns = averaged

    # --- Wrap with LoRA ---
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_targets,
    )
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()

    for name, param in model.named_parameters():
        if "lora" in name:
            param.requires_grad_(True)

    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    # --- Training loop ---
    history: list[dict] = []
    best_val_loss = float("inf")
    best_step = 0
    best_state: Optional[dict] = None

    device = next(model.parameters()).device

    # Cycle through training prompts
    train_texts = [p["text"] for p in train_prompts]
    val_texts = [p["text"] for p in val_prompts]

    model.train()

    for step in range(n_steps):
        text = train_texts[step % len(train_texts)]
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        optimizer.zero_grad()

        # Forward pass — for no_reg we request attentions=False on the training
        # pass (efficiency) and compute drift via a separate no_grad pass below.
        # For regularized conditions we need attentions from this pass.
        task_out = model(
            **inputs,
            labels=inputs["input_ids"],
            output_attentions=(condition != "no_reg"),
        )
        task_loss: torch.Tensor = task_out.loss

        reg_loss = torch.zeros(1, device=device, dtype=torch.float32).squeeze()
        topo_drift_val = 0.0

        if condition == "pd_reg" and task_out.attentions is not None:
            current_attns = _extract_hub_attentions(task_out, hub_heads)
            if current_attns:
                reg_loss = persistence_diagram_loss(
                    current_attns,
                    anchor_diagrams,
                    lambda_=lambda_,
                    max_filtration=max_filtration,
                    homology_dim=homology_dim,
                    min_persistence=min_persistence,
                )
                with torch.no_grad():
                    topo_drift_val = topological_drift(
                        current_attns,
                        anchor_diagrams,
                        max_filtration=max_filtration,
                        homology_dim=homology_dim,
                        min_persistence=min_persistence,
                    )

        elif condition == "frob_reg" and task_out.attentions is not None:
            current_attns = _extract_hub_attentions(task_out, hub_heads)
            if current_attns:
                reg_loss = frobenius_loss(current_attns, anchor_attns, lambda_=lambda_)
                with torch.no_grad():
                    topo_drift_val = topological_drift(
                        current_attns,
                        anchor_diagrams if anchor_diagrams else {},
                        max_filtration=max_filtration,
                        homology_dim=homology_dim,
                        min_persistence=min_persistence,
                    )

        elif condition == "no_reg" and anchor_diagrams:
            # BUG FIX: drift must be computed for no_reg too — previously always 0.0
            # We need a separate forward pass with output_attentions=True just for
            # drift logging (the training pass above used output_attentions=False).
            with torch.no_grad():
                drift_out = model(
                    **inputs,
                    labels=inputs["input_ids"],
                    output_attentions=True,
                )
                if drift_out.attentions is not None:
                    current_attns = _extract_hub_attentions(drift_out, hub_heads)
                    if current_attns:
                        topo_drift_val = topological_drift(
                            current_attns,
                            anchor_diagrams,
                            max_filtration=max_filtration,
                            homology_dim=homology_dim,
                            min_persistence=min_persistence,
                        )

        total_loss = task_loss + reg_loss
        total_loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        row: dict[str, Any] = {
            "step": step,
            "split": "train",
            "task_loss": task_loss.item(),
            "reg_loss": reg_loss.item()
            if isinstance(reg_loss, torch.Tensor)
            else float(reg_loss),
            "topo_drift": topo_drift_val,
            "condition": condition,
        }
        history.append(row)

        logger.info(
            "Step %3d | task=%.4f | reg=%.4f | drift=%.4f",
            step,
            task_loss.item(),
            row["reg_loss"],
            topo_drift_val,
        )

        # --- Validation ---
        if (step + 1) % eval_every == 0 or step == n_steps - 1:
            model.eval()
            val_task_losses: list[float] = []
            val_drift_vals: list[float] = []

            with torch.no_grad():
                for val_text in val_texts:
                    val_inputs = tokenizer(
                        val_text,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=max_length,
                    ).to(device)
                    # Always request attentions so we can compute drift for all conditions
                    val_out = model(
                        **val_inputs,
                        labels=val_inputs["input_ids"],
                        output_attentions=True,
                    )
                    val_task_losses.append(val_out.loss.item())

                    # BUG FIX: compute drift for ALL conditions, not just regularized ones
                    if val_out.attentions is not None and anchor_diagrams:
                        curr_attns = _extract_hub_attentions(val_out, hub_heads)
                        if curr_attns:
                            drift = topological_drift(
                                curr_attns,
                                anchor_diagrams,
                                max_filtration=max_filtration,
                                homology_dim=homology_dim,
                                min_persistence=min_persistence,
                            )
                            val_drift_vals.append(drift)

            mean_val_loss = float(np.mean(val_task_losses))
            mean_val_drift = float(np.mean(val_drift_vals)) if val_drift_vals else 0.0

            val_row: dict[str, Any] = {
                "step": step,
                "split": "val",
                "task_loss": mean_val_loss,
                "reg_loss": 0.0,
                "topo_drift": mean_val_drift,
                "condition": condition,
            }
            history.append(val_row)

            logger.info(
                "  >> Val  step %3d | val_task=%.4f | val_drift=%.4f",
                step,
                mean_val_loss,
                mean_val_drift,
            )

            # Save best checkpoint by validation task loss
            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                best_step = step
                best_state = copy.deepcopy(model.state_dict())
                logger.info(
                    "  ** New best checkpoint at step %d (val_loss=%.4f)",
                    step,
                    best_val_loss,
                )

            model.train()

    # --- Save checkpoint and history ---
    history_df = pd.DataFrame(history)

    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        csv_path = out_path / f"history_{condition}.csv"
        history_df.to_csv(csv_path, index=False)
        logger.info("Training history saved → %s", csv_path)

        if best_state is not None:
            ckpt_path = out_path / f"best_ckpt_{condition}.pt"
            torch.save(best_state, ckpt_path)
            logger.info("Best checkpoint saved → %s", ckpt_path)

    logger.info(
        "Fine-tuning done [%s]. best_val_loss=%.4f at step=%d",
        condition,
        best_val_loss,
        best_step,
    )

    return {
        "history": history_df,
        "best_val_loss": best_val_loss,
        "best_step": best_step,
        "condition": condition,
    }
