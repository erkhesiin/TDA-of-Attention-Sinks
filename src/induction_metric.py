"""
src/induction_metric.py
-----------------------
Quantify induction strength as a measurable, aggregable statistic.

Replaces the single skeleton-plot visualization from visualize_skeleton.py as
the primary evidence for the cone / induction-suppression claim.

Definition
----------
For a prompt with a repeated sub-sequence, the **induction score** of a head is
the fraction of attention weight (among the top-k outgoing edges per token) that
flows from token t to the token that follows the *prior* occurrence of the same
token identity:

    induction_score(h, prompt) =
        Σ_t  A_h[t, prev_next(t)]
        ──────────────────────────
        Σ_t  Σ_{top-k}  A_h[t, :]

where prev_next(t) = p + offset, p being the most recent prior position at
which the same token appeared (offset = 1 by default, i.e. "next-token"
prediction).

A score close to 1 means the head is nearly a pure induction head; a score
close to 0 means it attends elsewhere.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def find_repetition_pairs(
    token_ids: list[int],
    offset: int = 1,
) -> list[tuple[int, int]]:
    """
    For each token position *t* that has a prior occurrence *p* (p < t,
    same token id), return the pair (t, p + offset).

    These are the "induction pairs": token t should attend to the token
    immediately *after* the previous time this same token appeared, because
    that is where a perfect induction head would look to predict the next
    token after t.

    Only the *most recent* prior occurrence is used (last-occurrence
    semantics), which matches how induction heads are empirically observed
    to behave.

    Parameters
    ----------
    token_ids : list[int]
        Flat list of integer token ids for the prompt.
    offset : int
        How many positions after the prior occurrence to target (default 1).

    Returns
    -------
    list of (t, target) tuples
        t      : current position
        target : p + offset  (the position t "should" attend to)

    Notes
    -----
    Pairs where p + offset >= t (target is not in the past) or
    p + offset >= len(token_ids) are silently excluded.
    """
    n = len(token_ids)
    # last_seen[token_id] = most recent position p where token appeared
    last_seen: dict[int, int] = {}
    pairs: list[tuple[int, int]] = []

    for t, tid in enumerate(token_ids):
        if tid in last_seen:
            p = last_seen[tid]
            target = p + offset
            # target must be a valid past position (strictly before t)
            if 0 <= target < t:
                pairs.append((t, target))
        # Update last-seen *after* checking, so we use the prior occurrence
        last_seen[tid] = t

    return pairs


def induction_score(
    attn: np.ndarray,
    token_ids: list[int],
    top_k: int = 10,
    offset: int = 1,
) -> float:
    """
    Compute the induction score for a single attention head on a single prompt.

    The score is the fraction of the *top-k attention mass* (summed over all
    query positions) that lands on the induction target prev_next(t).

    Parameters
    ----------
    attn : np.ndarray
        Shape (seq_len, seq_len). Post-softmax attention weights.
        Rows are query positions, columns are key positions.
    token_ids : list[int]
        Flat token id list for the prompt (len must equal seq_len).
    top_k : int
        Number of top attention edges to consider *per query token*.
    offset : int
        Passed to find_repetition_pairs.

    Returns
    -------
    float
        Induction score in [0, 1].  Returns 0.0 if there are no repetition
        pairs in the prompt (i.e. nothing to measure).
    """
    seq_len = attn.shape[0]
    if len(token_ids) != seq_len:
        raise ValueError(f"len(token_ids)={len(token_ids)} != seq_len={seq_len}")

    pairs = find_repetition_pairs(token_ids, offset=offset)
    if not pairs:
        logger.debug("No repetition pairs found; induction_score=0.0")
        return 0.0

    # Build a fast set lookup: t → target
    induction_targets: dict[int, int] = {t: target for t, target in pairs}

    numerator = 0.0
    denominator = 0.0

    for t in range(seq_len):
        row = attn[t]  # shape (seq_len,)

        # Top-k indices by descending attention weight
        k = min(top_k, seq_len)
        top_indices = np.argpartition(row, -k)[-k:]  # unordered top-k
        top_mass = float(row[top_indices].sum())
        denominator += top_mass

        # If t is an induction query position, add the weight to its target
        if t in induction_targets:
            target = induction_targets[t]
            if 0 <= target < seq_len:
                numerator += float(row[target])

    if denominator == 0.0:
        return 0.0

    return numerator / denominator


# ---------------------------------------------------------------------------
# Aggregate comparison: masked vs. unmasked
# ---------------------------------------------------------------------------


def compare_induction_masked_vs_unmasked(
    model,
    tokenizer,
    prompts: list[dict],
    target_layer: int,
    target_head: int,
    top_k: int = 10,
    offset: int = 1,
    sink_idx: int = 0,
) -> pd.DataFrame:
    """
    For each prompt, compute the induction score with the sink active (normal)
    and with the sink masked out.

    Measures the cone / induction-suppression claim statistically: if removing
    the attention sink *increases* the induction score, the sink was suppressing
    induction (cone role); if removing it *decreases* the score, the sink was
    facilitating it (bridge-like behaviour for induction).

    Parameters
    ----------
    model : AutoModelForCausalLM
        Loaded HuggingFace model (output_attentions must be supported).
    tokenizer : AutoTokenizer
    prompts : list[dict]
        Each dict must have at least 'id' and 'text' keys (prompts.json format).
    target_layer : int
        Layer index of the head to analyse.
    target_head : int
        Head index to analyse.
    top_k : int
        Passed to induction_score.
    offset : int
        Passed to induction_score and find_repetition_pairs.
    sink_idx : int
        Token position of the attention sink (default 0 = BOS).

    Returns
    -------
    pd.DataFrame
        Columns: prompt_id, category, score_normal, score_masked, delta
        One row per prompt.

    Also logs:
        - Mean and 95% CI for score_normal and score_masked
        - Paired t-test p-value (two-tailed)
    """
    # Import here to avoid circular dependency at module level
    from src.models import get_attention_matrices, mask_sink  # noqa: PLC0415

    records: list[dict] = []

    for prompt_dict in prompts:
        pid = prompt_dict.get("id", "unknown")
        category = prompt_dict.get("category", "unknown")
        text = prompt_dict["text"]

        # Get token ids for induction pair detection
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        token_ids: list[int] = enc["input_ids"][0].tolist()

        # Retrieve attention matrices for the target head only
        matrices = get_attention_matrices(
            model,
            tokenizer,
            text,
            layers=[target_layer],
            heads=[target_head],
        )
        attn_normal = matrices[(target_layer, target_head)]
        attn_masked = mask_sink(attn_normal, sink_idx=sink_idx)

        score_normal = induction_score(
            attn_normal, token_ids, top_k=top_k, offset=offset
        )
        score_masked = induction_score(
            attn_masked, token_ids, top_k=top_k, offset=offset
        )
        delta = score_masked - score_normal

        records.append(
            {
                "prompt_id": pid,
                "category": category,
                "score_normal": score_normal,
                "score_masked": score_masked,
                "delta": delta,
            }
        )
        logger.debug(
            f"  {pid}: normal={score_normal:.4f}  masked={score_masked:.4f}"
            f"  delta={delta:+.4f}"
        )

    df = pd.DataFrame(records)

    if len(df) >= 2:
        _report_statistics(df, target_layer, target_head)

    return df


def _report_statistics(df: pd.DataFrame, layer: int, head: int) -> None:
    """Log summary statistics for a compare_induction_masked_vs_unmasked result."""
    n = len(df)
    alpha = 0.05

    for col in ("score_normal", "score_masked", "delta"):
        vals = df[col].values
        mean = vals.mean()
        se = vals.std(ddof=1) / np.sqrt(n)
        t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
        ci_lo = mean - t_crit * se
        ci_hi = mean + t_crit * se
        logger.info(
            f"L{layer}H{head} {col:14s}: mean={mean:.4f}  "
            f"95%CI=[{ci_lo:.4f}, {ci_hi:.4f}]"
        )

    # Paired t-test: does masking the sink significantly change induction score?
    t_stat, p_val = stats.ttest_rel(
        df["score_masked"].values, df["score_normal"].values
    )
    logger.info(
        f"L{layer}H{head} paired t-test (masked vs normal): "
        f"t={t_stat:.4f}, p={p_val:.4f}  (n={n})"
    )


# ---------------------------------------------------------------------------
# Bulk head survey
# ---------------------------------------------------------------------------


def survey_induction_heads(
    model,
    tokenizer,
    prompts: list[dict],
    layers: Optional[list[int]] = None,
    heads: Optional[list[int]] = None,
    top_k: int = 10,
    offset: int = 1,
    categories: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Compute the mean induction score for every (layer, head) pair across all
    prompts (optionally filtered by category).

    Useful for identifying which heads are strongest induction heads before
    running the masked-vs-unmasked comparison.

    Parameters
    ----------
    model, tokenizer : as in compare_induction_masked_vs_unmasked
    prompts : list[dict]
    layers, heads : optional filter lists
    top_k, offset : as in induction_score
    categories : optional list of prompt categories to restrict to

    Returns
    -------
    pd.DataFrame
        Columns: layer, head, mean_score, std_score, n_prompts
        Sorted by mean_score descending.
    """
    from src.models import get_attention_matrices  # noqa: PLC0415

    if categories:
        prompts = [p for p in prompts if p.get("category") in categories]
    if not prompts:
        raise ValueError("No prompts remain after category filtering.")

    # Pre-tokenize prompts so we can reuse token_ids
    tokenized: list[tuple[str, str, list[int]]] = []
    for pd_ in prompts:
        enc = tokenizer(
            pd_["text"], return_tensors="pt", truncation=True, max_length=512
        )
        tokenized.append((pd_["id"], pd_["text"], enc["input_ids"][0].tolist()))

    # Determine layer / head ranges from the first prompt's matrices
    sample_matrices = get_attention_matrices(model, tokenizer, tokenized[0][1])
    all_keys = list(sample_matrices.keys())
    if layers is not None:
        all_keys = [(l, h) for (l, h) in all_keys if l in layers]
    if heads is not None:
        all_keys = [(l, h) for (l, h) in all_keys if h in heads]

    records: list[dict] = []

    for layer_idx, head_idx in all_keys:
        scores: list[float] = []
        for pid, text, token_ids in tokenized:
            mats = get_attention_matrices(
                model, tokenizer, text, layers=[layer_idx], heads=[head_idx]
            )
            attn = mats[(layer_idx, head_idx)]
            s = induction_score(attn, token_ids, top_k=top_k, offset=offset)
            scores.append(s)

        records.append(
            {
                "layer": layer_idx,
                "head": head_idx,
                "mean_score": float(np.mean(scores)),
                "std_score": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
                "n_prompts": len(scores),
            }
        )
        logger.debug(f"L{layer_idx:02d}H{head_idx:02d} induction={np.mean(scores):.4f}")

    df = (
        pd.DataFrame(records)
        .sort_values("mean_score", ascending=False)
        .reset_index(drop=True)
    )
    return df
