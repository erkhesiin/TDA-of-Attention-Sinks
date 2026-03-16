"""
experiments/02_induction_suppression.py
---------------------------------------
Experiment 02: Induction Suppression Analysis.

Pipeline
--------
1. Load the model and atlas results produced by experiment 01.
2. Identify the top cone heads (highest mean ΔH1 in deep layers).
3. Run compare_induction_masked_vs_unmasked for those heads over the
   repetition prompt category.
4. Report mean induction score (normal vs. masked) with 95% CI and a
   paired t-test p-value.
5. Generate Figure 3 (induction comparison box plot) and Figure 5
   (attention skeleton grid for 3 example prompts).

Usage
-----
    # Run on default model (first in config)
    python experiments/02_induction_suppression.py

    # Run on a specific model alias
    python experiments/02_induction_suppression.py --model llama3

    # Point at a pre-computed atlas CSV instead of re-running experiment 01
    python experiments/02_induction_suppression.py --atlas results/llama3/atlas.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so ``src`` is importable
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.utils import get_logger, load_config, load_prompts, results_dir, set_seed

log = get_logger(__name__, level=logging.INFO)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 02 — Induction Suppression"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model alias from config (e.g. 'llama3').  Defaults to the first model.",
    )
    parser.add_argument(
        "--atlas",
        type=str,
        default=None,
        help=(
            "Path to a pre-computed aggregated atlas CSV "
            "(output of experiment 01).  When not provided the script looks "
            "for results/<alias>/atlas.csv."
        ),
    )
    parser.add_argument(
        "--top-n-heads",
        type=int,
        default=3,
        dest="top_n_heads",
        help="Number of top cone heads to analyse (default: 3).",
    )
    parser.add_argument(
        "--deep-layer-frac",
        type=float,
        default=0.5,
        dest="deep_layer_frac",
        help=(
            "Fraction of layers considered 'deep' when selecting cone heads. "
            "E.g. 0.5 means the top half of the model's layers (default: 0.5)."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML.  Defaults to config/default.yaml.",
    )
    parser.add_argument(
        "--no-skeleton",
        action="store_true",
        dest="no_skeleton",
        help="Skip generating Figure 5 (skeleton grid) — faster for quick runs.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helper: pick top cone heads from aggregated atlas
# ---------------------------------------------------------------------------


def _select_cone_heads(
    agg_df: pd.DataFrame,
    n_layers: int,
    top_n: int = 3,
    deep_frac: float = 0.5,
) -> list[tuple[int, int]]:
    """
    Return the *top_n* (layer, head) pairs with the highest mean_delta_h1
    that are also located in the deep half of the model.

    Parameters
    ----------
    agg_df : pd.DataFrame
        Aggregated atlas (columns: layer, head, mean_delta_h1, role, …).
    n_layers : int
        Total number of transformer layers in the model.
    top_n : int
        How many heads to return.
    deep_frac : float
        Layer index threshold fraction (e.g. 0.5 → only layers ≥ n_layers/2).

    Returns
    -------
    list of (layer, head)
    """
    deep_cutoff = int(n_layers * deep_frac)
    deep_cones = agg_df[
        (agg_df["role"] == "cone") & (agg_df["layer"] >= deep_cutoff)
    ].copy()

    if deep_cones.empty:
        log.warning(
            "No cone heads found in deep layers (>= layer %d).  "
            "Falling back to all-layer cone selection.",
            deep_cutoff,
        )
        deep_cones = agg_df[agg_df["role"] == "cone"].copy()

    if deep_cones.empty:
        log.error(
            "No cone heads found in atlas at all.  "
            "Run experiment 01 first and ensure the atlas has cone-classified heads."
        )
        return []

    deep_cones = deep_cones.sort_values("mean_delta_h1", ascending=False)
    selected = deep_cones.head(top_n)
    heads = list(zip(selected["layer"].tolist(), selected["head"].tolist()))
    log.info("Selected %d top cone heads: %s", len(heads), heads)
    return heads


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def _print_summary(df: pd.DataFrame, layer: int, head: int) -> None:
    """Pretty-print the induction comparison summary for one head."""
    n = len(df)
    if n < 2:
        log.warning("Too few prompts (%d) for statistical test.", n)
        return

    for col, label in [
        ("score_normal", "Normal (sink active)"),
        ("score_masked", "Masked (sink removed)"),
        ("delta", "Delta (masked - normal)"),
    ]:
        vals = df[col].values
        mean = vals.mean()
        se = vals.std(ddof=1) / np.sqrt(n)
        t_crit = stats.t.ppf(0.975, df=n - 1)
        ci_lo = mean - t_crit * se
        ci_hi = mean + t_crit * se
        log.info(
            "  %-28s mean=%.4f  95%%CI=[%.4f, %.4f]",
            label,
            mean,
            ci_lo,
            ci_hi,
        )

    t_stat, p_val = stats.ttest_rel(
        df["score_masked"].values, df["score_normal"].values
    )
    stars = (
        "***"
        if p_val < 0.001
        else "**"
        if p_val < 0.01
        else "*"
        if p_val < 0.05
        else "ns"
    )
    log.info(
        "  Paired t-test L%dH%d: t=%.4f, p=%.4f %s  (n=%d)",
        layer,
        head,
        t_stat,
        p_val,
        stars,
        n,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    # --- Resolve model ---
    models_cfg = cfg.get("models", [])
    if not models_cfg:
        log.error("No models defined in config.")
        sys.exit(1)

    if args.model:
        model_entry = next((m for m in models_cfg if m["alias"] == args.model), None)
        if model_entry is None:
            log.error("Model alias '%s' not found in config.", args.model)
            sys.exit(1)
    else:
        model_entry = models_cfg[0]

    model_name: str = model_entry["name"]
    model_alias: str = model_entry["alias"]

    log.info("=" * 60)
    log.info("Experiment 02 — Induction Suppression")
    log.info("Model: %s  (%s)", model_name, model_alias)
    log.info("=" * 60)

    # --- Load atlas ---
    if args.atlas:
        atlas_path = Path(args.atlas)
    else:
        atlas_path = results_dir(cfg, model_alias) / "atlas.csv"

    if not atlas_path.exists():
        log.error(
            "Aggregated atlas not found at %s.  "
            "Run experiment 01 first, or provide --atlas PATH.",
            atlas_path,
        )
        sys.exit(1)

    agg_df = pd.read_csv(atlas_path)
    log.info("Loaded aggregated atlas: %s  (%d rows)", atlas_path, len(agg_df))

    # --- Load model ---
    from src.models import load_model

    model, tokenizer = load_model(
        model_name, quantization=cfg.get("fine_tuning", {}).get("quantization", "4bit")
    )
    n_layers: int = model.config.num_hidden_layers

    # --- Select top cone heads ---
    cone_heads = _select_cone_heads(
        agg_df,
        n_layers=n_layers,
        top_n=args.top_n_heads,
        deep_frac=args.deep_layer_frac,
    )
    if not cone_heads:
        log.error("No cone heads available for analysis.  Exiting.")
        sys.exit(1)

    # --- Load prompts (repetition category for induction scoring) ---
    induction_cfg = cfg.get("induction", {})
    top_k: int = induction_cfg.get("top_k_edges", 10)
    offset: int = induction_cfg.get("offset", 1)

    all_prompts = load_prompts(cfg)
    repetition_prompts = [p for p in all_prompts if p["category"] == "repetition"]
    if not repetition_prompts:
        log.warning(
            "No 'repetition' category prompts found.  "
            "Using all prompts for induction scoring."
        )
        repetition_prompts = all_prompts

    log.info(
        "Using %d repetition prompts for induction analysis.", len(repetition_prompts)
    )

    # --- Run comparison for each top cone head ---
    from src.induction_metric import compare_induction_masked_vs_unmasked
    from src.visualize import plot_induction_comparison

    out_dir = results_dir(cfg, model_alias)
    all_comparison_dfs: list[pd.DataFrame] = []

    for layer_idx, head_idx in cone_heads:
        log.info("-" * 50)
        log.info("Analysing L%dH%d ...", layer_idx, head_idx)

        comparison_df = compare_induction_masked_vs_unmasked(
            model=model,
            tokenizer=tokenizer,
            prompts=repetition_prompts,
            target_layer=layer_idx,
            target_head=head_idx,
            top_k=top_k,
            offset=offset,
        )

        # Tag with head identity
        comparison_df["layer"] = layer_idx
        comparison_df["head"] = head_idx
        comparison_df["head_label"] = f"L{layer_idx}H{head_idx}"

        all_comparison_dfs.append(comparison_df)

        # Save per-head CSV
        csv_path = out_dir / f"induction_L{layer_idx}H{head_idx}.csv"
        comparison_df.to_csv(csv_path, index=False)
        log.info("Saved induction data → %s", csv_path)

        _print_summary(comparison_df, layer_idx, head_idx)

    # --- Combine and save master CSV ---
    if all_comparison_dfs:
        combined_df = pd.concat(all_comparison_dfs, ignore_index=True)
        combined_csv = out_dir / "induction_combined.csv"
        combined_df.to_csv(combined_csv, index=False)
        log.info("Saved combined induction data → %s", combined_csv)

    # --- Figure 3: Induction Score Comparison ---
    if all_comparison_dfs:
        log.info("Generating Figure 3 — Induction Score Comparison ...")

        # For each head, produce a separate figure (one box plot per head)
        for layer_idx, head_idx in cone_heads:
            head_df = next(
                (
                    d
                    for d in all_comparison_dfs
                    if d["layer"].iloc[0] == layer_idx and d["head"].iloc[0] == head_idx
                ),
                None,
            )
            if head_df is None:
                continue

            fig3_path = out_dir / f"fig3_induction_L{layer_idx}H{head_idx}.png"
            try:
                plot_induction_comparison(head_df, str(fig3_path))
                log.info("Figure 3 saved → %s", fig3_path)
            except Exception as exc:
                log.warning(
                    "Figure 3 generation failed for L%dH%d: %s",
                    layer_idx,
                    head_idx,
                    exc,
                )

    # --- Figure 5: Attention Skeleton Grid ---
    if not args.no_skeleton:
        log.info("Generating Figure 5 — Attention Skeleton Grid ...")
        _generate_skeleton_figure(
            model=model,
            tokenizer=tokenizer,
            prompts=repetition_prompts,
            cone_heads=cone_heads,
            out_dir=out_dir,
            cfg=cfg,
        )

    log.info("=" * 60)
    log.info("Experiment 02 complete.  Results in: %s", out_dir)
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# Figure 5 helper
# ---------------------------------------------------------------------------


def _generate_skeleton_figure(
    model,
    tokenizer,
    prompts: list[dict],
    cone_heads: list[tuple[int, int]],
    out_dir: Path,
    cfg: dict,
) -> None:
    """Build Figure 5 for the primary (first) cone head using 3 example prompts."""
    from src.models import get_attention_matrices, mask_sink
    from src.visualize import plot_skeleton_grid

    if not cone_heads:
        log.warning("No cone heads provided for skeleton figure.")
        return

    # Use the strongest cone head
    target_layer, target_head = cone_heads[0]

    # Pick up to 3 example prompts
    example_prompts = prompts[:3]
    if len(example_prompts) == 0:
        log.warning("No prompts available for skeleton figure.")
        return

    panels: list[dict] = []
    for pd_ in example_prompts:
        text = pd_["text"]
        try:
            matrices = get_attention_matrices(
                model,
                tokenizer,
                text,
                layers=[target_layer],
                heads=[target_head],
            )
            attn_normal = matrices[(target_layer, target_head)]
            attn_masked = mask_sink(attn_normal, sink_idx=0)

            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            token_ids = enc["input_ids"][0].tolist()
            token_strs = [tokenizer.decode([tid]) for tid in token_ids]

            panels.append(
                {
                    "attn_normal": attn_normal,
                    "attn_masked": attn_masked,
                    "token_labels": token_strs,
                    "prompt_label": f"L{target_layer}H{target_head} — {pd_['id']}",
                }
            )
        except Exception as exc:
            log.warning("Skeleton panel failed for prompt '%s': %s", pd_["id"], exc)

    if not panels:
        log.warning("No skeleton panels generated — skipping Figure 5.")
        return

    fig5_path = out_dir / f"fig5_skeleton_L{target_layer}H{target_head}.png"
    try:
        plot_skeleton_grid(panels, str(fig5_path))
        log.info("Figure 5 saved → %s", fig5_path)
    except Exception as exc:
        log.warning("Figure 5 generation failed: %s", exc)

        # Fallback: save individual single-panel skeletons
        from src.visualize import plot_skeleton

        for i, panel in enumerate(panels):
            for side, attn_key in [
                ("normal", "attn_normal"),
                ("masked", "attn_masked"),
            ]:
                path = (
                    out_dir
                    / f"fig5_skeleton_L{target_layer}H{target_head}_p{i}_{side}.png"
                )
                try:
                    plot_skeleton(
                        attn=panel[attn_key],
                        token_labels=panel["token_labels"],
                        title=f"{panel['prompt_label']} — {side}",
                        output_path=str(path),
                    )
                except Exception as exc2:
                    log.warning("Single skeleton also failed: %s", exc2)


if __name__ == "__main__":
    main()
