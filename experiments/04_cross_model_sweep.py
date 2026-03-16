"""
experiments/04_cross_model_sweep.py
------------------------------------
Experiment 04: Cross-Model Generalizability Sweep.

This is the key generalizability check missing from the original codebase.
For every model listed in config/default.yaml it:

  1. Runs the full topological atlas pipeline (Experiment 01 logic).
  2. Identifies top cone heads and runs the induction suppression comparison
     (Experiment 02 logic).
  3. Collates results into a single summary table:
       - Model alias
       - Total heads scanned
       - Fraction classified as bridge
       - Fraction classified as cone
       - Fraction classified as neutral
       - Mean stability score (consistency of role across prompts)
       - Mean ΔH1 for bridge heads
       - Mean ΔH1 for cone heads
       - Mean induction delta (masked − normal) for top cone heads
       - Whether the bridge/cone pattern is "consistent" (stability ≥ threshold)
  4. Saves the table as results/cross_model_summary.csv and prints it.
  5. Optionally re-uses pre-computed atlas CSVs (--use-cached) to skip
     re-running the TDA pipeline when results already exist.

Usage
-----
    # Full sweep of all configured models
    python experiments/04_cross_model_sweep.py

    # Only run specific model aliases
    python experiments/04_cross_model_sweep.py --models llama3 mistral

    # Re-use cached atlas results (skip TDA, only run induction scoring)
    python experiments/04_cross_model_sweep.py --use-cached

    # Quick smoke test: restrict layers/heads and use fewer prompts
    python experiments/04_cross_model_sweep.py --layers 0 8 16 24 --heads 0 1 2 3 --max-prompts 5
"""

from __future__ import annotations

import argparse
import gc
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.utils import get_logger, load_config, load_prompts, results_dir, set_seed

log = get_logger(__name__, level=logging.INFO)

# Minimum stability score for a model's bridge/cone pattern to be deemed
# "consistent" in the final summary table.
_CONSISTENCY_THRESHOLD = 0.6


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Experiment 04 — Cross-Model Sweep",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML.  Defaults to config/default.yaml.",
    )
    p.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Run only these model aliases.  Default: all models in config.",
    )
    p.add_argument(
        "--use-cached",
        action="store_true",
        dest="use_cached",
        help=(
            "Skip TDA if results/<alias>/atlas.csv already exists. "
            "Induction scoring is always re-run."
        ),
    )
    p.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Restrict TDA to specific layer indices (for quick tests).",
    )
    p.add_argument(
        "--heads",
        type=int,
        nargs="+",
        default=None,
        help="Restrict TDA to specific head indices (for quick tests).",
    )
    p.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        dest="max_prompts",
        help="Cap the number of prompts used for TDA (useful for quick tests).",
    )
    p.add_argument(
        "--top-n-heads",
        type=int,
        default=3,
        dest="top_n_heads",
        help="Number of top cone heads to use for induction scoring per model.",
    )
    p.add_argument(
        "--no-control",
        action="store_true",
        dest="no_control",
        help="Skip building the randomized-weights control atlas.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed override.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Single-model atlas pipeline (mirrors experiment 01)
# ---------------------------------------------------------------------------


def _run_atlas_for_model(
    model_cfg: dict,
    prompts: list[dict],
    tda_cfg: dict,
    run_control: bool,
    layers: Optional[list[int]],
    heads: Optional[list[int]],
    cfg: dict,
) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Build (and save) the aggregated atlas for one model.

    Returns
    -------
    agg_df : pd.DataFrame
        Aggregated atlas (layer, head, mean_delta_h1, std_delta_h1, role, stability).
    control_agg_df : pd.DataFrame or None
        Aggregated control atlas (or None if run_control=False or it failed).
    """
    from src.models import load_model
    from src.tda_pipeline import (
        aggregate_atlas,
        build_control_atlas,
        build_topological_atlas,
    )

    alias: str = model_cfg["alias"]
    model_name: str = model_cfg["name"]

    log.info("Loading model '%s' ...", model_name)
    t0 = time.perf_counter()
    model, tokenizer = load_model(model_name, quantization="4bit")
    log.info("Model loaded in %.1f s", time.perf_counter() - t0)

    tda_kwargs: dict = dict(
        sink_idx=0,
        bridge_threshold=float(tda_cfg.get("bridge_threshold", -0.03)),
        cone_threshold=float(tda_cfg.get("cone_threshold", 0.03)),
        max_filtration=float(tda_cfg.get("max_filtration", 1.0)),
        homology_dim=int(tda_cfg.get("homology_dim", 1)),
        min_persistence=float(tda_cfg.get("min_persistence", 0.05)),
        verbose=False,
    )
    if layers is not None:
        tda_kwargs["layers"] = layers
    if heads is not None:
        tda_kwargs["heads"] = heads

    log.info("Building topological atlas ...")
    atlas_df = build_topological_atlas(model, tokenizer, prompts, **tda_kwargs)
    agg_df = aggregate_atlas(atlas_df)

    out_dir = results_dir(cfg, alias)
    atlas_df.to_csv(out_dir / "atlas_raw.csv", index=False)
    agg_df.to_csv(out_dir / "atlas.csv", index=False)
    log.info("Atlas saved → %s", out_dir / "atlas.csv")

    control_agg_df = None
    if run_control:
        try:
            log.info("Building control atlas ...")
            ctrl_df = build_control_atlas(model, tokenizer, prompts, **tda_kwargs)
            control_agg_df = aggregate_atlas(ctrl_df)
            ctrl_df.to_csv(out_dir / "atlas_control_raw.csv", index=False)
            control_agg_df.to_csv(out_dir / "atlas_control.csv", index=False)
        except Exception as exc:
            log.warning("Control atlas failed for '%s': %s", alias, exc)

    # Free GPU memory
    del model
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass
    gc.collect()

    return agg_df, control_agg_df


# ---------------------------------------------------------------------------
# Single-model induction pipeline (mirrors experiment 02)
# ---------------------------------------------------------------------------


def _run_induction_for_model(
    model_cfg: dict,
    agg_df: pd.DataFrame,
    prompts: list[dict],
    top_n_heads: int,
    cfg: dict,
) -> dict:
    """
    Run the masked-vs-unmasked induction comparison for the top cone heads
    of one model.

    Returns a dict with summary statistics:
        mean_induction_normal, mean_induction_masked, mean_induction_delta,
        n_prompts, n_heads_analysed.
    """
    from src.induction_metric import compare_induction_masked_vs_unmasked
    from src.models import load_model

    alias: str = model_cfg["alias"]
    model_name: str = model_cfg["name"]

    # Identify top cone heads in deep layers
    n_layers_in_atlas = int(agg_df["layer"].max()) + 1
    deep_cutoff = n_layers_in_atlas // 2

    deep_cones = agg_df[
        (agg_df["role"] == "cone") & (agg_df["layer"] >= deep_cutoff)
    ].sort_values("mean_delta_h1", ascending=False)

    if deep_cones.empty:
        deep_cones = agg_df[agg_df["role"] == "cone"].sort_values(
            "mean_delta_h1", ascending=False
        )

    if deep_cones.empty:
        log.warning("No cone heads found for '%s' — induction scoring skipped.", alias)
        return {
            "mean_induction_normal": float("nan"),
            "mean_induction_masked": float("nan"),
            "mean_induction_delta": float("nan"),
            "n_prompts": 0,
            "n_heads_analysed": 0,
        }

    cone_heads = list(
        zip(
            deep_cones["layer"].head(top_n_heads).tolist(),
            deep_cones["head"].head(top_n_heads).tolist(),
        )
    )
    log.info("Induction scoring for '%s' — cone heads: %s", alias, cone_heads)

    # Load model
    model, tokenizer = load_model(model_name, quantization="4bit")

    induction_cfg = cfg.get("induction", {})
    top_k: int = induction_cfg.get("top_k_edges", 10)
    offset: int = induction_cfg.get("offset", 1)

    # Use repetition prompts; fall back to all prompts
    rep_prompts = [p for p in prompts if p["category"] == "repetition"]
    if not rep_prompts:
        rep_prompts = prompts

    all_deltas: list[float] = []
    all_normals: list[float] = []
    all_masked: list[float] = []

    out_dir = results_dir(cfg, alias)

    for layer_idx, head_idx in cone_heads:
        df = compare_induction_masked_vs_unmasked(
            model=model,
            tokenizer=tokenizer,
            prompts=rep_prompts,
            target_layer=layer_idx,
            target_head=head_idx,
            top_k=top_k,
            offset=offset,
        )
        df["layer"] = layer_idx
        df["head"] = head_idx
        df.to_csv(out_dir / f"induction_L{layer_idx}H{head_idx}.csv", index=False)

        all_normals.extend(df["score_normal"].tolist())
        all_masked.extend(df["score_masked"].tolist())
        all_deltas.extend(df["delta"].tolist())

    # Free GPU memory
    del model
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass
    gc.collect()

    return {
        "mean_induction_normal": float(np.mean(all_normals))
        if all_normals
        else float("nan"),
        "mean_induction_masked": float(np.mean(all_masked))
        if all_masked
        else float("nan"),
        "mean_induction_delta": float(np.mean(all_deltas))
        if all_deltas
        else float("nan"),
        "n_prompts": len(rep_prompts),
        "n_heads_analysed": len(cone_heads),
    }


# ---------------------------------------------------------------------------
# Summary row builder
# ---------------------------------------------------------------------------


def _build_summary_row(
    model_cfg: dict,
    agg_df: pd.DataFrame,
    induction_stats: dict,
    control_agg_df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Compile a single summary row for the cross-model table.

    Parameters
    ----------
    model_cfg : dict
        Config entry for this model (name, alias).
    agg_df : pd.DataFrame
        Aggregated trained-model atlas.
    induction_stats : dict
        Output of _run_induction_for_model.
    control_agg_df : pd.DataFrame or None
        Aggregated control atlas, used to compute contrast metrics.

    Returns
    -------
    dict
        One row for the cross-model summary DataFrame.
    """
    alias = model_cfg["alias"]
    model_name = model_cfg["name"]

    total_heads = len(agg_df)
    if total_heads == 0:
        return {
            "model": model_name,
            "alias": alias,
            "total_heads": 0,
            "frac_bridge": float("nan"),
            "frac_cone": float("nan"),
            "frac_neutral": float("nan"),
            "mean_stability": float("nan"),
            "mean_delta_h1_bridge": float("nan"),
            "mean_delta_h1_cone": float("nan"),
            "mean_induction_normal": float("nan"),
            "mean_induction_masked": float("nan"),
            "mean_induction_delta": float("nan"),
            "n_induction_heads": 0,
            "bridge_cone_consistent": False,
            "control_max_abs_delta": float("nan"),
            "contrast_trained_vs_control": float("nan"),
        }

    n_bridge = (agg_df["role"] == "bridge").sum()
    n_cone = (agg_df["role"] == "cone").sum()
    n_neutral = (agg_df["role"] == "neutral").sum()

    frac_bridge = n_bridge / total_heads
    frac_cone = n_cone / total_heads
    frac_neutral = n_neutral / total_heads

    mean_stability = float(agg_df["stability"].mean())

    bridge_rows = agg_df[agg_df["role"] == "bridge"]
    cone_rows = agg_df[agg_df["role"] == "cone"]
    mean_delta_bridge = (
        float(bridge_rows["mean_delta_h1"].mean())
        if len(bridge_rows) > 0
        else float("nan")
    )
    mean_delta_cone = (
        float(cone_rows["mean_delta_h1"].mean()) if len(cone_rows) > 0 else float("nan")
    )

    # "Consistent" = enough heads are non-neutral AND mean stability is high
    bridge_cone_consistent = (
        frac_bridge + frac_cone
    ) > 0.05 and mean_stability >= _CONSISTENCY_THRESHOLD

    # Control comparison
    control_max_abs_delta = float("nan")
    contrast = float("nan")
    if control_agg_df is not None and not control_agg_df.empty:
        control_max_abs_delta = float(control_agg_df["mean_delta_h1"].abs().max())
        trained_max = float(agg_df["mean_delta_h1"].abs().max())
        contrast = trained_max - control_max_abs_delta

    row = {
        "model": model_name,
        "alias": alias,
        "total_heads": total_heads,
        "frac_bridge": round(frac_bridge, 4),
        "frac_cone": round(frac_cone, 4),
        "frac_neutral": round(frac_neutral, 4),
        "mean_stability": round(mean_stability, 4),
        "mean_delta_h1_bridge": round(mean_delta_bridge, 5)
        if not np.isnan(mean_delta_bridge)
        else float("nan"),
        "mean_delta_h1_cone": round(mean_delta_cone, 5)
        if not np.isnan(mean_delta_cone)
        else float("nan"),
        "mean_induction_normal": round(
            induction_stats.get("mean_induction_normal", float("nan")), 4
        ),
        "mean_induction_masked": round(
            induction_stats.get("mean_induction_masked", float("nan")), 4
        ),
        "mean_induction_delta": round(
            induction_stats.get("mean_induction_delta", float("nan")), 4
        ),
        "n_induction_heads": induction_stats.get("n_heads_analysed", 0),
        "bridge_cone_consistent": bridge_cone_consistent,
        "control_max_abs_delta": round(control_max_abs_delta, 5)
        if not np.isnan(control_max_abs_delta)
        else float("nan"),
        "contrast_trained_vs_control": round(contrast, 5)
        if not np.isnan(contrast)
        else float("nan"),
    }
    return row


# ---------------------------------------------------------------------------
# Pretty-print summary table
# ---------------------------------------------------------------------------


def _print_summary_table(summary_df: pd.DataFrame) -> None:
    """Log the summary table in a readable format."""
    log.info("")
    log.info("=" * 90)
    log.info("CROSS-MODEL SWEEP SUMMARY")
    log.info("=" * 90)

    display_cols = [
        "alias",
        "total_heads",
        "frac_bridge",
        "frac_cone",
        "frac_neutral",
        "mean_stability",
        "mean_induction_delta",
        "bridge_cone_consistent",
    ]
    available_cols = [c for c in display_cols if c in summary_df.columns]
    table_str = summary_df[available_cols].to_string(
        index=False, float_format=lambda x: f"{x:.4f}"
    )
    for line in table_str.splitlines():
        log.info("  %s", line)

    log.info("")
    log.info("Column descriptions:")
    log.info(
        "  frac_bridge           — fraction of (layer,head) pairs classified as bridge"
    )
    log.info(
        "  frac_cone             — fraction of (layer,head) pairs classified as cone"
    )
    log.info(
        "  mean_stability        — mean fraction of prompts where role was consistent"
    )
    log.info(
        "  mean_induction_delta  — mean (masked − normal) induction score for top cone heads"
    )
    log.info(
        "  bridge_cone_consistent— True if ≥5%% non-neutral heads with stability ≥ %.2f",
        _CONSISTENCY_THRESHOLD,
    )
    log.info("=" * 90)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    set_seed(seed)

    # --- Resolve which models to run ---
    all_model_cfgs: list[dict] = cfg.get("models", [])
    if not all_model_cfgs:
        log.error("No models defined in config/default.yaml.  Exiting.")
        sys.exit(1)

    if args.models:
        alias_set = set(args.models)
        models_to_run = [m for m in all_model_cfgs if m["alias"] in alias_set]
        unknown = alias_set - {m["alias"] for m in models_to_run}
        if unknown:
            log.warning(
                "Unknown model aliases (will be skipped): %s.  Available: %s",
                sorted(unknown),
                [m["alias"] for m in all_model_cfgs],
            )
    else:
        models_to_run = all_model_cfgs

    if not models_to_run:
        log.error("No valid models to run after filtering.  Exiting.")
        sys.exit(1)

    log.info(
        "Cross-model sweep: %d model(s): %s",
        len(models_to_run),
        [m["alias"] for m in models_to_run],
    )

    # --- Load prompts ---
    all_prompts = load_prompts(cfg)
    if args.max_prompts is not None:
        all_prompts = all_prompts[: args.max_prompts]
        log.info("Prompt cap applied: using %d prompts.", len(all_prompts))
    if not all_prompts:
        log.error("No prompts loaded.  Check data/prompts.json.")
        sys.exit(1)

    tda_cfg: dict = cfg.get("tda", {})
    run_control = not args.no_control

    # --- Per-model sweep ---
    summary_rows: list[dict] = []
    failed_models: list[str] = []

    for model_cfg in models_to_run:
        alias = model_cfg["alias"]
        log.info("")
        log.info("━" * 64)
        log.info("Processing model: %s  (%s)", alias, model_cfg["name"])
        log.info("━" * 64)

        try:
            # ── 1. Atlas ──────────────────────────────────────────────────
            out_dir = results_dir(cfg, alias)
            cached_atlas_path = out_dir / "atlas.csv"

            if args.use_cached and cached_atlas_path.exists():
                log.info(
                    "Using cached atlas: %s  (pass --use-cached=False to recompute)",
                    cached_atlas_path,
                )
                agg_df = pd.read_csv(cached_atlas_path)

                # BUG FIX: always try to load a cached control atlas when use-cached is set.
                # Previously this block was missing, so control_max_abs_delta was always NaN
                # even when the control had been computed in a prior run.
                ctrl_path = out_dir / "atlas_control.csv"
                if ctrl_path.exists():
                    control_agg_df = pd.read_csv(ctrl_path)
                    log.info("Loaded cached control atlas: %s", ctrl_path)
                else:
                    control_agg_df = None
                    log.warning(
                        "No cached control atlas found at %s. "
                        "Re-run without --use-cached or run with --no-control to suppress this warning.",
                        ctrl_path,
                    )
            else:
                agg_df, control_agg_df = _run_atlas_for_model(
                    model_cfg=model_cfg,
                    prompts=all_prompts,
                    tda_cfg=tda_cfg,
                    run_control=run_control,
                    layers=args.layers,
                    heads=args.heads,
                    cfg=cfg,
                )

            if agg_df is None or agg_df.empty:
                log.error("Empty atlas for model '%s' — skipping induction.", alias)
                summary_rows.append(
                    _build_summary_row(model_cfg, pd.DataFrame(), {}, None)
                )
                continue

            log.info(
                "Atlas: %d heads  |  bridge=%d  cone=%d  neutral=%d",
                len(agg_df),
                (agg_df["role"] == "bridge").sum(),
                (agg_df["role"] == "cone").sum(),
                (agg_df["role"] == "neutral").sum(),
            )

            # ── 2. Induction scoring ──────────────────────────────────────
            induction_stats = _run_induction_for_model(
                model_cfg=model_cfg,
                agg_df=agg_df,
                prompts=all_prompts,
                top_n_heads=args.top_n_heads,
                cfg=cfg,
            )
            log.info(
                "Induction delta (masked − normal): %.4f  "
                "(normal=%.4f, masked=%.4f, n_heads=%d)",
                induction_stats.get("mean_induction_delta", float("nan")),
                induction_stats.get("mean_induction_normal", float("nan")),
                induction_stats.get("mean_induction_masked", float("nan")),
                induction_stats.get("n_heads_analysed", 0),
            )

            # ── 3. Build summary row ──────────────────────────────────────
            row = _build_summary_row(
                model_cfg=model_cfg,
                agg_df=agg_df,
                induction_stats=induction_stats,
                control_agg_df=control_agg_df,
            )
            summary_rows.append(row)

        except Exception as exc:
            log.error("Model '%s' failed: %s", alias, exc, exc_info=True)
            failed_models.append(alias)
            # Add a row with NaNs so the table shape is consistent
            summary_rows.append(_build_summary_row(model_cfg, pd.DataFrame(), {}, None))

    # --- Build and save summary table ---
    summary_df = pd.DataFrame(summary_rows)
    summary_path = _REPO_ROOT / "results" / "cross_model_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    log.info("Cross-model summary saved → %s", summary_path)

    # --- Print table ---
    _print_summary_table(summary_df)

    # --- Generate combined visualizations ---
    _generate_cross_model_figures(summary_df, cfg)

    if failed_models:
        log.warning("The following models failed: %s", failed_models)
    else:
        log.info("Cross-model sweep complete — all models finished successfully.")


# ---------------------------------------------------------------------------
# Cross-model figures
# ---------------------------------------------------------------------------


def _generate_cross_model_figures(summary_df: pd.DataFrame, cfg: dict) -> None:
    """
    Generate a bar-chart summary figure comparing all models.

    Plots:
      (a) Fraction of heads: bridge / cone / neutral per model
      (b) Mean induction delta (masked - normal) per model
      (c) Mean stability score per model
    """
    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available — skipping cross-model figures.")
        return

    from src.utils import figures_dir

    fig_dir = figures_dir(cfg)

    if summary_df.empty:
        log.warning("No summary data — skipping cross-model figures.")
        return

    mpl.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "figure.dpi": 150,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
        }
    )

    aliases = summary_df["alias"].tolist()
    x = range(len(aliases))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    # (a) Role fractions
    ax = axes[0]
    bar_w = 0.25
    for i, (role, color) in enumerate(
        [
            ("frac_bridge", "#D62728"),
            ("frac_cone", "#1F77B4"),
            ("frac_neutral", "#AAAAAA"),
        ]
    ):
        vals = summary_df[role].fillna(0).tolist()
        offsets = [xi + (i - 1) * bar_w for xi in x]
        label = role.replace("frac_", "").capitalize()
        ax.bar(offsets, vals, width=bar_w, label=label, color=color, alpha=0.85)

    ax.set_xticks(list(x))
    ax.set_xticklabels(aliases, rotation=20, ha="right")
    ax.set_ylabel("Fraction of Heads")
    ax.set_ylim(0, 1.05)
    ax.set_title("(a) Head Role Distribution", fontweight="bold")
    ax.legend(fontsize=9)

    # (b) Mean induction delta
    ax = axes[1]
    deltas = summary_df["mean_induction_delta"].fillna(0).tolist()
    bar_colors = ["#2CA02C" if d > 0 else "#D62728" for d in deltas]
    ax.bar(list(x), deltas, color=bar_colors, alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(aliases, rotation=20, ha="right")
    ax.set_ylabel("Mean Induction Delta\n(masked − normal)")
    ax.set_title("(b) Induction Score Delta\nfor Top Cone Heads", fontweight="bold")

    # Annotate consistent/inconsistent
    for xi, (delta, consistent) in enumerate(
        zip(deltas, summary_df["bridge_cone_consistent"].tolist())
    ):
        marker = "✓" if consistent else "✗"
        ax.text(
            xi,
            delta + (0.005 if delta >= 0 else -0.015),
            marker,
            ha="center",
            va="bottom",
            fontsize=12,
            color="#2CA02C" if consistent else "#D62728",
        )

    # (c) Mean stability
    ax = axes[2]
    stabilities = summary_df["mean_stability"].fillna(0).tolist()
    ax.bar(
        list(x),
        stabilities,
        color="#9467BD",
        alpha=0.85,
    )
    ax.axhline(
        _CONSISTENCY_THRESHOLD,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=f"Consistency threshold ({_CONSISTENCY_THRESHOLD})",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(aliases, rotation=20, ha="right")
    ax.set_ylabel("Mean Stability Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("(c) Role Consistency\n(stability across prompts)", fontweight="bold")
    ax.legend(fontsize=9)

    fig.suptitle(
        "Cross-Model Generalizability Sweep",
        fontsize=14,
        fontweight="bold",
    )

    fig_path = fig_dir / "cross_model_summary.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Cross-model summary figure saved → %s", fig_path)

    # Also save to results/
    results_fig_path = _REPO_ROOT / "results" / "cross_model_summary.png"
    results_fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig_path_src = fig_dir / "cross_model_summary.png"
    if fig_path_src.exists():
        import shutil

        shutil.copy2(fig_path_src, results_fig_path)
        log.info("Cross-model summary figure also copied → %s", results_fig_path)


if __name__ == "__main__":
    main()
