"""
experiments/03_cf_experiment.py
--------------------------------
Experiment 03: Catastrophic Forgetting (CF) Experiment.

Pipeline
--------
1. Load a single model (LLaMA 3.1-8B by default, configurable via --model).
2. Run build_topological_atlas on the training prompt set to identify hub heads.
3. Compute anchor diagrams (persistence diagrams before fine-tuning begins).
4. Fine-tune under three conditions in sequence:
     Condition A — no regularization
     Condition B — persistence-diagram Wasserstein regularization (new)
     Condition C — Frobenius-norm regularization (original method, ablation)
5. For each condition, evaluate on a held-out validation set every eval_every steps.
6. Save training history CSVs and best checkpoints per condition.
7. Generate Figure 4 (training curves: task loss + topological drift).

Usage
-----
    # Run all three conditions with default config
    python experiments/03_cf_experiment.py

    # Run a single condition
    python experiments/03_cf_experiment.py --conditions no_reg pd_reg

    # Override model
    python experiments/03_cf_experiment.py --model llama3

    # Skip atlas step and provide hub heads directly
    python experiments/03_cf_experiment.py --hub-heads 19,0 8,3

    # Quick smoke test (few steps, small prompt set)
    python experiments/03_cf_experiment.py --steps 10 --eval-every 5
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so ``src`` is importable
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.utils import get_logger, load_config, load_prompts, results_dir, set_seed

log = get_logger(__name__, level=logging.INFO)

_ALL_CONDITIONS = ("no_reg", "pd_reg", "frob_reg")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 03 — Catastrophic Forgetting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model alias from config (e.g. 'llama3').  Defaults to first model.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML.  Defaults to config/default.yaml.",
    )
    parser.add_argument(
        "--conditions",
        type=str,
        nargs="+",
        choices=list(_ALL_CONDITIONS),
        default=list(_ALL_CONDITIONS),
        help="Which regularization conditions to run.",
    )
    parser.add_argument(
        "--hub-heads",
        type=str,
        nargs="+",
        default=None,
        dest="hub_heads",
        help=(
            "Explicit hub heads in 'layer,head' format (e.g. --hub-heads 19,0 8,3). "
            "When provided the atlas step is skipped."
        ),
    )
    parser.add_argument(
        "--atlas",
        type=str,
        default=None,
        help=(
            "Path to a pre-computed aggregated atlas CSV (experiment 01 output). "
            "When provided the atlas step is skipped."
        ),
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override fine_tuning.steps from config.",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=None,
        dest="eval_every",
        help="Override fine_tuning.eval_every from config.",
    )
    parser.add_argument(
        "--train-categories",
        type=str,
        nargs="+",
        default=None,
        dest="train_categories",
        help=(
            "Prompt categories for training (default: all). "
            "The original experiment used repetition/adversarial prompts."
        ),
    )
    parser.add_argument(
        "--val-categories",
        type=str,
        nargs="+",
        default=None,
        dest="val_categories",
        help="Prompt categories for validation (default: all).",
    )
    parser.add_argument(
        "--skip-atlas",
        action="store_true",
        dest="skip_atlas",
        default=False,
        help=(
            "Skip the atlas step entirely and use a hard-coded fallback set of "
            "hub heads.  Only useful for quick debugging."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Hub-head selection helpers
# ---------------------------------------------------------------------------


def _parse_hub_heads_arg(raw: list[str]) -> list[tuple[int, int]]:
    """Parse CLI '--hub-heads 19,0 8,3' into [(19, 0), (8, 3)]."""
    result: list[tuple[int, int]] = []
    for item in raw:
        try:
            l_str, h_str = item.split(",")
            result.append((int(l_str.strip()), int(h_str.strip())))
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"Could not parse hub head spec '{item}'.  "
                "Expected format: 'layer,head' (e.g. '19,0')."
            )
    return result


def _select_hub_heads_from_atlas(
    agg_df: pd.DataFrame,
    hub_threshold: float = 0.3,
) -> list[tuple[int, int]]:
    """
    Return (layer, head) pairs where |mean_delta_h1| >= hub_threshold.
    Sorted by magnitude descending.
    """
    from src.tda_pipeline import get_hub_heads  # noqa: PLC0415

    heads = get_hub_heads(agg_df, hub_threshold=hub_threshold)
    log.info("Hub heads selected (|ΔH1| >= %.2f): %s", hub_threshold, heads[:10])
    return heads


# ---------------------------------------------------------------------------
# Atlas step
# ---------------------------------------------------------------------------


def _build_or_load_atlas(
    model,
    tokenizer,
    cfg: dict,
    prompts: list[dict],
    out_dir: Path,
    atlas_csv: str | None,
) -> pd.DataFrame:
    """
    Either load an existing aggregated atlas CSV or run experiment 01's
    atlas builder on the training prompts.

    Returns the *aggregated* atlas DataFrame.
    """
    if atlas_csv is not None:
        path = Path(atlas_csv)
        if not path.exists():
            raise FileNotFoundError(f"Atlas CSV not found: {path}")
        agg_df = pd.read_csv(path)
        log.info("Loaded pre-computed atlas from %s  (%d rows)", path, len(agg_df))
        return agg_df

    # Check if a previous run already produced results
    cached_path = out_dir / "atlas_for_cf.csv"
    if cached_path.exists():
        agg_df = pd.read_csv(cached_path)
        log.info(
            "Found cached atlas at %s  (%d rows) — skipping re-computation.",
            cached_path,
            len(agg_df),
        )
        return agg_df

    log.info("No pre-computed atlas found — running atlas scan on training prompts ...")

    from src.tda_pipeline import aggregate_atlas, build_topological_atlas  # noqa

    tda_cfg = cfg.get("tda", {})
    atlas_raw = build_topological_atlas(
        model,
        tokenizer,
        prompts,
        sink_idx=0,
        bridge_threshold=float(tda_cfg.get("bridge_threshold", -0.03)),
        cone_threshold=float(tda_cfg.get("cone_threshold", 0.03)),
        max_filtration=float(tda_cfg.get("max_filtration", 1.0)),
        homology_dim=int(tda_cfg.get("homology_dim", 1)),
        min_persistence=float(tda_cfg.get("min_persistence", 0.05)),
        verbose=True,
    )
    agg_df = aggregate_atlas(atlas_raw)

    agg_df.to_csv(cached_path, index=False)
    log.info("Atlas saved → %s", cached_path)

    return agg_df


# ---------------------------------------------------------------------------
# Results → visualize.plot_training_curves format converter
# ---------------------------------------------------------------------------


def _build_results_dict(
    histories: dict[str, pd.DataFrame],
) -> dict[str, dict]:
    """
    Convert a dict of {condition: history_df} into the format expected by
    src.visualize.plot_training_curves.

    Training and validation rows are stored in the same DataFrame; we split
    them here.
    """
    results: dict[str, dict] = {}

    for condition, df in histories.items():
        train_df = df[df["split"] == "train"].sort_values("step")
        val_df = df[df["split"] == "val"].sort_values("step")

        results[condition] = {
            "steps": train_df["step"].tolist(),
            "task_train": train_df["task_loss"].tolist(),
            "task_val": _interpolate_val(
                val_df["step"].tolist(),
                val_df["task_loss"].tolist(),
                train_df["step"].tolist(),
            ),
            "topo_drift": train_df["topo_drift"].tolist(),
            "l_topo": train_df["reg_loss"].tolist(),
        }

    return results


def _interpolate_val(
    val_steps: list[int],
    val_losses: list[float],
    train_steps: list[int],
) -> list[float]:
    """
    Linearly interpolate validation losses onto the training step grid so that
    both curves share the same x-axis in the plot.

    When val_steps is empty or shorter than train_steps, missing values are
    filled with NaN (matplotlib skips NaN points).
    """
    if not val_steps:
        return [float("nan")] * len(train_steps)

    interp = np.interp(
        train_steps,
        val_steps,
        val_losses,
        left=float("nan"),
        right=float("nan"),
    )
    return interp.tolist()


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def _print_summary(histories: dict[str, pd.DataFrame]) -> None:
    """Log a compact comparison table of final metrics per condition."""
    rows = []
    for condition, df in histories.items():
        val_df = df[df["split"] == "val"].sort_values("step")
        train_df = df[df["split"] == "train"].sort_values("step")

        final_train_loss = (
            float(train_df["task_loss"].iloc[-1]) if len(train_df) > 0 else float("nan")
        )
        best_val_loss = (
            float(val_df["task_loss"].min()) if len(val_df) > 0 else float("nan")
        )
        final_drift = (
            float(train_df["topo_drift"].iloc[-1])
            if len(train_df) > 0
            else float("nan")
        )
        rows.append(
            {
                "condition": condition,
                "final_train_loss": final_train_loss,
                "best_val_loss": best_val_loss,
                "final_topo_drift": final_drift,
            }
        )

    summary = pd.DataFrame(rows)
    log.info("\nSummary across conditions:\n%s", summary.to_string(index=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    # --- Resolve model ---
    models_cfg: list[dict] = cfg.get("models", [])
    if not models_cfg:
        log.error("No models defined in config.")
        sys.exit(1)

    if args.model:
        model_entry = next((m for m in models_cfg if m["alias"] == args.model), None)
        if model_entry is None:
            log.error(
                "Model alias '%s' not found.  Known: %s",
                args.model,
                [m["alias"] for m in models_cfg],
            )
            sys.exit(1)
    else:
        model_entry = models_cfg[0]

    model_name: str = model_entry["name"]
    model_alias: str = model_entry["alias"]

    log.info("=" * 64)
    log.info("Experiment 03 — Catastrophic Forgetting")
    log.info("Model   : %s  (%s)", model_name, model_alias)
    log.info("Cond.   : %s", args.conditions)
    log.info("=" * 64)

    # --- Config values (with CLI overrides) ---
    ft_cfg = cfg.get("fine_tuning", {})
    reg_cfg = cfg.get("regularization", {})

    n_steps: int = args.steps if args.steps is not None else ft_cfg.get("steps", 200)
    eval_every: int = (
        args.eval_every if args.eval_every is not None else ft_cfg.get("eval_every", 20)
    )
    n_train: int = ft_cfg.get("train_prompts", 40)
    n_val: int = ft_cfg.get("val_prompts", 20)
    hub_threshold: float = float(reg_cfg.get("hub_threshold", 0.3))

    # Inject CLI overrides back into cfg so finetune() picks them up
    cfg["fine_tuning"]["steps"] = n_steps
    cfg["fine_tuning"]["eval_every"] = eval_every

    # --- Load prompts ---
    all_prompts = load_prompts(cfg)

    train_prompts = [
        p
        for p in all_prompts
        if args.train_categories is None or p["category"] in args.train_categories
    ][:n_train]

    val_prompts = [
        p
        for p in all_prompts
        if args.val_categories is None or p["category"] in args.val_categories
    ][:n_val]

    if not train_prompts:
        log.error(
            "No training prompts available (categories=%s).", args.train_categories
        )
        sys.exit(1)
    if not val_prompts:
        log.error(
            "No validation prompts available (categories=%s).", args.val_categories
        )
        sys.exit(1)

    log.info(
        "Train prompts: %d | Val prompts: %d", len(train_prompts), len(val_prompts)
    )

    out_dir = results_dir(cfg, model_alias)

    # --- Load model ---
    from src.models import load_model

    model, tokenizer = load_model(model_name, quantization="4bit")

    # --- Resolve hub heads ---
    if args.hub_heads:
        hub_heads = _parse_hub_heads_arg(args.hub_heads)
        log.info("Using explicitly provided hub heads: %s", hub_heads)

    elif args.skip_atlas:
        # Hard-coded fallback for quick debugging (original paper's heads)
        hub_heads = [(8, 3), (19, 0)]
        log.warning("Skipping atlas step.  Using fallback hub heads: %s", hub_heads)

    else:
        agg_df = _build_or_load_atlas(
            model,
            tokenizer,
            cfg=cfg,
            prompts=train_prompts,
            out_dir=out_dir,
            atlas_csv=args.atlas,
        )
        hub_heads = _select_hub_heads_from_atlas(agg_df, hub_threshold=hub_threshold)
        if not hub_heads:
            log.warning(
                "No hub heads found above threshold %.2f.  "
                "Lowering threshold to 0.05 and retrying.",
                hub_threshold,
            )
            hub_heads = _select_hub_heads_from_atlas(agg_df, hub_threshold=0.05)
        if not hub_heads:
            log.error(
                "Still no hub heads found.  Cannot run regularization conditions."
                "  Re-run experiment 01 or provide --hub-heads explicitly."
            )
            sys.exit(1)

    log.info("Hub heads for regularization: %s", hub_heads)

    # Save hub heads list for reference
    hub_df = pd.DataFrame(hub_heads, columns=["layer", "head"])
    hub_df.to_csv(out_dir / "hub_heads.csv", index=False)

    # --- Free model memory before loading again inside finetune() ---
    # finetune() calls load_model internally with quantization + LoRA, so we
    # delete the model we used for the atlas to avoid double-loading on GPU.
    del model
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass

    # --- Run fine-tuning for each condition ---
    from src.topo_reg import finetune

    histories: dict[str, pd.DataFrame] = {}

    for condition in args.conditions:
        log.info("=" * 64)
        log.info("Starting condition: %s", condition)
        log.info("=" * 64)

        try:
            result = finetune(
                model_name=model_name,
                train_prompts=train_prompts,
                val_prompts=val_prompts,
                hub_heads=hub_heads,
                condition=condition,
                cfg=cfg,
                output_dir=str(out_dir),
            )
            histories[condition] = result["history"]

            log.info(
                "Condition '%s' done.  best_val_loss=%.4f at step=%d",
                condition,
                result["best_val_loss"],
                result["best_step"],
            )

        except Exception as exc:
            log.error(
                "Condition '%s' failed: %s",
                condition,
                exc,
                exc_info=True,
            )
            # Insert an empty DataFrame so downstream code doesn't break
            histories[condition] = pd.DataFrame(
                columns=[
                    "step",
                    "split",
                    "task_loss",
                    "reg_loss",
                    "topo_drift",
                    "condition",
                ]
            )

    # --- Summary table ---
    if histories:
        _print_summary(histories)

    # --- Figure 4: Training Curves ---
    if histories:
        log.info("Generating Figure 4 — Training Curves ...")
        from src.visualize import plot_training_curves

        # Map internal condition names to visualize.py convention
        _COND_REMAP = {
            "no_reg": "no_reg",
            "pd_reg": "persistence_diag",
            "frob_reg": "frobenius",
        }

        results_for_plot = _build_results_dict(
            {_COND_REMAP.get(k, k): v for k, v in histories.items()}
        )

        fig4_path = out_dir / "fig4_training_curves.png"
        try:
            plot_training_curves(results_for_plot, str(fig4_path))
            log.info("Figure 4 saved → %s", fig4_path)
        except Exception as exc:
            log.error("Figure 4 generation failed: %s", exc)

    # --- Combine and save all histories ---
    if histories:
        combined = pd.concat(
            [
                df.assign(condition=cond)
                for cond, df in histories.items()
                if not df.empty
            ],
            ignore_index=True,
        )
        combined_path = out_dir / "cf_histories_combined.csv"
        combined.to_csv(combined_path, index=False)
        log.info("Combined training histories saved → %s", combined_path)

    log.info("=" * 64)
    log.info("Experiment 03 complete.  Results in: %s", out_dir)
    log.info("=" * 64)


if __name__ == "__main__":
    main()
