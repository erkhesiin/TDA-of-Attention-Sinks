"""
experiments/01_topological_atlas.py
------------------------------------
Experiment 1: Topological Atlas

For each model in config/default.yaml:
  1. Load the model and tokenizer.
  2. Run build_topological_atlas over the full prompt suite.
  3. Run build_control_atlas on the same model (randomized weights).
  4. Aggregate per-(layer, head) statistics (mean ΔH1, std, stability).
  5. Save raw and aggregated DataFrames to results/{alias}/.
  6. Generate Figure 1 (heatmap) and Figure 2 (layer profile).

Usage
-----
    # Run all models defined in config/default.yaml
    python experiments/01_topological_atlas.py

    # Run a single model by alias
    python experiments/01_topological_atlas.py --model llama3

    # Restrict to a subset of layers / heads for a quick smoke test
    python experiments/01_topological_atlas.py --model llama3 --layers 0 8 16 24 --heads 0 1 2 3

    # Skip the control atlas (saves time)
    python experiments/01_topological_atlas.py --model llama3 --no-control
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Make sure the repo root is on sys.path so `src` is importable
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.models import load_model
from src.tda_pipeline import (
    aggregate_atlas,
    build_control_atlas,
    build_topological_atlas,
)
from src.utils import get_logger, load_config, load_prompts, results_dir, set_seed
from src.visualize import plot_layer_profile, plot_topological_atlas

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Topological Atlas — Experiment 1",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a custom YAML config. Defaults to config/default.yaml.",
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Run only the model with this alias (e.g. 'llama3'). "
        "When omitted, all models in the config are run.",
    )
    p.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Restrict the scan to specific layer indices (e.g. --layers 0 8 16 24).",
    )
    p.add_argument(
        "--heads",
        type=int,
        nargs="+",
        default=None,
        help="Restrict the scan to specific head indices (e.g. --heads 0 1 2 3).",
    )
    p.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=None,
        help="Only use prompts from these categories for the atlas. "
        "Default: all categories.",
    )
    p.add_argument(
        "--no-control",
        action="store_true",
        default=False,
        help="Skip building the randomized-weights control atlas.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed override. Falls back to config value (default 42).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Per-model pipeline
# ---------------------------------------------------------------------------


def run_model(
    model_cfg: dict,
    prompts: list[dict],
    tda_cfg: dict,
    run_control: bool,
    layers: list[int] | None,
    heads: list[int] | None,
    output_root: Path,
) -> None:
    """Run the full atlas pipeline for a single model."""
    alias: str = model_cfg["alias"]
    model_name: str = model_cfg["name"]

    log.info("=" * 64)
    log.info("Model: %s  (alias=%s)", model_name, alias)
    log.info("=" * 64)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    model, tokenizer = load_model(model_name, quantization="4bit")
    log.info("Model loaded in %.1f s", time.perf_counter() - t0)

    # TDA hyper-parameters from config
    tda_kwargs: dict = {
        "sink_idx": 0,
        "bridge_threshold": float(tda_cfg.get("bridge_threshold", -0.03)),
        "cone_threshold": float(tda_cfg.get("cone_threshold", 0.03)),
        "max_filtration": float(tda_cfg.get("max_filtration", 1.0)),
        "homology_dim": int(tda_cfg.get("homology_dim", 1)),
        "min_persistence": float(tda_cfg.get("min_persistence", 0.05)),
    }
    if layers is not None:
        tda_kwargs["layers"] = layers
    if heads is not None:
        tda_kwargs["heads"] = heads

    # ------------------------------------------------------------------
    # Trained model atlas
    # ------------------------------------------------------------------
    log.info("Building topological atlas (trained model) ...")
    t1 = time.perf_counter()
    atlas_df = build_topological_atlas(
        model, tokenizer, prompts, verbose=True, **tda_kwargs
    )
    log.info(
        "Atlas complete in %.1f s.  Rows: %d", time.perf_counter() - t1, len(atlas_df)
    )

    if atlas_df.empty:
        log.error("Atlas DataFrame is empty — TDA may have failed for all prompts.")
        return

    agg_df = aggregate_atlas(atlas_df)
    log.info(
        "Aggregated: %d (layer, head) pairs.  Bridge: %d, Cone: %d, Neutral: %d",
        len(agg_df),
        (agg_df["role"] == "bridge").sum(),
        (agg_df["role"] == "cone").sum(),
        (agg_df["role"] == "neutral").sum(),
    )

    # ------------------------------------------------------------------
    # Control atlas
    # ------------------------------------------------------------------
    control_df = None
    control_agg_df = None

    if run_control:
        log.info("Building control atlas (randomized weights) ...")
        t2 = time.perf_counter()
        try:
            control_df = build_control_atlas(
                model, tokenizer, prompts, verbose=False, **tda_kwargs
            )
            control_agg_df = aggregate_atlas(control_df)
            log.info(
                "Control atlas complete in %.1f s.  Rows: %d",
                time.perf_counter() - t2,
                len(control_df),
            )
        except Exception as exc:
            log.error("Control atlas failed: %s", exc)
            control_df = None
            control_agg_df = None

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    out_dir = results_dir(model_alias=alias)
    out_dir.mkdir(parents=True, exist_ok=True)

    atlas_path = out_dir / "atlas_raw.csv"
    atlas_df.to_csv(atlas_path, index=False)
    log.info("Raw atlas saved → %s", atlas_path)

    agg_path = out_dir / "atlas.csv"
    agg_df.to_csv(agg_path, index=False)
    log.info("Aggregated atlas saved → %s", agg_path)

    if control_df is not None and control_agg_df is not None:
        ctrl_path = out_dir / "atlas_control_raw.csv"
        control_df.to_csv(ctrl_path, index=False)
        log.info("Control raw atlas saved → %s", ctrl_path)

        ctrl_agg_path = out_dir / "atlas_control.csv"
        control_agg_df.to_csv(ctrl_agg_path, index=False)
        log.info("Control aggregated atlas saved → %s", ctrl_agg_path)

    # ------------------------------------------------------------------
    # Figure 1: Topological Atlas heatmap
    # ------------------------------------------------------------------
    fig1_path = out_dir / "fig1_topological_atlas.png"
    try:
        plot_topological_atlas(
            df=agg_df,
            output_path=str(fig1_path),
            control_df=control_agg_df if control_agg_df is not None else None,
            bridge_threshold=float(tda_kwargs["bridge_threshold"]),
            cone_threshold=float(tda_kwargs["cone_threshold"]),
        )
        log.info("Figure 1 saved → %s", fig1_path)
    except Exception as exc:
        log.error("Figure 1 failed: %s", exc)

    # ------------------------------------------------------------------
    # Figure 2: Bridge/Cone Layer Profile
    # ------------------------------------------------------------------
    fig2_path = out_dir / "fig2_layer_profile.png"
    try:
        plot_layer_profile(df=atlas_df, output_path=str(fig2_path))
        log.info("Figure 2 saved → %s", fig2_path)
    except Exception as exc:
        log.error("Figure 2 failed: %s", exc)

    # ------------------------------------------------------------------
    # Print summary table to stdout
    # ------------------------------------------------------------------
    log.info("\nTop 10 hub heads by |mean ΔH1|:")
    top10 = (
        agg_df.assign(abs_delta=agg_df["mean_delta_h1"].abs())
        .sort_values("abs_delta", ascending=False)
        .head(10)[
            ["layer", "head", "mean_delta_h1", "std_delta_h1", "role", "stability"]
        ]
        .to_string(index=False, float_format=lambda x: f"{x:+.4f}")
    )
    for line in top10.splitlines():
        log.info("  %s", line)

    # Free GPU memory before loading the next model
    del model
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    # Load configuration
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    set_seed(seed)

    tda_cfg: dict = cfg.get("tda", {})

    # Load prompts (optionally filtered by category)
    prompts = load_prompts(cfg, categories=args.categories)
    if not prompts:
        log.error(
            "No prompts loaded (categories=%s).  Check data/prompts.json.",
            args.categories,
        )
        sys.exit(1)
    log.info("Using %d prompts for the atlas sweep.", len(prompts))

    # Select models
    all_models: list[dict] = cfg.get("models", [])
    if not all_models:
        log.error("No models defined in config.  Check config/default.yaml.")
        sys.exit(1)

    if args.model is not None:
        models_to_run = [m for m in all_models if m["alias"] == args.model]
        if not models_to_run:
            known = [m["alias"] for m in all_models]
            log.error(
                "Model alias '%s' not found in config.  Known aliases: %s",
                args.model,
                known,
            )
            sys.exit(1)
    else:
        models_to_run = all_models

    log.info(
        "Running atlas for %d model(s): %s",
        len(models_to_run),
        [m["alias"] for m in models_to_run],
    )

    output_root = _REPO_ROOT / cfg.get("paths", {}).get("results_dir", "results")

    # Run per-model pipeline
    failed: list[str] = []
    for model_cfg in models_to_run:
        try:
            run_model(
                model_cfg=model_cfg,
                prompts=prompts,
                tda_cfg=tda_cfg,
                run_control=not args.no_control,
                layers=args.layers,
                heads=args.heads,
                output_root=output_root,
            )
        except Exception as exc:
            log.error(
                "Model '%s' failed with error: %s",
                model_cfg["alias"],
                exc,
                exc_info=True,
            )
            failed.append(model_cfg["alias"])

    if failed:
        log.warning("The following models failed: %s", failed)
    else:
        log.info("All models completed successfully.")


if __name__ == "__main__":
    main()
