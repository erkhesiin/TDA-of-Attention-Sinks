"""
src/visualize.py
----------------
Publication-quality figures for the TDA-of-Attention-Sinks project.

All five required figures are implemented here with consistent styling:
  Figure 1 — Topological Atlas heatmap (trained vs. randomized control)
  Figure 2 — Bridge/Cone Layer Profile (fraction of heads per layer)
  Figure 3 — Induction Score Comparison (masked vs. normal, by category)
  Figure 4 — Training Curves (CF experiment: task loss + topological drift)
  Figure 5 — Attention Skeleton (qualitative, 3 prompts side-by-side)

Usage
-----
    from src.visualize import (
        plot_topological_atlas,
        plot_layer_profile,
        plot_induction_comparison,
        plot_training_curves,
        plot_skeleton,
    )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from scipy import stats

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

_STYLE_DEFAULTS: dict = {
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
}

_ROLE_PALETTE: dict[str, str] = {
    "bridge": "#D62728",  # red
    "cone": "#1F77B4",  # blue
    "neutral": "#AAAAAA",  # grey
}

_CONDITION_PALETTE: dict[str, str] = {
    "no_reg": "#888888",
    "frobenius": "#FF7F0E",
    "persistence_diag": "#2CA02C",
}

_DPI = 300


def _apply_style() -> None:
    """Apply global rcParams once per plotting call."""
    mpl.rcParams.update(_STYLE_DEFAULTS)


def _save(fig: plt.Figure, output_path: str | Path, tight: bool = True) -> None:
    """Save *fig* to *output_path*, creating parent dirs if needed."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure → %s", path)


# ---------------------------------------------------------------------------
# Figure 1 — Topological Atlas
# ---------------------------------------------------------------------------


def plot_topological_atlas(
    df: pd.DataFrame,
    output_path: str,
    control_df: Optional[pd.DataFrame] = None,
    bridge_threshold: float = -0.03,
    cone_threshold: float = 0.03,
    vmin: float = -0.10,
    vmax: float = 0.10,
) -> None:
    """
    Figure 1: Topological Atlas heatmap.

    Displays mean ΔH1 for every (layer, head) pair as a 2-D colour map.
    When *control_df* is supplied the figure shows three panels side-by-side:
        (a) mean ΔH1 — trained model
        (b) std  ΔH1 — trained model (variance companion)
        (c) mean ΔH1 — randomized control

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``src.tda_pipeline.aggregate_atlas``.
        Required columns: layer, head, mean_delta_h1, std_delta_h1.
    output_path : str
        Where to write the PNG/PDF.
    control_df : pd.DataFrame or None
        Aggregated atlas for the randomized-weights control model.
    bridge_threshold, cone_threshold : float
        Thresholds used to draw role-boundary annotations in the colour bar.
    vmin, vmax : float
        Symmetric colour scale limits.  The midpoint (white) is always 0.
    """
    _apply_style()

    n_layers = int(df["layer"].max()) + 1
    n_heads = int(df["head"].max()) + 1

    def _to_grid(frame: pd.DataFrame, col: str) -> np.ndarray:
        grid = np.full((n_layers, n_heads), np.nan)
        for _, row in frame.iterrows():
            grid[int(row["layer"]), int(row["head"])] = row[col]
        return grid

    mean_grid = _to_grid(df, "mean_delta_h1")
    std_grid = _to_grid(df, "std_delta_h1")

    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    n_panels = 3 if control_df is not None else 2
    fig, axes = plt.subplots(
        1, n_panels, figsize=(6.5 * n_panels, 6), constrained_layout=True
    )
    if n_panels == 1:
        axes = [axes]

    # Panel (a): mean ΔH1
    im_a = axes[0].imshow(
        mean_grid,
        aspect="auto",
        origin="lower",
        cmap="RdBu_r",
        norm=norm,
        interpolation="nearest",
    )
    axes[0].set_title("(a) Mean ΔH1 — Trained Model")
    axes[0].set_xlabel("Head Index")
    axes[0].set_ylabel("Layer Index")
    _tick_every(axes[0], n_heads, n_layers, step=4)
    cb_a = fig.colorbar(im_a, ax=axes[0], fraction=0.046, pad=0.04)
    cb_a.set_label("Mean ΔH1")
    cb_a.ax.axhline(bridge_threshold, color="k", lw=1.2, linestyle="--")
    cb_a.ax.axhline(cone_threshold, color="k", lw=1.2, linestyle="--")

    # Panel (b): std ΔH1
    im_b = axes[1].imshow(
        std_grid,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        vmin=0.0,
        interpolation="nearest",
    )
    axes[1].set_title("(b) Std ΔH1 — Trained Model")
    axes[1].set_xlabel("Head Index")
    axes[1].set_ylabel("Layer Index")
    _tick_every(axes[1], n_heads, n_layers, step=4)
    cb_b = fig.colorbar(im_b, ax=axes[1], fraction=0.046, pad=0.04)
    cb_b.set_label("Std ΔH1 (across prompts)")

    # Panel (c): control
    if control_df is not None:
        ctrl_grid = _to_grid(control_df, "mean_delta_h1")
        im_c = axes[2].imshow(
            ctrl_grid,
            aspect="auto",
            origin="lower",
            cmap="RdBu_r",
            norm=norm,
            interpolation="nearest",
        )
        axes[2].set_title("(c) Mean ΔH1 — Randomized Control")
        axes[2].set_xlabel("Head Index")
        axes[2].set_ylabel("Layer Index")
        _tick_every(axes[2], n_heads, n_layers, step=4)
        cb_c = fig.colorbar(im_c, ax=axes[2], fraction=0.046, pad=0.04)
        cb_c.set_label("Mean ΔH1 (control)")

    fig.suptitle("Figure 1 — Topological Atlas", fontsize=14, fontweight="bold")
    _save(fig, output_path, tight=False)


def _tick_every(ax: plt.Axes, n_cols: int, n_rows: int, step: int = 4) -> None:
    ax.set_xticks(range(0, n_cols, step))
    ax.set_yticks(range(0, n_rows, step))


# ---------------------------------------------------------------------------
# Figure 2 — Bridge/Cone Layer Profile
# ---------------------------------------------------------------------------


def plot_layer_profile(
    df: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Figure 2: Bridge/Cone Layer Profile.

    Line plot showing, for each layer, the fraction of heads classified as
    bridge / cone / neutral.  Shaded region = ± 1 std of that fraction
    across prompts (i.e. the stability of the classification).

    Parameters
    ----------
    df : pd.DataFrame
        Per-prompt atlas rows. Required columns:
        layer, head, delta_h1, role, prompt_id.
        (output of ``src.tda_pipeline.build_topological_atlas`` — *not* the
        aggregated version, so we have per-prompt role labels).
    output_path : str
    """
    _apply_style()

    if df.empty:
        logger.warning("plot_layer_profile: empty DataFrame — skipping.")
        return

    layers = sorted(df["layer"].unique())
    roles = ["bridge", "cone", "neutral"]
    n_heads_per_layer = df.groupby("layer")["head"].nunique()

    # For each layer, compute per-prompt fraction of heads per role
    layer_role_data: dict[str, dict[int, list[float]]] = {r: {} for r in roles}

    for layer in layers:
        layer_df = df[df["layer"] == layer]
        n_heads = n_heads_per_layer[layer]
        for prompt_id, g in layer_df.groupby("prompt_id"):
            for role in roles:
                frac = (g["role"] == role).sum() / max(n_heads, 1)
                layer_role_data[role].setdefault(layer, []).append(frac)

    fig, ax = plt.subplots(figsize=(10, 5))

    for role in roles:
        means, stds = [], []
        for layer in layers:
            vals = layer_role_data[role].get(layer, [0.0])
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals)))
        means = np.array(means)
        stds = np.array(stds)

        color = _ROLE_PALETTE[role]
        ax.plot(layers, means, label=role.capitalize(), color=color, linewidth=2)
        ax.fill_between(
            layers,
            np.clip(means - stds, 0, 1),
            np.clip(means + stds, 0, 1),
            color=color,
            alpha=0.18,
        )

    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Fraction of Heads")
    ax.set_title("Figure 2 — Bridge / Cone Layer Profile", fontweight="bold")
    ax.set_ylim(0.0, 1.05)
    ax.set_xlim(layers[0], layers[-1])
    ax.legend(loc="upper right")
    ax.set_xticks(layers[:: max(1, len(layers) // 16)])

    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Figure 3 — Induction Score Comparison
# ---------------------------------------------------------------------------


def plot_induction_comparison(
    df: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Figure 3: Induction Score Comparison (masked vs. normal).

    Box plot with jittered individual points, grouped by prompt category.
    Annotates the figure with the paired t-test p-value (two-tailed).

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``src.induction_metric.compare_induction_masked_vs_unmasked``.
        Required columns: prompt_id, category, score_normal, score_masked, delta.
    output_path : str
    """
    _apply_style()

    if df.empty:
        logger.warning("plot_induction_comparison: empty DataFrame — skipping.")
        return

    categories = sorted(df["category"].unique())
    n_cats = len(categories)

    fig, axes = plt.subplots(
        1,
        n_cats if n_cats > 1 else 1,
        figsize=(max(6, 3.5 * n_cats), 6),
        sharey=True,
    )
    if n_cats == 1:
        axes = [axes]

    for ax, cat in zip(axes, categories):
        sub = df[df["category"] == cat]
        if sub.empty:
            ax.set_visible(False)
            continue

        normal_vals = sub["score_normal"].values
        masked_vals = sub["score_masked"].values

        # Box plot
        bp = ax.boxplot(
            [normal_vals, masked_vals],
            labels=["Normal\n(sink active)", "Masked\n(sink removed)"],
            patch_artist=True,
            medianprops=dict(color="black", linewidth=2),
            whiskerprops=dict(linewidth=1.5),
            capprops=dict(linewidth=1.5),
        )
        bp["boxes"][0].set_facecolor("#AEC6CF")
        bp["boxes"][1].set_facecolor("#FFB347")

        # Jitter
        rng = np.random.default_rng(seed=42)
        for i, vals in enumerate([normal_vals, masked_vals], start=1):
            jitter = rng.uniform(-0.1, 0.1, size=len(vals))
            ax.scatter(
                np.full(len(vals), i) + jitter,
                vals,
                color="black",
                alpha=0.55,
                s=22,
                zorder=3,
            )

        # Paired t-test annotation
        if len(normal_vals) >= 2:
            t_stat, p_val = stats.ttest_rel(masked_vals, normal_vals)
            p_str = _p_annotation(p_val)
            y_max = max(normal_vals.max(), masked_vals.max())
            ax.annotate(
                f"paired t-test\np={p_val:.3f} {p_str}",
                xy=(1.5, y_max * 1.05),
                ha="center",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="grey", alpha=0.7),
            )

        ax.set_title(f"Category: {cat}", fontweight="bold")
        ax.set_ylabel("Induction Score" if ax is axes[0] else "")

    fig.suptitle(
        "Figure 3 — Induction Score: Normal vs. Sink-Masked",
        fontsize=13,
        fontweight="bold",
    )
    _save(fig, output_path)


def _p_annotation(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


# ---------------------------------------------------------------------------
# Figure 4 — Training Curves (CF Experiment)
# ---------------------------------------------------------------------------


def plot_training_curves(
    results: dict,
    output_path: str,
) -> None:
    """
    Figure 4: Training Curves for the Catastrophic Forgetting experiment.

    Two subplots:
        (a) Task loss — train and val curves for each condition
        (b) Topological drift metric across training steps

    Parameters
    ----------
    results : dict
        Keys are condition labels: e.g. "no_reg", "frobenius", "persistence_diag".
        Each value is a dict with keys:
            "steps"       : list[int]
            "task_train"  : list[float]
            "task_val"    : list[float]   (may be shorter than task_train if
                                           evaluated every N steps)
            "topo_drift"  : list[float]   (aligned with steps)
            "l_topo"      : list[float]   (regularization loss; 0 for no_reg)

        Optionally, for multi-seed runs each value can be a list of the above
        dicts; the function will average and shade ± 1 std automatically.

    output_path : str

    Example *results* structure (single seed)
    ------------------------------------------
        {
            "no_reg": {
                "steps": [0, 20, 40, ...],
                "task_train": [2.3, 2.1, ...],
                "task_val":   [2.4, 2.2, ...],
                "topo_drift": [0.0, 0.05, ...],
                "l_topo":     [0.0, 0.0, ...],
            },
            "persistence_diag": { ... },
            "frobenius": { ... },
        }
    """
    _apply_style()

    if not results:
        logger.warning("plot_training_curves: empty results dict — skipping.")
        return

    fig, (ax_task, ax_drift) = plt.subplots(1, 2, figsize=(13, 5))

    for condition, data in results.items():
        color = _CONDITION_PALETTE.get(condition, None)
        label = _condition_label(condition)

        # Normalise to list-of-dicts for multi-seed support
        if isinstance(data, dict):
            runs = [data]
        else:
            runs = list(data)

        steps = np.array(runs[0]["steps"])

        # --- subplot (a): task loss ---
        for split, linestyle in [("task_train", "-"), ("task_val", "--")]:
            curves = np.array([r[split] for r in runs if split in r])
            if curves.ndim == 1:
                curves = curves[np.newaxis, :]
            if curves.size == 0:
                continue
            min_len = min(curves.shape[1], len(steps))
            curves = curves[:, :min_len]
            s = steps[:min_len]
            mean = curves.mean(axis=0)
            std = curves.std(axis=0) if len(runs) > 1 else np.zeros_like(mean)

            split_label = "train" if split == "task_train" else "val"
            ax_task.plot(
                s,
                mean,
                label=f"{label} ({split_label})",
                color=color,
                linestyle=linestyle,
                linewidth=2,
            )
            if len(runs) > 1:
                ax_task.fill_between(s, mean - std, mean + std, color=color, alpha=0.15)

        # --- subplot (b): topological drift ---
        if any("topo_drift" in r for r in runs):
            curves = np.array([r["topo_drift"] for r in runs if "topo_drift" in r])
            if curves.ndim == 1:
                curves = curves[np.newaxis, :]
            min_len = min(curves.shape[1], len(steps))
            curves = curves[:, :min_len]
            s = steps[:min_len]
            mean = curves.mean(axis=0)
            std = curves.std(axis=0) if len(runs) > 1 else np.zeros_like(mean)
            ax_drift.plot(s, mean, label=label, color=color, linewidth=2)
            if len(runs) > 1:
                ax_drift.fill_between(
                    s, mean - std, mean + std, color=color, alpha=0.15
                )

    ax_task.set_xlabel("Training Step")
    ax_task.set_ylabel("Task Loss (cross-entropy)")
    ax_task.set_title("(a) Task Loss", fontweight="bold")
    ax_task.legend(loc="upper right", fontsize=9)

    ax_drift.set_xlabel("Training Step")
    ax_drift.set_ylabel("Topological Drift")
    ax_drift.set_title("(b) Topological Drift", fontweight="bold")
    ax_drift.legend(loc="upper left", fontsize=9)

    fig.suptitle(
        "Figure 4 — Catastrophic Forgetting Experiment: Training Curves",
        fontsize=13,
        fontweight="bold",
    )
    _save(fig, output_path)


def _condition_label(condition: str) -> str:
    mapping = {
        "no_reg": "No Regularization",
        "frobenius": "Frobenius Reg.",
        "persistence_diag": "PD Reg. (Wasserstein)",
    }
    return mapping.get(condition, condition)


# ---------------------------------------------------------------------------
# Figure 5 — Attention Skeleton
# ---------------------------------------------------------------------------


def plot_skeleton(
    attn: np.ndarray,
    token_labels: list[str],
    title: str,
    output_path: str,
    threshold: float = 0.10,
    max_edges: int = 40,
) -> None:
    """
    Figure 5 (single panel): Attention Skeleton for one (layer, head, condition).

    Draws the strongest attention edges as a directed graph on a circular
    layout, with the sink token (index 0) anchored at the centre.

    Parameters
    ----------
    attn : np.ndarray
        Shape (seq_len, seq_len). Post-softmax attention weights.
    token_labels : list[str]
        Decoded token strings (len = seq_len).
    title : str
        Panel title (e.g. "L19H0 — Normal").
    output_path : str
        Destination file path.
    threshold : float
        Minimum attention weight for an edge to be drawn.
    max_edges : int
        Hard cap on number of edges drawn (top-max_edges by weight).
    """
    _apply_style()

    seq_len = attn.shape[0]
    n_labels = min(len(token_labels), seq_len)
    labels_clean = [
        _clean_token(token_labels[i]) if i < n_labels else str(i)
        for i in range(seq_len)
    ]

    G = nx.DiGraph()
    for i in range(seq_len):
        G.add_node(i, label=f"{i}\n{labels_clean[i]}")

    # Collect all edges above threshold, keep top max_edges by weight
    edge_candidates: list[tuple[float, int, int]] = []
    for src in range(1, seq_len):  # skip source = sink (row 0)
        for tgt in range(seq_len):
            w = float(attn[src, tgt])
            if w >= threshold:
                edge_candidates.append((w, src, tgt))
    edge_candidates.sort(reverse=True)
    edge_candidates = edge_candidates[:max_edges]

    for w, src, tgt in edge_candidates:
        G.add_edge(src, tgt, weight=w)

    fig, ax = plt.subplots(figsize=(9, 9), facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    # Circular layout with sink at centre
    if seq_len > 1:
        rim_nodes = [i for i in range(seq_len) if i != 0]
        angles = np.linspace(0, 2 * np.pi, len(rim_nodes), endpoint=False)
        pos: dict[int, np.ndarray] = {0: np.array([0.0, 0.0])}
        for node, angle in zip(rim_nodes, angles):
            pos[node] = np.array([np.cos(angle), np.sin(angle)])
    else:
        pos = {0: np.array([0.0, 0.0])}

    # Node colours
    node_colors = _skeleton_node_colors(labels_clean, seq_len)

    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=520,
        alpha=0.9,
    )

    # Edge widths scaled by weight
    if G.edges():
        edge_weights = np.array([G[u][v]["weight"] for u, v in G.edges()])
        edge_widths = np.clip(edge_weights * 5, 0.5, 5.0).tolist()
        # colour map: cool = weak, hot = strong
        edge_cmap = plt.cm.plasma
        norm_w = mpl.colors.Normalize(
            vmin=float(edge_weights.min()), vmax=float(edge_weights.max())
        )
        edge_colors = [edge_cmap(norm_w(w)) for w in edge_weights]

        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            edge_color=edge_colors,
            width=edge_widths,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=16,
            connectionstyle="arc3,rad=0.12",
            alpha=0.85,
        )

    label_pos = {k: (v[0], v[1] + 0.09) for k, v in pos.items()}
    nx.draw_networkx_labels(
        G,
        label_pos,
        ax=ax,
        labels=nx.get_node_attributes(G, "label"),
        font_color="white",
        font_size=8,
        font_weight="bold",
    )

    ax.set_title(title, color="white", fontsize=13, pad=14)
    ax.axis("off")
    _save(fig, output_path, tight=False)


def _clean_token(tok: str) -> str:
    """Strip BPE artifacts and whitespace for display."""
    return tok.replace("Ġ", " ").replace("Ċ", "↵").replace("▁", " ").strip()[:10]


def _skeleton_node_colors(labels: list[str], n: int) -> list[str]:
    colors = []
    for i in range(n):
        if i == 0:
            colors.append("#00FF88")  # sink: bright green
        else:
            lw = labels[i].lower()
            if any(w in lw for w in ("fox", "cat", "dog", "bird")):
                colors.append("#FF4444")
            elif any(w in lw for w in ("lazy", "quick", "slow", "fast")):
                colors.append("#FFA500")
            elif any(c.isdigit() for c in lw):
                colors.append("#A0A0FF")
            else:
                colors.append("#BBBBBB")
    return colors


# ---------------------------------------------------------------------------
# Compound figure: 3 skeleton panels side-by-side (used by experiment 02)
# ---------------------------------------------------------------------------


def plot_skeleton_grid(
    panels: list[dict],
    output_path: str,
) -> None:
    """
    Figure 5 (full): 3 × 2 grid of skeleton plots.

    Each row shows one example prompt; each column shows the normal (left) and
    sink-masked (right) skeleton for that prompt.

    Parameters
    ----------
    panels : list[dict]
        Length-3 list (one per example prompt).  Each dict must have:
            "attn_normal"  : np.ndarray  shape (seq_len, seq_len)
            "attn_masked"  : np.ndarray  shape (seq_len, seq_len)
            "token_labels" : list[str]
            "prompt_label" : str          short description for the row title
    output_path : str
    """
    _apply_style()

    n_rows = len(panels)
    fig = plt.figure(figsize=(18, 8 * n_rows), facecolor="#1a1a2e")
    gs = gridspec.GridSpec(n_rows, 2, figure=fig, hspace=0.35, wspace=0.1)

    for row_idx, panel in enumerate(panels):
        for col_idx, (attn_key, side_label) in enumerate(
            [
                ("attn_normal", "Normal (sink active)"),
                ("attn_masked", "Masked (sink removed)"),
            ]
        ):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            ax.set_facecolor("#1a1a2e")

            attn = panel[attn_key]
            token_labels = panel["token_labels"]
            prompt_label = panel.get("prompt_label", f"Prompt {row_idx + 1}")

            seq_len = attn.shape[0]
            labels_clean = [_clean_token(token_labels[i]) for i in range(seq_len)]
            G = nx.DiGraph()
            for i in range(seq_len):
                G.add_node(i, label=f"{i}\n{labels_clean[i]}")

            edge_candidates: list[tuple[float, int, int]] = []
            for src in range(1, seq_len):
                for tgt in range(seq_len):
                    w = float(attn[src, tgt])
                    if w >= 0.10:
                        edge_candidates.append((w, src, tgt))
            edge_candidates.sort(reverse=True)
            edge_candidates = edge_candidates[:40]
            for w, src, tgt in edge_candidates:
                G.add_edge(src, tgt, weight=w)

            rim_nodes = [i for i in range(seq_len) if i != 0]
            angles = np.linspace(0, 2 * np.pi, max(len(rim_nodes), 1), endpoint=False)
            pos: dict[int, np.ndarray] = {0: np.array([0.0, 0.0])}
            for node, angle in zip(rim_nodes, angles):
                pos[node] = np.array([np.cos(angle), np.sin(angle)])

            node_colors = _skeleton_node_colors(labels_clean, seq_len)

            nx.draw_networkx_nodes(
                G,
                pos,
                ax=ax,
                node_color=node_colors,
                node_size=420,
                alpha=0.9,
            )

            if G.edges():
                edge_weights = np.array([G[u][v]["weight"] for u, v in G.edges()])
                edge_widths = np.clip(edge_weights * 5, 0.5, 4.5).tolist()
                edge_cmap = plt.cm.plasma
                norm_w = mpl.colors.Normalize(
                    vmin=float(edge_weights.min()), vmax=float(edge_weights.max())
                )
                edge_colors = [edge_cmap(norm_w(w)) for w in edge_weights]
                nx.draw_networkx_edges(
                    G,
                    pos,
                    ax=ax,
                    edge_color=edge_colors,
                    width=edge_widths,
                    arrows=True,
                    arrowstyle="-|>",
                    arrowsize=14,
                    connectionstyle="arc3,rad=0.12",
                    alpha=0.82,
                )

            lp = {k: (v[0], v[1] + 0.09) for k, v in pos.items()}
            nx.draw_networkx_labels(
                G,
                lp,
                ax=ax,
                labels=nx.get_node_attributes(G, "label"),
                font_color="white",
                font_size=7,
                font_weight="bold",
            )

            col_title = f"{prompt_label} — {side_label}"
            ax.set_title(col_title, color="white", fontsize=11, pad=10)
            ax.axis("off")

    fig.suptitle(
        "Figure 5 — Attention Skeletons (Normal vs. Sink-Masked)",
        color="white",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    _save(fig, output_path, tight=False)
