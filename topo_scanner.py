"""
topo_scanner.py
---
Generates the Atlas.
Calculates the "Delta Lifetime" (Masked - Normal) of H1 loops for all attention heads.

NOTE: Uses undirected Vietoris-Rips. Symmetrization (A->B becomes A-B) implies that "loops" involving the sink
represent "semantic cycles" rather than causal flow cycles.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer
from gtda.homology import VietorisRipsPersistence
import os

# Config
MODEL_ID = "Qwen/Qwen3-Next-80B-A3B-Instruct"
PROMPT = "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog."
RANDOMIZE_WEIGHTS = False  # set true to generate control atlas
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_model(randomize=False):
    print(f">>> Loading Model (Randomize={randomize})...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = HookedTransformer.from_pretrained(
        MODEL_ID,
        device=device,
        dtype=torch.float16,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
    )

    if randomize:
        print(">>> Scrambling Attn Weights (Control)...")
        with torch.no_grad():
            for name, param in model.named_parameters():
                if "attn" in name and ("W_" in name):
                    std = param.std()
                    param.copy_(torch.randn_like(param) * std)
    return model


def compute_h1_delta(attention_pattern):
    """
    Computes (Max H1 Lifetime Masked) - (Max H1 Lifetime Normal).
    Uses Undirected Vietoris-Rips.
    """
    # preprocess distance matrix (1 - attention)
    dist = 1.0 - attention_pattern.detach().cpu().numpy()
    np.fill_diagonal(dist, 0)

    # symmetrizing for metric space
    dist_sym = 0.5 * (dist + dist.T)

    # create masked version (sink node 0 disconnected)
    dist_masked = dist_sym.copy()
    dist_masked[0, :] = 1.0
    dist_masked[:, 0] = 1.0

    # compute persistence
    vr = VietorisRipsPersistence(metric="precomputed", homology_dimensions=[1])

    # batch both matrices for speed
    batch = np.stack([dist_sym, dist_masked])
    diagrams = vr.fit_transform(batch)

    lifetimes = []
    for diagram in diagrams:
        h1_points = diagram[diagram[:, 2] == 1.0]  # dim 1
        if len(h1_points) == 0:
            lifetimes.append(0.0)
        else:
            # lifetime = death - birth
            lifetimes.append(np.max(h1_points[:, 1] - h1_points[:, 0]))

    return lifetimes[1] - lifetimes[0]  # masked - normal


# Global storage
atlas_data = np.zeros((32, 32))


def scan_hook(pattern, hook):
    layer = hook.layer()
    print(f"Scanning Layer {layer}...")
    heads = pattern[0]
    for h in range(heads.shape[0]):
        atlas_data[layer, h] = compute_h1_delta(heads[h])
    return pattern


if __name__ == "__main__":
    model = load_model(randomize=RANDOMIZE_WEIGHTS)

    pattern_hook_names = [
        f"blocks.{i}.attn.hook_pattern" for i in range(model.cfg.n_layers)
    ]

    with torch.no_grad():
        model.run_with_hooks(
            PROMPT, fwd_hooks=[(name, scan_hook) for name in pattern_hook_names]
        )

    print(">>> Plotting Atlas...")
    title = (
        "CONTROL ATLAS (Randomized)"
        if RANDOMIZE_WEIGHTS
        else "TOPOLOGICAL ATLAS (Learned)"
    )

    plt.figure(figsize=(10, 8))
    # RdBu map: red = bridge (-), blue = cone (+)
    plt.imshow(
        atlas_data,
        cmap="RdBu",
        interpolation="nearest",
        vmin=-0.1,
        vmax=0.1,
        origin="lower",
    )

    plt.colorbar(label="Delta Lifetime (Masked - Normal)")
    plt.title(title, fontsize=16)
    plt.xlabel("Head Index", fontsize=12)
    plt.ylabel("Layer Index", fontsize=12)

    plt.xticks(range(0, 32, 4))
    plt.yticks(range(0, 32, 4))

    plt.tight_layout()
    filename = "control_atlas.png" if RANDOMIZE_WEIGHTS else "topological_atlas.png"
    plt.savefig(filename, dpi=300)
    print(f">>> Saved to {filename}")
