"""
visualize_skeleton.py
---
Visualizes the 'Attention Skeleton' of L19H0.
Draws the single strongest attention connection for each token.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from transformer_lens import HookedTransformer

# Config
LAYER = 19
HEAD = 0
MASK_SINK = True  # set true for masked, false for unmasked


def plot_skeleton(model, prompt):
    print(f">>> Extracting Attention for L{LAYER}H{HEAD} (Masked={MASK_SINK})...")

    _, cache = model.run_with_cache(prompt, remove_batch_dim=True)
    tokens = model.to_str_tokens(prompt)
    n_tokens = len(tokens)

    attn = cache["pattern", LAYER][HEAD].detach().cpu().numpy()

    if MASK_SINK:
        print(">>> Masking Sink (forcing model to look elsewhere)...")
        attn[:, 0] = 0.0  # zero out attention to sink
        # renormalize rows so probabilities sum to 1
        row_sums = attn.sum(axis=1, keepdims=True)
        attn = attn / (row_sums + 1e-9)

    G = nx.DiGraph()  # directed graph

    for i, t in enumerate(tokens):
        clean_t = t.replace("Ġ", "").strip()
        G.add_node(i, label=f"{i}\n{clean_t}")

    # for every token (from 1), find the strongest attention
    for source in range(1, n_tokens):
        # find the index of the max attention value in this row
        target = np.argmax(attn[source])
        strength = attn[source, target]

        # only draw if the connection is somewhat strong/meaningful
        if strength > 0.1:
            G.add_edge(source, target, weight=strength)

    plt.figure(figsize=(10, 10), facecolor="black")
    ax = plt.gca()
    ax.set_facecolor("black")

    pos = nx.circular_layout(G)
    pos[0] = np.array([0, 0])

    node_colors = []
    for i in range(n_tokens):
        t = tokens[i].lower()
        if i == 0:
            node_colors.append("#00FF00")  # Sink (Green)
        elif "fox" in t:
            node_colors.append("#FF4444")  # Fox (Red)
        elif "lazy" in t:
            node_colors.append("#FFA500")  # Lazy (Orange)
        else:
            node_colors.append("#AAAAAA")

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=600)

    weights = [G[u][v]["weight"] * 3 for u, v in G.edges()]
    nx.draw_networkx_edges(
        G,
        pos,
        edge_color="#888888",
        width=weights,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=15,
        connectionstyle="arc3,rad=0.1",
    )

    label_pos = {k: (v[0], v[1] + 0.08) for k, v in pos.items()}
    nx.draw_networkx_labels(
        G,
        label_pos,
        labels=nx.get_node_attributes(G, "label"),
        font_color="white",
        font_size=9,
        font_weight="bold",
    )

    state = "MASKED (Spiral)" if MASK_SINK else "NORMAL (Star)"
    plt.title(
        f"L{LAYER}H{HEAD} Attention Skeleton: {state}", color="white", fontsize=16
    )
    plt.axis("off")

    filename = f"skeleton_plot_{state}.png"
    plt.savefig(filename, facecolor="black", dpi=300)
    print(f">>> Saved to {filename}")


if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = HookedTransformer.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct", device=device, dtype=torch.float16
    )
    PROMPT = "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog."
    plot_skeleton(model, PROMPT)
