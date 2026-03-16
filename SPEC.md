# SPEC.md — TDA of Attention Sinks (Improved)

This document specifies the redesigned codebase for the TDA-of-Attention-Sinks
project. It is organized into modules that map roughly onto the existing files
(`topo_scanner.py`, `topo_reg.py`, `visualize_skeleton.py`) but with significantly
expanded scope, stricter methodology, and better separation of concerns.

---

## 0. Goals & Non-Goals

### Goals
- Replicate and strengthen the original bridge/cone finding across multiple
  models and diverse prompts.
- Replace the ad hoc Frobenius-norm regularization loss with a true
  topology-aware loss based on persistence diagrams.
- Quantify the induction-suppression claim statistically rather than via a
  single skeleton plot.
- Add a validation pipeline for the Catastrophic Forgetting (CF) experiment
  that includes held-out evaluation.
- Produce publication-quality figures with consistent styling.

### Non-Goals
- Replacing the Vietoris-Rips pipeline with directed TDA (out of scope for
  now; noted as a future direction in a dedicated section).
- Training models from scratch.
- Supporting closed-weight models (OpenAI, Anthropic, etc.).

---

## 1. Repo Structure

```
TDA-of-Attention-Sinks/
├── SPEC.md                    # this file
├── README.md
├── requirements.txt
├── config/
│   └── default.yaml           # all hyperparameters in one place
├── data/
│   └── prompts.json           # curated prompt suite (see §3)
├── src/
│   ├── __init__.py
│   ├── models.py              # model loading + attention extraction
│   ├── tda_pipeline.py        # TDA core (replaces topo_scanner.py)
│   ├── induction_metric.py    # NEW: quantify induction strength
│   ├── topo_reg.py            # improved regularization (replaces topo_reg.py)
│   ├── visualize.py           # all plotting (replaces visualize_skeleton.py)
│   └── utils.py
├── experiments/
│   ├── 01_topological_atlas.py
│   ├── 02_induction_suppression.py
│   ├── 03_cf_experiment.py
│   └── 04_cross_model_sweep.py
├── results/
│   └── (generated at runtime, gitignored except for final figures)
└── static/                    # final figures for blog/paper
```

---

## 2. Configuration (`config/default.yaml`)

All experiments read from a single config file. No magic constants in source.

```yaml
models:
  - name: meta-llama/Llama-3.1-8B-Instruct
    alias: llama3
  - name: mistralai/Mistral-7B-Instruct-v0.2
    alias: mistral
  - name: google/gemma-2-9b-it
    alias: gemma2
  # Add more as hardware allows

tda:
  max_filtration: 1.0
  homology_dim: 1          # H1 cycles
  symmetrize: true         # Vietoris-Rips requires undirected; document this
  min_persistence: 0.05    # filter trivial cycles

induction:
  top_k_edges: 10          # edges to consider per token in skeleton
  offset: 1                # "next-token" offset to measure

regularization:
  method: persistence_diagram   # "frobenius" (old) | "persistence_diagram" (new)
  lambda: 50.0
  hub_selection: top_delta_h1   # heads with |ΔH1| > threshold
  hub_threshold: 0.3

fine_tuning:
  method: lora
  quantization: 4bit
  lora_r: 8
  lora_alpha: 16
  steps: 200               # increased from 60
  eval_every: 20
  train_prompts: 40        # up from the original handful
  val_prompts: 20          # NEW: held-out set

seed: 42
```

---

## 3. Prompt Suite (`data/prompts.json`)

The original experiment used a single repeated pangram. Replace with a curated
suite covering distinct linguistic structures. Minimum 20 prompts for the
atlas; 40 train + 20 val for the CF experiment.

### Required prompt categories

```
repetition/      — same sentence repeated 2×, 3× (tests induction)
syntactic/       — long-range subject-verb agreement, nested clauses
factual/         — factual recall with distractors
narrative/       — multi-sentence stories (tests context coherence)
code/            — short Python or pseudocode snippets
adversarial/     — prompts designed to confuse induction (e.g. A→B, B→A)
```

### Format
```json
{
  "prompts": [
    {
      "id": "rep_001",
      "category": "repetition",
      "text": "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog.",
      "notes": "original pangram, repeated 2x"
    },
    ...
  ]
}
```

---

## 4. `src/models.py` — Model Loading & Attention Extraction

### Purpose
Centralize all HuggingFace model loading and attention hook registration.
The original code had this scattered; make it reusable across experiments.

### Interface

```python
def load_model(model_name: str, quantization: str = "4bit") -> tuple[model, tokenizer]:
    """Load model with optional quantization. Returns (model, tokenizer)."""

def get_attention_matrices(
    model,
    tokenizer,
    prompt: str,
    layers: list[int] | None = None,
    heads: list[int] | None = None,
) -> dict[tuple[int, int], np.ndarray]:
    """
    Run a forward pass and return attention matrices.

    Returns:
        dict keyed by (layer, head) → numpy array of shape (seq_len, seq_len)
        Values are post-softmax attention weights (not logits).
    """

def mask_sink(attn_matrix: np.ndarray, sink_idx: int = 0) -> np.ndarray:
    """
    Return a copy of attn_matrix with row and column `sink_idx` zeroed
    and rows re-normalized so they sum to 1.
    Raises ValueError if sink_idx out of range.
    """
```

### Notes
- Always set `output_attentions=True` in the forward pass.
- Store a separate unmasked and masked copy for each head. Do not mutate in place.
- Use `torch.no_grad()` for all inference.
- Log the actual sink token text (may not always be [BOS]) and its index.

---

## 5. `src/tda_pipeline.py` — TDA Core

This replaces `topo_scanner.py`. The logic is similar but more rigorous.

### Key changes from original
1. **Document the symmetrization explicitly** — add a warning in the docstring
   and a comment in the code that `D_sym = 0.5*(D + D^T)` loses directionality.
2. **Per-prompt aggregation** — run the filtration over the full prompt suite
   and aggregate delta lifetimes by (layer, head), not just one prompt.
3. **Persistence diagram output** — return the full diagram, not just lifetime
   scalars, so the regularization module can use them.

### Interface

```python
def attention_to_distance(attn: np.ndarray) -> np.ndarray:
    """D_ij = 1 - A_ij, then symmetrize: D_sym = 0.5*(D + D^T)."""

def compute_persistence(
    distance_matrix: np.ndarray,
    max_filtration: float = 1.0,
    homology_dim: int = 1,
    min_persistence: float = 0.05,
) -> np.ndarray:
    """
    Run Vietoris-Rips via giotto-tda.
    Returns persistence diagram: array of shape (n_cycles, 2) with [birth, death].
    Filters out cycles with (death - birth) < min_persistence.
    """

def compute_delta_lifetime(
    attn_unmasked: np.ndarray,
    attn_masked: np.ndarray,
    **persistence_kwargs,
) -> float:
    """
    ΔH1 = mean_lifetime(unmasked) - mean_lifetime(masked).
    Negative → sink is a bridge (removing it destroys loops).
    Positive → sink is a cone (removing it creates loops).
    """

def build_topological_atlas(
    model,
    tokenizer,
    prompts: list[str],
    **kwargs,
) -> pd.DataFrame:
    """
    Run TDA over all (layer, head) pairs for every prompt.
    Returns a DataFrame with columns:
        [prompt_id, layer, head, delta_h1, role]
    where role ∈ {"bridge", "cone", "neutral"} based on sign and magnitude.
    """

def build_control_atlas(
    model,
    tokenizer,
    prompts: list[str],
    **kwargs,
) -> pd.DataFrame:
    """
    Same as build_topological_atlas but on a model with randomized weights.
    Use model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
    or manually randomize attention projection weights only.
    Compare result to trained atlas to confirm topology is learned.
    """
```

### Aggregation
When building the atlas over multiple prompts, report:
- Mean ΔH1 per (layer, head) across prompts
- Standard deviation of ΔH1 — **this was missing in the original**
- Number of prompts where role was consistent (bridge/cone stability score)

---

## 6. `src/induction_metric.py` — Quantify Induction Strength (NEW)

The original showed one skeleton plot for one head on one prompt. This module
turns the induction claim into a measurable, aggregable statistic.

### Definition
For a prompt with a repeated subsequence, the **induction score** of a head is
the fraction of attention weight (among top-k edges) that flows from token `t`
to the token that follows the prior occurrence of the same token:

```
induction_score(h, prompt) = Σ_t A_h[t, prev_next(t)] / Σ_t Σ_{top-k} A_h[t, :]
```

where `prev_next(t)` is the index of the token immediately following the most
recent prior occurrence of the same token identity.

### Interface

```python
def find_repetition_pairs(
    token_ids: list[int],
    offset: int = 1,
) -> list[tuple[int, int]]:
    """
    For each token position t that has a prior occurrence p,
    return (t, p + offset) pairs — i.e. t attends to "what came after p".
    """

def induction_score(
    attn: np.ndarray,
    token_ids: list[int],
    top_k: int = 10,
    offset: int = 1,
) -> float:
    """Compute induction score for a single attention head on a single prompt."""

def compare_induction_masked_vs_unmasked(
    model,
    tokenizer,
    prompts: list[str],
    target_layer: int,
    target_head: int,
) -> pd.DataFrame:
    """
    For each prompt, compute induction_score with sink active and masked.
    Return DataFrame with columns: [prompt_id, score_normal, score_masked, delta].
    Report mean and 95% CI over prompts.
    """
```

This replaces the single skeleton visualization as the primary evidence for
the cone/induction-suppression claim.

---

## 7. `src/topo_reg.py` — Improved Regularization

### Problem with the original
The original loss `L_topo = λ Σ_h ||A_h^current - A_h^anchor||_F` penalizes
any change in attention weights for hub heads. This trivially reduces
topological drift but is essentially freezing those heads, not preserving
topology per se.

### New approach: Persistence-Diagram Loss

Use the Wasserstein distance between the persistence diagram of the current
attention matrix and the anchor diagram:

```
L_topo = λ Σ_h W2(PD(A_h^current), PD(A_h^anchor))
```

where W2 is the 2-Wasserstein distance between persistence diagrams.

Use `gudhi` or `persim` for Wasserstein distance computation.

### Interface

```python
def compute_anchor_diagrams(
    model,
    tokenizer,
    prompts: list[str],
    hub_heads: list[tuple[int, int]],
) -> dict[tuple[int, int], np.ndarray]:
    """
    Before fine-tuning: compute and save persistence diagrams for hub heads.
    Returns dict: (layer, head) → persistence diagram array.
    """

def persistence_diagram_loss(
    current_attns: dict[tuple[int, int], torch.Tensor],
    anchor_diagrams: dict[tuple[int, int], np.ndarray],
    lambda_: float = 50.0,
) -> torch.Tensor:
    """
    Compute L_topo as Wasserstein distance sum over hub heads.
    Must be differentiable w.r.t. current_attns.
    Use persim.wasserstein for the distance; detach for the anchor.
    Falls back to Frobenius norm if Wasserstein computation fails
    (with a logged warning).
    """

def topological_drift(
    current_attns: dict[tuple[int, int], torch.Tensor],
    anchor_diagrams: dict[tuple[int, int], np.ndarray],
) -> float:
    """Scalar drift metric for logging during training. Not used for gradients."""
```

### Fine-tuning loop requirements
- Evaluate on held-out validation prompts every `eval_every` steps.
- Log: task loss (train), task loss (val), topological drift, L_topo value.
- Save the best checkpoint by val task loss, not train loss.
- Run with and without regularization and save both curves.

---

## 8. `src/visualize.py` — Plotting

Consolidate all plotting into one module with consistent styling. All figures
should be publication-quality (300 DPI, consistent font sizes, labeled axes).

### Required figures

**Figure 1: Topological Atlas (heatmap)**
- X-axis: head index, Y-axis: layer index
- Color: mean ΔH1 across prompts
- Include error bars or a companion variance heatmap
- Side-by-side: trained model vs. randomized control

**Figure 2: Bridge/Cone Layer Profile**
- X-axis: layer, Y-axis: fraction of heads classified as bridge / cone / neutral
- Line plot, with shaded region = 1 std across prompts
- This is NEW — shows the layer-depth transition quantitatively

**Figure 3: Induction Score Comparison**
- Box plot: induction score (masked) vs. induction score (normal) per prompt
- Grouped by prompt category
- Include a paired t-test p-value in the plot annotation

**Figure 4: Training Curves (CF experiment)**
- Two subplots: (a) task loss train + val, (b) topological drift
- Two lines per subplot: with regularization vs. without
- Shade ± 1 std if running multiple seeds

**Figure 5: Skeleton Plot (qualitative)**
- Keep from original, but show 3 example prompts side-by-side (masked vs. normal)
- Label tokens clearly, color edges by attention weight

### Interface

```python
def plot_topological_atlas(df: pd.DataFrame, output_path: str): ...
def plot_layer_profile(df: pd.DataFrame, output_path: str): ...
def plot_induction_comparison(df: pd.DataFrame, output_path: str): ...
def plot_training_curves(results: dict, output_path: str): ...
def plot_skeleton(attn: np.ndarray, token_labels: list[str], title: str, output_path: str): ...
```

---

## 9. Experiments

### `01_topological_atlas.py`
1. Load each model from config.
2. For each model, run `build_topological_atlas` over all prompts.
3. Run `build_control_atlas` for the same model.
4. Save DataFrames to `results/{model_alias}/atlas.csv`.
5. Generate Figures 1 and 2.

### `02_induction_suppression.py`
1. For each model, identify the top cone heads (highest mean ΔH1 in deep layers).
2. Run `compare_induction_masked_vs_unmasked` for those heads over the repetition
   prompt category.
3. Report mean induction score (normal vs. masked) with 95% CI and p-value.
4. Generate Figure 3 and Figure 5.

### `03_cf_experiment.py`
1. Load a single model (LLaMA 3.1-8B by default).
2. Identify hub heads from the atlas.
3. Compute anchor diagrams.
4. Fine-tune with LoRA:
   - Condition A: no regularization
   - Condition B: persistence-diagram regularization
   - Condition C: Frobenius regularization (old method, for comparison)
5. Evaluate val task loss every `eval_every` steps.
6. Generate Figure 4.

### `04_cross_model_sweep.py`
1. Run experiments 01 and 02 for all models in config.
2. Produce a summary table: for each model, report fraction of heads classified
   as bridge/cone, mean induction delta, and whether the pattern is consistent.
3. This is the key generalizability check that was missing from the original.

---

## 10. Limitations to Document

The rebuilt code should include a `LIMITATIONS.md` that explicitly states:

1. **Undirected TDA**: Vietoris-Rips symmetrizes the attention matrix, losing
   causal directionality. Loops involving the sink represent structural closure,
   not causal flow. Future work should explore path homology or magnitude
   homology for directed graphs.

2. **4-bit quantization**: Fine-tuning in 4-bit may affect attention
   distributions in ways that confound topological measurements. Results should
   be interpreted with this caveat.

3. **Single forward pass**: Each TDA measurement is taken from one forward pass
   per prompt. Attention matrices can vary with sampling; use greedy decoding
   for reproducibility.

4. **Hub head selection**: Hub heads are identified post-hoc from the same data
   used to evaluate regularization. A proper pipeline would identify hubs on a
   held-out atlas set.

5. **Scope of CF claim**: The fine-tuning task (pattern A → pattern B induction
   breaking) is artificial. The CF framing is a hypothesis-generating result,
   not a general demonstration of topological collapse.

---

## 11. Dependencies (`requirements.txt`)

```
torch>=2.2.0
transformers>=4.40.0
peft>=0.10.0
bitsandbytes>=0.43.0
giotto-tda>=0.6.0
gudhi>=3.9.0
persim>=0.3.1
numpy>=1.26.0
pandas>=2.2.0
scipy>=1.13.0
matplotlib>=3.8.0
seaborn>=0.13.0
pyyaml>=6.0
tqdm>=4.66.0
```

---

## 12. What Was NOT Changed

- The core TDA framing (persistent homology on attention matrices) is sound and
  is preserved.
- The bridge/cone taxonomy is a useful organizing concept and is kept.
- The randomized-weights control is kept and expanded to all models.
- The LoRA fine-tuning setup is kept; only the loss function and evaluation
  pipeline are improved.
