# LIMITATIONS.md — TDA of Attention Sinks

This document explicitly catalogues the known methodological limitations of
this project. These are not bugs to be fixed but **design constraints and
epistemic caveats** that must be stated clearly when interpreting results or
citing this work.

---

## 1. Undirected TDA (Symmetrization Loses Causal Directionality)

**What we do.**
`src/tda_pipeline.attention_to_distance` converts an attention matrix
`A` into a distance matrix via `D_ij = 1 − A_ij`, then symmetrizes it:

```
``D_sym = 0.5 × (D + D^T)``
```

This is required because Vietoris-Rips filtration is defined on an
*undirected* metric space.

**What we lose.**
Transformer attention is inherently directed: token `i` attending to token
`j` is not the same as token `j` attending to token `i`. The symmetrization
merges these two directional edges into a single undirected edge weighted by
their average. As a consequence:

- A **loop (H1 cycle)** detected in the resulting simplicial complex reflects
  *structural closure* among a group of tokens — not a directed causal
  flow cycle.
- The bridge/cone classification tells us whether the sink *participates in*
  undirected semantic clusters, not whether it is a causal bottleneck in the
  information-routing sense.

**Interpretation guidance.**
Results should be stated as: *"the sink token occupies a structurally
significant position in the undirected attention graph"*, not as *"information
must pass through the sink."*

**Future work.**
Path homology (Grigor'yan et al.) and magnitude homology are candidate
frameworks for directed TDA. These are out of scope for this release and are
documented as a future direction.

---

## 2. 4-Bit Quantization May Distort Attention Distributions

**What we do.**
All models are loaded in 4-bit NF4 quantization via `bitsandbytes` for
memory efficiency on consumer hardware.

**What is affected.**
Post-softmax attention weights are computed from quantized query/key
projections. The quantization error in these projections can shift attention
distributions away from the full-precision baseline. Specifically:

- Sharply peaked attention patterns (e.g., strong sink attraction) may become
  slightly more diffuse.
- Rare long-range attention edges may be amplified or suppressed relative to
  the full-precision model.
- The `ΔH1` values computed in the TDA pipeline and the induction scores in
  `src/induction_metric.py` are therefore approximations of the true
  full-precision values.

**Magnitude of the effect.**
We have not quantified the quantization-induced shift in `ΔH1` or induction
score distributions. Running the same experiments in `bfloat16` or `float32`
(where hardware allows) is recommended before drawing strong quantitative
conclusions.

**Interpretation guidance.**
Treat all numeric thresholds (`bridge_threshold`, `cone_threshold`,
`hub_threshold`) as approximate. Qualitative patterns (which heads are bridges
vs. cones) are likely more robust than precise magnitude comparisons across
precision levels.

---

## 3. Single Forward Pass per Prompt

**What we do.**
Each TDA measurement and induction score is computed from a **single**
deterministic forward pass per prompt, using greedy (argmax) decoding / no
sampling. `torch.no_grad()` is used throughout.

**What is affected.**
Transformer attention weights are *not* stochastic at inference time when
`temperature=0` (greedy decoding). However:

- **Prompt sensitivity**: attention patterns can vary substantially with small
  changes to the input. A single prompt does not fully characterize a head's
  behaviour. We address this partially through the multi-prompt atlas, but
  the number of prompts (≤60) is small relative to the input distribution.
- **Context length effects**: very short prompts may not activate induction
  behaviour reliably. Very long prompts may compress early-layer attention due
  to positional effects.
- **Batching artefacts**: if running multiple prompts in a single batch
  (padded), padding tokens can affect attention patterns. All experiments in
  this codebase run single-sequence (batch size 1) to avoid this.

**Interpretation guidance.**
The multi-prompt atlas aggregation (mean ΔH1 ± std) and the stability score
partially account for prompt sensitivity. However, the standard deviations
reported should not be interpreted as confidence intervals over the full input
distribution — they reflect variance only within the curated prompt suite.

---

## 4. Post-Hoc Hub Head Selection (Circularity Risk)

**What we do.**
Hub heads are identified from the topological atlas built on the same prompt
data that is subsequently used to:
  (a) evaluate regularization effectiveness in the CF experiment, and
  (b) measure induction score changes.

**The problem.**
Selecting hub heads on the same data used to evaluate them inflates the
apparent effect size. A head that happens to exhibit high `|ΔH1|` on the
training/evaluation prompts will be selected and then confirmed on those
same prompts — this is a form of double-dipping.

**A proper pipeline would:**
1. Identify hub heads on a **held-out atlas set** (separate prompts not used
   for regularization training or induction scoring).
2. Only then evaluate induction suppression and CF regularization on the
   remaining prompts.

**Current status.**
This split is not yet implemented. The current codebase uses all available
prompts for both atlas construction and downstream evaluation. Results should
therefore be interpreted as **exploratory / hypothesis-generating**, not as
confirmatory.

**Mitigation in place.**
The prompt suite covers six structurally distinct categories
(`repetition`, `syntactic`, `factual`, `narrative`, `code`, `adversarial`).
Bridge/cone heads that are consistent *across categories* are less likely to
be artefacts of category-specific selection.

---

## 5. Scope of the Catastrophic Forgetting Claim

**What we do.**
The CF experiment fine-tunes a model on adversarial prompts designed to break
repetition-based induction (e.g. A→B, B→A swaps), then measures whether the
topological structure of hub heads changes.

**What this demonstrates.**
A narrow, artificial demonstration: *training on induction-breaking patterns
causes measurable topological drift in heads identified as induction-related.*
The persistence-diagram regularization reduces this drift relative to
unregularized training.

**What this does NOT demonstrate.**

1. **Generality**: the fine-tuning task is synthetic and small-scale (≤200
   steps, ≤40 prompts). This is not a general demonstration of topological
   collapse during real-world RLHF or instruction fine-tuning.

2. **Causal attribution**: we observe a correlation between topological drift
   and degraded induction scores. We do not demonstrate that topological drift
   *causes* capability loss; both may be downstream of the weight changes.

3. **Downstream task performance**: no held-out NLP benchmarks (MMLU, HellaSwag,
   etc.) are evaluated. The validation set is drawn from the same prompt
   distribution as training, limiting generalizability claims.

4. **Optimality of the regularization**: the persistence-diagram loss is better
   motivated than the Frobenius baseline, but we have not compared it against
   other topology-preserving fine-tuning methods (e.g. EWC, MAS, or
   activation-space anchoring).

**Interpretation guidance.**
Frame the CF experiment as: *"a proof-of-concept showing that
persistence-diagram regularization reduces measurable topological drift during
fine-tuning on a specific synthetic task"*, not as *"a general solution to
catastrophic forgetting in large language models."*

---

## Summary Table

| # | Limitation | Severity | Mitigation in Codebase |
|---|-----------|----------|------------------------|
| 1 | Undirected TDA loses directionality | Medium | Documented in module docstrings; future work noted |
| 2 | 4-bit quantization distorts attention | Low–Medium | Flagged in `src/models.py`; config supports `None` quantization |
| 3 | Single forward pass per prompt | Low | Multi-prompt atlas + stability score partially addresses this |
| 4 | Post-hoc hub head selection | Medium–High | Flagged as exploratory; category-diverse prompt suite as partial mitigation |
| 5 | Narrow CF scope | Medium | Explicit framing in experiment scripts and README |

---

*Last updated: see git log. Please update this file when new experiments are added or methodological changes are made.*