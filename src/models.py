"""
src/models.py
-------------
Centralized HuggingFace model loading and attention extraction.

Replaces the scattered model-loading code in the original topo_scanner.py
and topo_reg.py. All experiments should import from here.
"""

import logging
from typing import Optional

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(
    model_name: str,
    quantization: Optional[str] = "4bit",
) -> tuple:
    """
    Load a HuggingFace causal-LM and its tokenizer.

    Parameters
    ----------
    model_name : str
        HuggingFace model hub path, e.g. "meta-llama/Llama-3.1-8B-Instruct".
    quantization : str or None
        "4bit"  → load in 4-bit NF4 via bitsandbytes (recommended for ≤24 GB VRAM)
        "8bit"  → load in 8-bit via bitsandbytes
        None    → full precision (fp16 on CUDA, fp32 on CPU/MPS)

    Returns
    -------
    model : AutoModelForCausalLM
        Model with output_attentions always available via forward-pass kwarg.
    tokenizer : AutoTokenizer
    """
    logger.info(f"Loading model '{model_name}' (quantization={quantization}) ...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = None
    if quantization == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    elif quantization == "8bit":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16 if quantization is None else None,
        attn_implementation="eager",  # required to get per-head attention tensors
    )

    # Disable KV-cache so attention outputs are always accessible
    model.config.use_cache = False
    model.eval()

    logger.info(
        f"Model loaded. Layers: {model.config.num_hidden_layers}, "
        f"Heads: {model.config.num_attention_heads}"
    )
    return model, tokenizer


# ---------------------------------------------------------------------------
# Attention extraction
# ---------------------------------------------------------------------------


def get_attention_matrices(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    layers: Optional[list] = None,
    heads: Optional[list] = None,
) -> dict:
    """
    Run a single greedy forward pass and return post-softmax attention weights.

    Uses torch.no_grad() — never call this inside a training loop.

    Parameters
    ----------
    model : AutoModelForCausalLM
    tokenizer : AutoTokenizer
    prompt : str
        Raw text prompt.
    layers : list[int] or None
        Layer indices to return. None → all layers.
    heads : list[int] or None
        Head indices to return. None → all heads.

    Returns
    -------
    dict
        Keyed by (layer_idx, head_idx) → np.ndarray of shape (seq_len, seq_len).
        Values are post-softmax attention weights (rows sum to 1).

    Side-effects
    ------------
    Logs the sink token text (position 0) and its token id.
    """
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    input_ids: torch.Tensor = inputs["input_ids"]

    # Log the sink token
    sink_token_id = int(input_ids[0, 0])
    sink_token_text = tokenizer.decode([sink_token_id])
    logger.debug(
        f"Sink token: repr={repr(sink_token_text)!r}, id={sink_token_id}, "
        f"seq_len={input_ids.shape[1]}"
    )

    # Move inputs to model's device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # outputs.attentions: tuple of (1, n_heads, seq_len, seq_len) per layer
    n_layers = len(outputs.attentions)
    n_heads = outputs.attentions[0].shape[1]

    layer_range = layers if layers is not None else range(n_layers)
    head_range = heads if heads is not None else range(n_heads)

    result: dict = {}
    for layer_idx in layer_range:
        for head_idx in head_range:
            attn_tensor = outputs.attentions[layer_idx][
                0, head_idx
            ]  # (seq_len, seq_len)
            result[(layer_idx, head_idx)] = attn_tensor.float().cpu().numpy()

    return result


# ---------------------------------------------------------------------------
# Sink masking
# ---------------------------------------------------------------------------


def mask_sink(attn_matrix: np.ndarray, sink_idx: int = 0) -> np.ndarray:
    """
    Return a copy of ``attn_matrix`` with the sink token's row and column
    zeroed out, then row-renormalize so each row sums to 1.

    This simulates "what does the head attend to when the sink is unavailable?"

    Parameters
    ----------
    attn_matrix : np.ndarray
        Shape (seq_len, seq_len). Must be a 2-D float array.
    sink_idx : int
        Token position of the sink (default 0, i.e. the BOS / first token).

    Returns
    -------
    np.ndarray
        Copy of attn_matrix with sink row/column zeroed and rows renormalized.

    Raises
    ------
    ValueError
        If sink_idx is out of range for the given matrix.
    """
    seq_len = attn_matrix.shape[0]
    if not (0 <= sink_idx < seq_len):
        raise ValueError(f"sink_idx={sink_idx} is out of range for seq_len={seq_len}")

    masked = attn_matrix.copy()

    # Zero out all attention *to* the sink (column) and *from* the sink (row)
    masked[:, sink_idx] = 0.0
    masked[sink_idx, :] = 0.0

    # Renormalize non-sink rows so they sum to 1
    row_sums = masked.sum(axis=1, keepdims=True)
    # Avoid division by zero: rows that summed to zero stay zero
    nonzero_mask = (row_sums > 0).squeeze(axis=1)
    masked[nonzero_mask] = masked[nonzero_mask] / row_sums[nonzero_mask]

    return masked


# ---------------------------------------------------------------------------
# Convenience: extract paired (unmasked, masked) matrices for a head
# ---------------------------------------------------------------------------


def get_paired_attention(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    layer: int,
    head: int,
    sink_idx: int = 0,
) -> tuple:
    """
    Convenience wrapper: returns (attn_unmasked, attn_masked) for a single
    (layer, head) pair on a single prompt.

    Parameters
    ----------
    model, tokenizer : as in get_attention_matrices
    prompt : str
    layer, head : int
    sink_idx : int

    Returns
    -------
    attn_unmasked : np.ndarray  shape (seq_len, seq_len)
    attn_masked   : np.ndarray  shape (seq_len, seq_len)
    token_ids     : list[int]   flattened input_ids
    token_strs    : list[str]   decoded token strings (one per position)
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    token_ids: list = inputs["input_ids"][0].tolist()
    token_strs: list = [tokenizer.decode([tid]) for tid in token_ids]

    matrices = get_attention_matrices(
        model, tokenizer, prompt, layers=[layer], heads=[head]
    )
    attn_unmasked = matrices[(layer, head)]
    attn_masked = mask_sink(attn_unmasked, sink_idx=sink_idx)

    return attn_unmasked, attn_masked, token_ids, token_strs
