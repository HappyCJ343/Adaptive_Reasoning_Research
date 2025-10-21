"""Helpers for estimating TinyLlama layer-wise FLOPs.

The estimators implemented here follow the analytical formulas that are
commonly used for decoder-only Transformers.  The goal is to provide a
lightweight way to inspect the computational cost of each block without
requiring a heavy profiler or hardware specific tooling.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd
from transformers import PretrainedConfig


@dataclass
class FlopBreakdown:
    """Container for bookkeeping intermediate FLOP components."""

    qkv_proj: float
    attention_scores: float
    attention_context: float
    feed_forward: float
    misc: float

    @property
    def total(self) -> float:
        return self.qkv_proj + self.attention_scores + self.attention_context + self.feed_forward + self.misc


def _ffn_linear_flops(
    hidden_size: int,
    intermediate_size: int,
    sequence_length: int,
    gated_ffn: bool,
    multiply_add_factor: int,
) -> float:
    """Return FLOPs contributed by the feed-forward block for a single layer.

    The LLaMA/TinyLlama architecture uses a gated feed-forward block similar to
    SwiGLU.  This implies two input projections (gate and up) followed by a
    down projection.  When ``gated_ffn`` is set to ``False`` we assume the
    traditional Transformer MLP with one up and one down projection.
    """

    up_proj = sequence_length * hidden_size * intermediate_size
    down_proj = sequence_length * intermediate_size * hidden_size
    gate_proj = sequence_length * hidden_size * intermediate_size if gated_ffn else 0

    linear_terms = (up_proj + down_proj + gate_proj) * multiply_add_factor

    # Element-wise gating + activation (SiLU) roughly cost one multiply per
    # element.  We keep them separate from the matrix multiplications so that
    # they can be optionally excluded from the summary if desired.
    gating_mul = sequence_length * intermediate_size if gated_ffn else 0
    activation_mul = sequence_length * intermediate_size  # SiLU approx

    return linear_terms + gating_mul + activation_mul


def _attention_projection_flops(
    hidden_size: int,
    num_heads: int,
    num_key_value_heads: int,
    sequence_length: int,
    multiply_add_factor: int,
) -> float:
    """Return the FLOPs from the Q/K/V/O linear projections.

    The grouped-query attention employed by TinyLlama reduces the dimensionality
    for the key/value projections.  We model that explicitly via the
    ``num_key_value_heads`` argument.
    """

    head_dim = hidden_size // num_heads
    kv_dim = head_dim * num_key_value_heads

    q_proj = sequence_length * hidden_size * hidden_size
    k_proj = sequence_length * hidden_size * kv_dim
    v_proj = sequence_length * hidden_size * kv_dim
    o_proj = sequence_length * hidden_size * hidden_size

    return (q_proj + k_proj + v_proj + o_proj) * multiply_add_factor


def _attention_score_flops(
    num_heads: int,
    sequence_length: int,
    head_dim: int,
    multiply_add_factor: int,
    include_softmax: bool,
) -> FlopBreakdown:
    """Return a :class:`FlopBreakdown` for the self-attention block."""

    score_mat = num_heads * sequence_length * sequence_length * head_dim
    context_mat = num_heads * sequence_length * sequence_length * head_dim
    softmax_ops = num_heads * sequence_length * sequence_length if include_softmax else 0

    attention_scores = score_mat * multiply_add_factor
    attention_context = context_mat * multiply_add_factor
    misc = softmax_ops

    return FlopBreakdown(
        qkv_proj=0.0,  # placeholder; filled in higher level
        attention_scores=attention_scores,
        attention_context=attention_context,
        feed_forward=0.0,
        misc=misc,
    )


def estimate_layer_flops(
    config: PretrainedConfig,
    sequence_length: int,
    batch_size: int = 1,
    gated_ffn: bool = True,
    include_softmax: bool = True,
    multiply_add_factor: int = 2,
) -> pd.DataFrame:
    """Estimate the per-layer FLOPs of a decoder-only Transformer.

    Parameters
    ----------
    config:
        Hugging Face configuration object associated with the model.
    sequence_length:
        Number of tokens processed in the prompt (prefill) stage.
    batch_size:
        How many sequences are processed together.  FLOPs scale linearly with
        batch size.
    gated_ffn:
        Whether to account for an additional gate projection in the feed-forward
        block (as used in LLaMA/TinyLlama).
    include_softmax:
        Include the cost of the softmax normalization in the attention module.
    multiply_add_factor:
        Number of FLOPs per multiply-add.  ``2`` matches the common definition
        used in hardware whitepapers.

    Returns
    -------
    pandas.DataFrame
        Layer-wise FLOPs with columns describing the total and per-component
        contributions.  Values are expressed in absolute FLOPs, not GigaFLOPs,
        to avoid premature rounding.
    """

    hidden_size = config.hidden_size
    intermediate_size = config.intermediate_size
    num_heads = config.num_attention_heads
    num_key_value_heads = getattr(config, "num_key_value_heads", num_heads)
    num_layers = config.num_hidden_layers
    head_dim = hidden_size // num_heads

    rows = []
    for layer_idx in range(num_layers):
        qkv_proj_flops = _attention_projection_flops(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_key_value_heads=num_key_value_heads,
            sequence_length=sequence_length,
            multiply_add_factor=multiply_add_factor,
        )

        attention_breakdown = _attention_score_flops(
            num_heads=num_heads,
            sequence_length=sequence_length,
            head_dim=head_dim,
            multiply_add_factor=multiply_add_factor,
            include_softmax=include_softmax,
        )

        ff_flops = _ffn_linear_flops(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            sequence_length=sequence_length,
            gated_ffn=gated_ffn,
            multiply_add_factor=multiply_add_factor,
        )

        misc = attention_breakdown.misc
        total_layer_flops = batch_size * (
            qkv_proj_flops
            + attention_breakdown.attention_scores
            + attention_breakdown.attention_context
            + ff_flops
            + misc
        )

        per_token_flops = total_layer_flops / (batch_size * sequence_length)

        rows.append(
            {
                "layer_index": layer_idx,
                "sequence_length": sequence_length,
                "batch_size": batch_size,
                "total_flops": total_layer_flops,
                "total_gflops": total_layer_flops / 1e9,
                "per_token_flops": per_token_flops,
                "per_token_gflops": per_token_flops / 1e9,
                "qkv_proj_flops": batch_size * qkv_proj_flops,
                "attention_scores_flops": batch_size * attention_breakdown.attention_scores,
                "attention_context_flops": batch_size * attention_breakdown.attention_context,
                "feed_forward_flops": batch_size * ff_flops,
                "miscellaneous_flops": batch_size * misc,
            }
        )

    df = pd.DataFrame(rows)
    return df


def summarize_model_flops(layer_df: pd.DataFrame) -> Dict[str, float]:
    """Aggregate statistics derived from :func:`estimate_layer_flops` output."""

    if layer_df.empty:
        raise ValueError("Layer FLOP DataFrame is empty; ensure estimate_layer_flops executed correctly.")

    batch_size = int(layer_df["batch_size"].iloc[0])
    sequence_length = int(layer_df["sequence_length"].iloc[0])
    num_layers = len(layer_df)

    total_model_flops = float(layer_df["total_flops"].sum())
    per_token = total_model_flops / (batch_size * sequence_length)

    return {
        "num_layers": num_layers,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "total_model_flops": total_model_flops,
        "total_model_gflops": total_model_flops / 1e9,
        "model_per_token_flops": per_token,
        "model_per_token_gflops": per_token / 1e9,
    }
