"""Utility functions for benchmarking TinyLlama inference latency."""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.tokenization_utils_base import BatchEncoding


@dataclass
class InferenceBatch:
    prompts: Sequence[str]
    encoding: BatchEncoding
    prompt_token_counts: Sequence[int]


@dataclass
class InferenceSummary:
    latencies: List[float]
    tokens_generated: int
    tokens_per_second: float
    avg_latency_s: float
    std_latency_s: float
    prompt_tokens: int
    generated_tokens_per_example: float


def _chunked(sequence: Sequence[str], batch_size: int) -> Iterable[Sequence[str]]:
    for index in range(0, len(sequence), batch_size):
        yield sequence[index : index + batch_size]


def _prepare_batches(
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str],
    batch_size: int,
    max_length: Optional[int] = None,
) -> List[InferenceBatch]:
    batches: List[InferenceBatch] = []
    for batch_prompts in _chunked(prompts, batch_size):
        encoding = tokenizer(
            list(batch_prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        attention_mask = encoding.get("attention_mask")
        if attention_mask is None:
            raise ValueError("Tokenizer must provide an attention_mask for latency measurement.")

        prompt_token_counts = attention_mask.sum(dim=1).tolist()
        batches.append(
            InferenceBatch(
                prompts=batch_prompts,
                encoding=encoding,
                prompt_token_counts=prompt_token_counts,
            )
        )
    return batches


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)


def measure_inference_latency(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str],
    max_new_tokens: int,
    batch_size: int = 1,
    warmup_runs: int = 1,
    measurement_runs: int = 3,
    generation_kwargs: Optional[Mapping[str, object]] = None,
    max_length: Optional[int] = None,
) -> InferenceSummary:
    """Benchmark greedy generation latency for a set of prompts.

    The function performs an optional warm-up phase before collecting timing
    statistics.  Latency is measured for the end-to-end ``model.generate`` call,
    including sampling overhead.  Averages are reported across measurement runs.
    """

    if measurement_runs <= 0:
        raise ValueError("measurement_runs must be positive.")
    if warmup_runs < 0:
        raise ValueError("warmup_runs must be non-negative.")

    device = next(model.parameters()).device
    model.eval()

    batches = _prepare_batches(
        tokenizer=tokenizer,
        prompts=list(prompts),
        batch_size=batch_size,
        max_length=max_length,
    )

    if not batches:
        raise ValueError("No prompts were provided for inference measurement.")

    gen_kwargs: Dict[str, object] = {"do_sample": False, "use_cache": True}
    if generation_kwargs:
        gen_kwargs.update(generation_kwargs)

    def _run_once() -> Dict[str, float]:
        total_latency = 0.0
        total_new_tokens = 0
        total_prompt_tokens = 0

        for batch in batches:
            encoded = batch.encoding.to(device)
            start = time.perf_counter()
            outputs = model.generate(**encoded, max_new_tokens=max_new_tokens, **gen_kwargs)
            _synchronize(device)
            latency = time.perf_counter() - start
            total_latency += latency

            input_length = encoded["input_ids"].shape[-1]
            output_length = outputs.shape[-1]
            generated = (output_length - input_length) * outputs.shape[0]
            total_new_tokens += int(generated)
            total_prompt_tokens += int(sum(batch.prompt_token_counts))

        return {
            "latency": total_latency,
            "new_tokens": total_new_tokens,
            "prompt_tokens": total_prompt_tokens,
        }

    # Warm-up runs to stabilize caches / graph compilation.
    for _ in range(warmup_runs):
        _run_once()

    latencies: List[float] = []
    total_new_tokens = 0
    total_prompt_tokens = 0
    for _ in range(measurement_runs):
        result = _run_once()
        latencies.append(result["latency"])
        total_new_tokens += int(result["new_tokens"])
        total_prompt_tokens += int(result["prompt_tokens"])

    avg_latency = statistics.mean(latencies)
    std_latency = statistics.pstdev(latencies) if len(latencies) > 1 else 0.0
    total_time = sum(latencies)
    tokens_per_second = total_new_tokens / total_time if total_time > 0 else 0.0
    examples = len(prompts)
    generated_tokens_per_example = total_new_tokens / examples if examples > 0 else 0.0

    return InferenceSummary(
        latencies=latencies,
        tokens_generated=total_new_tokens,
        tokens_per_second=tokens_per_second,
        avg_latency_s=avg_latency,
        std_latency_s=std_latency,
        prompt_tokens=total_prompt_tokens,
        generated_tokens_per_example=generated_tokens_per_example,
    )
