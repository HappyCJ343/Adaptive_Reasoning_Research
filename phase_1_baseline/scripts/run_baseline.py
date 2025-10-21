"""Command line entrypoint for reproducing the TinyLlama baseline metrics."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import List, Sequence

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from phase_1_baseline.src.flops_utils import estimate_layer_flops, summarize_model_flops
from phase_1_baseline.src.inference_timer import InferenceSummary, measure_inference_latency

DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_PROMPTS: List[str] = [
    "Summarize the core idea behind adaptive computation time in neural networks.",
    "Explain why grouped-query attention can reduce the inference cost of large language models.",
    "List three practical applications where TinyLlama-sized models are beneficial.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TinyLlama baseline reproduction helper")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL, help="Model identifier from Hugging Face hub or local path")
    parser.add_argument("--sequence-length", type=int, default=128, help="Prompt token length used for FLOP estimation")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of prompts to evaluate per batch")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Maximum number of tokens to generate during latency measurement")
    parser.add_argument("--device", type=str, default=None, help="Target device identifier (e.g., cuda, cuda:0, cpu)")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"], help="Torch dtype for model weights")
    parser.add_argument("--warmup-runs", type=int, default=1, help="Number of warm-up runs before measuring latency")
    parser.add_argument("--measurement-runs", type=int, default=3, help="Number of timing runs to average over")
    parser.add_argument("--prompt-file", type=str, default=None, help="Optional path to a newline-delimited prompt file")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory used to store logs (defaults to phase_1_baseline/logs)")
    parser.add_argument("--skip-inference", action="store_true", help="Skip loading the model and measuring latency")
    parser.add_argument("--disable-gated-ffn", action="store_true", help="Assume a non-gated feed-forward block when estimating FLOPs")
    parser.add_argument("--no-softmax-flops", action="store_true", help="Exclude softmax ops from the FLOP estimate")
    parser.add_argument("--print-summary", action="store_true", help="Print metrics summary as JSON")
    return parser.parse_args()


def load_prompts(prompt_file: str | None) -> List[str]:
    if prompt_file is None:
        return DEFAULT_PROMPTS

    path = Path(prompt_file)
    if not path.exists():
        raise FileNotFoundError(f"Prompt file {prompt_file} does not exist")

    with path.open("r", encoding="utf-8") as handle:
        prompts = [line.strip() for line in handle if line.strip()]
    if not prompts:
        raise ValueError("Prompt file did not contain any non-empty lines")
    return prompts


def resolve_device(device_str: str | None) -> torch.device:
    if device_str:
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_dtype(dtype_str: str, device: torch.device) -> torch.dtype:
    if dtype_str == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = mapping[dtype_str]
    if device.type == "cpu" and dtype in {torch.float16}:
        raise ValueError("float16 weights are not supported on CPU devices")
    return dtype


def prepare_tokenizer(model_name: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def measure_latency(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: Sequence[str],
    args: argparse.Namespace,
) -> InferenceSummary:
    return measure_inference_latency(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        warmup_runs=args.warmup_runs,
        measurement_runs=args.measurement_runs,
        max_length=args.sequence_length,
    )


def write_layerwise_flops(layer_df, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    layer_df.to_csv(output_path, index=False)


def append_performance_row(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp",
        "model_name",
        "device",
        "dtype",
        "batch_size",
        "sequence_length",
        "max_new_tokens",
        "warmup_runs",
        "measurement_runs",
        "avg_latency_s",
        "std_latency_s",
        "tokens_per_second",
        "tokens_generated",
        "prompt_tokens",
        "total_model_flops",
        "total_model_gflops",
        "model_per_token_flops",
        "model_per_token_gflops",
    ]

    exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    prompts = load_prompts(args.prompt_file)

    script_dir = Path(__file__).resolve().parent
    phase_root = script_dir.parent
    log_dir = Path(args.output_dir) if args.output_dir else phase_root / "logs"
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    config = AutoConfig.from_pretrained(args.model_name)
    layer_df = estimate_layer_flops(
        config=config,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        gated_ffn=not args.disable_gated_ffn,
        include_softmax=not args.no_softmax_flops,
    )
    layer_path = log_dir / f"layerwise_flops_{timestamp}.csv"
    write_layerwise_flops(layer_df, layer_path)
    flops_summary = summarize_model_flops(layer_df)

    metrics_row = {
        "timestamp": timestamp,
        "model_name": args.model_name,
        "device": str(device),
        "dtype": str(dtype),
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "max_new_tokens": args.max_new_tokens,
        "warmup_runs": args.warmup_runs,
        "measurement_runs": args.measurement_runs,
        "avg_latency_s": None,
        "std_latency_s": None,
        "tokens_per_second": None,
        "tokens_generated": None,
        "prompt_tokens": len(prompts) * args.sequence_length,
        "total_model_flops": flops_summary["total_model_flops"],
        "total_model_gflops": flops_summary["total_model_gflops"],
        "model_per_token_flops": flops_summary["model_per_token_flops"],
        "model_per_token_gflops": flops_summary["model_per_token_gflops"],
    }

    latency_summary: InferenceSummary | None = None
    if not args.skip_inference:
        tokenizer = prepare_tokenizer(args.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        model.to(device)
        latency_summary = measure_latency(model, tokenizer, prompts, args)

        metrics_row.update(
            {
                "avg_latency_s": latency_summary.avg_latency_s,
                "std_latency_s": latency_summary.std_latency_s,
                "tokens_per_second": latency_summary.tokens_per_second,
                "tokens_generated": latency_summary.tokens_generated,
                "prompt_tokens": latency_summary.prompt_tokens,
            }
        )

    csv_path = log_dir / "performance_metrics.csv"
    append_performance_row(csv_path, metrics_row)

    summary_payload = {
        "layerwise_flops_file": str(layer_path),
        "metrics_file": str(csv_path),
        "flops_summary": flops_summary,
        "latency_summary": None if latency_summary is None else latency_summary.__dict__,
    }
    if args.print_summary:
        print(json.dumps(summary_payload, indent=2))
    else:
        print("Saved layerwise FLOPs to", layer_path)
        print("Appended metrics to", csv_path)


if __name__ == "__main__":
    main()
