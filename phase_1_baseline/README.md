# Phase 1 — TinyLlama Baseline Reproduction

This directory collects the artefacts used to reproduce the TinyLlama
baseline for the adaptive reasoning project.  The workflow is organised as
follows:

- `scripts/run_baseline.py` — command line helper that estimates layer-wise
  FLOPs and, optionally, measures end-to-end generation latency.
- `src/` — reusable utilities shared by notebooks and scripts.
- `logs/` — CSV artefacts produced by the baseline script.
- `reports/` — written summaries of the reproduction effort.
- `slides/` — presentation material for the November project meeting.

## Usage

The scripts assume that PyTorch (>=2.1) and Hugging Face Transformers (>=4.40)
are available in the Python environment:

```bash
pip install torch transformers pandas
```

```bash
python phase_1_baseline/scripts/run_baseline.py \
  --model-name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --sequence-length 128 \
  --batch-size 1 \
  --max-new-tokens 64 \
  --measurement-runs 3 \
  --print-summary
```

The script will download the TinyLlama weights, estimate the per-layer FLOPs,
run the requested number of latency trials, and save the artefacts under
`phase_1_baseline/logs/`.

Use `--skip-inference` if only the analytical FLOP estimate is required (for
example when running on a lightweight machine).

For convenience a small set of default prompts is provided.  Custom prompts can
be supplied with `--prompt-file path/to/prompts.txt` where each line stores one
prompt.

## Reproducing the Reported Metrics

1. Run the baseline script on the target hardware (CPU or GPU) to populate
   `logs/performance_metrics.csv`.
2. Inspect the generated layer-wise FLOPs CSV files for diagnosing expensive
   Transformer blocks.
3. Summarise the findings in `reports/replication_report.md` — an initial draft
   is provided and should be updated as new evidence becomes available.

This setup serves as the ground truth for later phases that investigate
adaptive depth and dynamic computation strategies.
