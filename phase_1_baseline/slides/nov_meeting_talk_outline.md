# November Meeting — Phase 1 Baseline Highlights

1. **Motivation**
   - Refresh ACT intuition (Graves, 2016) and motivation for dynamic depth.
   - Clarify why TinyLlama is the chosen reference model.

2. **Reproduction Pipeline**
   - Walk through the `run_baseline.py` script and logging outputs.
   - Discuss prompt selection and measurement protocol.

3. **Key Metrics**
   - Layer-wise FLOP distribution (attention vs. feed-forward).
   - Latency profile on the reference GPU and CPU hosts.

4. **Energy Instrumentation Plan**
   - Outline integration with power sensors and logging schema updates.

5. **Outlook for Phase 2**
   - Planned dynamic depth controller experiments.
   - Required benchmarks and evaluation datasets.
