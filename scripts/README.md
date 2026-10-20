# `scripts/`

## `runtime/` — runtime benchmarking results

| File | Produced by | Measures |
|---|---|---|
| `results/bridge_data_generation_timings.json` | `data_pipeline/benchmark_pipeline.py` | per-stage data-generation time on BridgeDataV2 — the "Data-generation stage" column of the README runtime table |
| `results/bridge_dummy_inference_latency.json` | `sge_goal/pipelines/flux_kontext_bridge_calvin/run_pipeline_bridge_dummy_latency.py` | per-step inference latency summed over a 10-image BridgeDataV2 set; divided by 10 this gives the "Inference stage" column |
| `results/bridge_inference_latency.json` | an earlier version of the same latency driver | — |

```bash
cd data_pipeline && PYTHONPATH=./XMem python3 benchmark_pipeline.py --batch_size 100 --cleanup
```
