# EPD sizing comparison on four H800 GPUs

The experiment pairs the public `epd-provisioning` allocator with vLLM's frozen
per-stage calibration. It reuses the prepared vLLM overlay, Apptainer image,
calibration and model cache. The benchmark first samples request shapes through
the proxy at low load and feeds vLLM's estimates for per-GPU E and PD capacity
to the public allocator. The primary comparison is the recommended split
against 2E:2PD at the same GPU budget.

The trace deployment uses the EPD tools in the
[vLLM planner branch](https://github.com/Kelvin-Al/vllm/tree/epd-capacity-planner/examples/disaggregated/disaggregated_encoder/epd_capacity).
The published synthetic plan records the actual runner, allocator and
calibration digests. Use a frozen calibration for your exact deployment.

`workload.json` fixes a synthetic multimodal cohort with mid-sized images and
longer generations to exercise PD-heavy serving. Every wave uses the same seed
and request shapes. For each split, the benchmark tests 75% of predicted equal
capacity, the midpoint of predicted equal and recommended capacity, and 125%
of predicted recommended capacity. The experiment freezes all three offered
rates before either arm runs. Each arm starts fresh and the second round
reverses arm order. Both rounds run in one Slurm allocation.

The primary measurement is PD successful completions per second in the middle
60% of actual arrivals. The output also includes throughput including drain,
P90 TTFT and TPOT, actual arrival rate, and peak worker queues. The raw logs and
counters remain alongside `design.json`, `plan.json` and `results.json`. Report
both paired differences at each load. If the method also chooses 2E:2PD, the
runner records `same_split` and skips the duplicate comparison.

## Launch

Stage these three experiment files and the release's `planner.py` together in
a writable directory on the cluster. Submit from that directory after the
vLLM job finishes. The prepared deployment directory must include
`run_epd_superpod.py`, the overlay under `overrides/vllm`, and the proxy and
trace exporter under `examples/disaggregated/disaggregated_encoder`.

```bash
export EPD_SOURCE=<vllm_deployment_dir>
export EPD_CALIBRATION=<frozen_calibration_json>
export EPD_SIF=<apptainer_image>
export EPD_HF_CACHE=<model_cache>
sbatch --account=<slurm_account> --partition=normal experiment.sbatch
```
