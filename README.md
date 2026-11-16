# EPD GPU provisioning

> **Validated on two NVIDIA H800 GPUs.** The 1E:1PD smoke test wrote encoder
> cache files and returned a complete streamed response with Qwen2.5-VL-3B-Instruct.
> In a 30-second low-load MMMU validation replay, 20 of 21 requests completed
> within the window, with zero errors and one unfinished request.

Start with the [cluster quickstart](CLUSTER.md) to prepare MMMU, launch vLLM
workers, and measure E+PD allocations.

## Four-H800 allocator replay

On one H800 node, the allocator selected **1E:3PD** from live request-shape
samples for a synthetic multimodal workload with midsize images and long
generations. The two splits used the same four GPUs, Qwen2.5-VL-3B-Instruct, the
vLLM 0.33.0 trace deployment, and identical seeded 480-request waves.

| Actual offered requests/s | 2E:2PD completions/s | 1E:3PD completions/s | Paired changes |
| ---: | ---: | ---: | ---: |
| 6.07 | 6.11–6.13 | 6.03–6.09 | −0.7%, −1.4% |
| 10.11 | 8.82–8.83 | 10.15–10.18 | +15.5%, +15.0% |
| 15.17 | 8.55–8.90 | 12.91–12.98 | +51.1%, +46.0% |

Completion rates use the middle 60% of actual arrivals. At the highest
load, P90 time to first token fell from 19.7–23.5 s to 6.0–7.8 s. All 5,760
requests completed across two rounds, including drain. The gain appeared
when the equal split's PD pool queued requests. At low load, both splits
kept up with arrivals. This run shows that the allocator can choose an uneven
split and improve serving under a PD-heavy workload. The measured gains apply
to this workload and node.

[Experiment code](experiments/four-gpu/) ·
[Frozen plan](experiments/four-gpu/results/synthetic/plan.json) ·
[Measured results](experiments/four-gpu/results/synthetic/results.json) ·
[Evidence scope and MMMU follow-up](https://github.com/Kelvin-Al/epd-provisioning/issues/1)

## Earlier four-H800 MMMU split sweep

**2E:2PD achieved the highest completed-request throughput in both rounds**
of a same-node comparison against 1E:3PD and 3E:1PD. Each round cycled through the
same 128 MMMU validation requests with the same arrival schedule across
splits, offering 100 requests/s for a 45-second measurement window after
warmup. All six runs recorded zero request errors and zero client drops.
Unfinished requests at the end of the window were retained in the counts.

Absolute throughput varied across nodes. In particular, 3E:1PD completed
requests substantially faster in the initial sweep on another H800 node.
That sweep also hit the client concurrency limit for 3E:1PD at its highest
offered rate. The ranking above comes from the matched comparison on the
same node. Capacity estimates are specific to the measured configuration
and workload.

This example is a fresh Python implementation of the allocation rule and encoder
resource-demand method in *Queueing-Theoretic GPU Provisioning for
Disaggregated Multimodal LLM Serving*. One GPU serves each effective encoder (E)
or joint prefill/decode (PD) instance. Standalone commands run from the repository
root with Python 3.10 or later. Only the calibration fitter requires NumPy.

## Compute a split

```bash
python epd.py plan \
  --gpus 8 --e-rate 57 --pd-rate 14.3 --target-rps 80
```

Rates are requests per second per instance. The planner tests the feasible floor
and ceiling of the fractional crossing and reports the maximizing integer
splits, the selected split, its bottleneck, the decision interval for the rate
ratio, and the strict target-rate check. With the paper's H800 rate inputs it
selects 2E:6PD and gives a model boundary of 85.8 req/s. `--incumbent 2` retains
an already deployed 2-E split if it ties for first. Without an incumbent, ties
use the smaller encoder pool.

- `--rate-bounds E_LOW E_HIGH PD_LOW PD_HIGH` optimizes the lower-rate corner.
  The result also reports whether the chosen split stays optimal throughout the
  supplied rectangle. The coverage of the rate bounds is an external input.
- `--ttft-rates E_LIMIT PD_LIMIT` optimizes the boundary using supplied stage
  limits. Interpreting it as a TTFT guarantee requires the uniform stationary
  tail conditions in Appendix A.3. TPOT requires separate validation.
- A target equal to the boundary fails the strict load check. A below-boundary
  result under the effective-server model alone carries no latency guarantee.

## Calibrate from measurements

Provide a JSON object with one `anchors` row per visual-token count and a
`parallelism` object. Example row and parallelism format:

```json
{
  "anchors": [{
    "visual_tokens": 256,
    "serial_ms":    {"H": 0.3, "M": 20, "C": 30, "X": 32, "G": 27, "R": 5},
    "saturated_ms": {"H": 0.2, "M": 20, "C": 0.4, "X": 32, "G": 27, "R": 0.2}
  }],
  "parallelism": {"H": 1, "M": 2, "C": 1, "X": 1, "G": 1, "R": 1}
}
```

The example illustrates the format. Supply at least three distinct anchors for
fitting and use five to eight for the paper's measurement protocol. H is ingress,
M preprocessing, C coordination, X the worker–GPU cycle, G the nested GPU
interval, and R result handoff. Serial inputs are mean spans with one request in
flight. Saturated inputs are **active resource milliseconds per completion**
from an encoder-limited pass, before dividing by effective parallelism.
Exclude queueing and aggregate components sharing a limited resource before
using this six-lane schema. The default fit forms follow the paper's Table 4.
Each anchor receives equal weight. Check the fits and saturation plateaus on
the target before using the resulting rates operationally.

```bash
python epd.py calibrate \
  --input anchors.json --output encoder-model.json
python epd.py plan --gpus 8 \
  --encoder-model encoder-model.json --workload workload.json --pd-rate 14.3
```

`workload.json` is a JSON array such as
`[{"visual_tokens": 256}, {"visual_tokens": 576}]`. Include one row per
request. Supply a measured effective PD rate, or omit it and include
`pd_demand_ms` in each row from your calibrated PD demand model. The encoder
resource-load rate is an **upper bound**, reported explicitly as a planning
proxy. The code averages each normalized lane over the traffic mix before
taking the maximum, per Appendix A.2. Serial elapsed time uses H+M+C+X+R
because X already includes G. Predictions outside the anchor range are rejected.

## Run a CPU A/B comparison through InferRSI

For optional InferRSI integration, run the commands in this section from an [InferRSI checkout](https://github.com/InferMatrixLab/InferRSI) with this package at `examples/epd_provisioning`.

The synthetic benchmark implements two FIFO pools, independently uniform
routing, Poisson offered arrivals, and constant per-request services of
`1 / stage_rate`. Both arms get the same seeded arrivals, service requirements,
and routing draws. It measures PD completions per second in a fixed window,
retains queued counts at cutoff, and excludes drain after the window. This
demonstrates the effective tandem model, with no vLLM batching, connector,
or accelerator effects. A single frozen seed repeated by the runner verifies
deterministic replay; use distinct prespecified seeds for sampling variability.

InferRSI's source verification requires a Git checkout with a commit. If you
unpacked a source ZIP and have no Git checkout, initialize and commit your
local copy before running the integration command. The standalone planner and
validation commands do not need Git.

```bash
export PYTHONPATH="$PWD/src"
python examples/epd_provisioning/epd.py validate
common='--gpus 8 --e-rate 57 --pd-rate 14.3 --offered-rps 130 --seed 17 --warmup 30 --duration 120'
python -m inferrsi run --repo "$PWD" --backend cpu --repo-type vllm \
  --python "$(command -v python)" --scenario multimodal.epd_provisioning \
  --baseline-command "python examples/epd_provisioning/epd.py simulate --policy equal $common --output {result}" \
  --candidate-command "python examples/epd_provisioning/epd.py simulate --policy provisioned $common --output {result}" \
  --validation-command "python examples/epd_provisioning/epd.py validate" \
  --metric-key metrics.throughput_rps --metric-unit requests/s --direction maximize \
  --require-trial-check checks.valid_simulation --comparison-scope synthetic_fifo \
  --repetitions 2 --max-minutes 2 --output-dir "$(mktemp -d)"
```

Validation checks the planner against exhaustive integer search and exercises
the rate interval, strict sizing, mixed resource demands, and FIFO semantics.
The validation script tests the model and implementation; it does not check
the outputs of a deployed VLM.

## Use with real serving

See [the cluster guide](CLUSTER.md) for a standalone single-node NVIDIA Linux
runner, dataset preparation, and comparison commands.

Use the planner to select the candidate split for your measured model and
hardware. Keep the same total GPU budget, model revision, image trace, cache
policy, offered rate, warmup, measurement window and output-token limit across
arms. Replace the synthetic trial commands with your own EPD deployment and
load-generator wrappers. Have each wrapper emit its throughput at
`INFERRSI_RESULT_PATH` (or pass `{result}` as an output argument), retain
failed and unfinished request counts, and clean up the processes it starts.
Supply a serving-specific validation command and gates for output correctness
and any SLO claim. Saturation throughput and SLO-compliant throughput require
different qualification rules. The calibration example takes measurement
summaries as input. The cluster runner and dataset adapters are new
implementations. The paper's original profiler hooks, traces and deployment
scripts are unavailable.
