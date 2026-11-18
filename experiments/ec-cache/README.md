# MMMU cold EC and prefilled EC experiments

The cold sweep compares all three four-GPU splits at a nominal 34.27
requests/s in two reversed orders on one H800 node. Fresh workers and empty
EC storage are used per wave. Request-scoped image keys are shared between
E and PD and prevent cross-request embedding reuse. The runner verifies
720 actual encoded images and zero EC or resident-cache hits per wave.

The cold/warm audit fixes 1E:3PD. A separate priming deployment encodes the
cohort into persisted EC files. Four fresh worker deployments run cold,
warm, warm and cold at 34.27 requests/s. Cold waves populate their initially
empty EC directory. Warm waves use the prefilled directory.

The encoder records successful visual-forward calls and the number of image
items they process. Scheduler instrumentation records the first cache
decision per request and image, distinguishing EC and resident hits.
Full-wave counts and counts in the middle 60% of actual arrivals are
reported alongside E/PD completion rates in the same central window.
This audit adds lightweight CPU event writes to both conditions.

## Stage and submit

Use the prepared vLLM trace deployment documented in the
[four-GPU experiment](../four-gpu/README.md). It supplies the upstream runner,
proxy, overlay, Apptainer image and frozen calibration.

From this repository root, stage the experiment code together.

```bash
mkdir -p <batch_directory>
cp experiments/ec-cache/{audit.py,audit.sbatch,cold_sweep.py} <batch_directory>/
cp experiments/four-gpu/experiment.py <batch_directory>/
```

Set these environment variables before submission.

```bash
export EPD_SOURCE=<prepared_vllm_trace_deployment>
export EPD_CALIBRATION=<frozen_calibration_json>
export EPD_SIF=<prepared_apptainer_image>
export EPD_HF_CACHE=<model_cache>
export EPD_COHORT_ROOT=<parent_of_cohort_directory>
cd <batch_directory>
export EPD_COLD_SWEEP=1
sbatch --account=<slurm_account> --partition=normal audit.sbatch
```

`EPD_COHORT_ROOT/cohort/replay.json` and its images must be prepared as in
the MMMU protocol in the four-GPU experiment. For the 1E:3PD cold/warm
audit, unset `EPD_COLD_SWEEP` before submission.

## Recorded results

- [Cold sweep](results/cold/README.md) has six complete waves and their
  [frozen plan](results/cold/plan.json) and [results](results/cold/results.json).
- [Cold/warm audit plan](results/cold-warm/plan.json) and
  [results](results/cold-warm/results.json) preserve all four measured waves
  and priming counts.
- Earlier [synthetic](../four-gpu/results/synthetic/results.json) and
  [MMMU replay](../four-gpu/results/mmmu/results.json) results remain preserved.

The source digests for `audit.py` and `cold_sweep.py` match the frozen
plans. The copied upstream runner and generated proxy digests refer to the
cluster-generated scripts. Those generated files contain deployment paths
and are reconstructed by the scripts here.

The original custom-image adapter did not forward its declared seed and
request count. Its measured waves used client seed 0 and the 720 supplied
rows. The cold sweep explicitly forwards seed 9197 and 720 requests.
Keep these execution differences when comparing across experiments.

The first cold attempt failed on the first split because the proxy assigned
inconsistent producer and consumer keys. The corrected code shares the key
before both requests are built. The full six-wave rerun is reported.
The [failed attempt's frozen plan](results/failed-cold-attempt/plan.json)
and [failure status](results/failed-cold-attempt/status.json) are retained.
