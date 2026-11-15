# E+PD experiments on a single NVIDIA node

> The H800 smoke test and low-load MMMU replay used CUDA 12.9 and the
> PyTorch sampler. The [four-H800 comparison](README.md#four-h800-split-comparison)
> reports the same-node results and cross-node variation.

Use one Linux node with an NVIDIA driver compatible with the vLLM CUDA wheel,
Python 3.12, local scratch, and at least two allocated GPUs. Each GPU hosts one
vLLM instance. The default model follows the upstream E+PD example. For the
paper's larger model, pass `--model Qwen/Qwen3.5-9B` and size the context and GPU
memory settings for your node. Use `--model-revision` to pin the weights.

```text
workload → load driver → E worker → shared encoder-cache files
            load driver ← PD worker ← shared encoder-cache files
```

The driver waits for the E worker's embedding files, then sends the full
request to a combined prefill/decode worker and reads streamed output. It
uses loopback connections. Workers stop after every wave and the runner
removes that wave's private cache. Point `--cache-root` at local scratch with
space for the embeddings generated during one wave.

## Install and send a real request

From the package directory on an allocated x86-64 Linux node with a driver
compatible with CUDA 12.9, use the H800 configuration below. The vLLM wheel
and PyTorch index must match. Set the sampler environment variable before
starting each smoke or benchmark run.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
VLLM_WHEEL='https://github.com/vllm-project/vllm/releases/download/v0.22.0/vllm-0.22.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl'
VLLM_SHA256='405319a7fde1af416eef6cbab0315d354d8b9c1a45b97097a9f1c3707bf05f74'
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu129 \
  "${VLLM_WHEEL}#sha256=${VLLM_SHA256}" -r requirements-cluster.txt
python -m pip check
export VLLM_USE_FLASHINFER_SAMPLER=0
python epd.py validate
mkdir -p runs
python cluster.py prepare --dataset mmmu --samples 32 --output workload-mmmu
python cluster.py smoke --gpus 0,1 --workload workload-mmmu --output runs/smoke
```

Use the GPU identifiers assigned to your job. Omit `--gpus` if the scheduler
sets `CUDA_VISIBLE_DEVICES`. For offline deployment, prepare the workload on a
connected machine and copy the directory and a model snapshot. Pass the snapshot
directory as `--model` and install dependencies from the cluster's wheel mirror.
Use a new output directory for each command.

## Select a workload

| CLI value | Source | Default split |
| --- | --- | --- |
| `mmmu` | [MMMU](https://huggingface.co/datasets/MMMU/MMMU) | Validation across all subject subsets, with images and multiple-choice options |
| `vqav2` | [VQAv2](https://huggingface.co/datasets/lmms-lab-encoder/VQAv2) | Validation, image and question |
| `textvqa` | [TextVQA](https://huggingface.co/datasets/lmms-lab-encoder/textvqa) | Validation, image and question |

Preparation streams data, records the resolved revision and materializes
the selected images. Use `--revision` to reuse a dataset revision and `--subset`
to restrict MMMU subjects. Sampling is seeded and respects `--max-images` and
`--max-pixels`. For your own dataset, put a `workload.json` and its images in
one directory:

```json
{"requests": [{"text": "What is in this image?", "images": ["images/example.png"]}]}
```

Review the upstream dataset terms before redistributing materialized data.

## Measure and compare splits

Start with a load sweep. Treat these offered rates as examples and adjust them
for the target model and hardware. Estimate the PD rate from a plateau with one
PD worker and enough E workers to feed it. With two GPUs, establish the
limiting stage before interpreting the 1:1 result as a PD rate.

```bash
python cluster.py bench --gpus 0 --e-only --workload workload-mmmu \
  --rps 5 --rps 15 --rps 30 --output runs/encoder

python cluster.py bench --gpus 0,1,2,3 --workload workload-mmmu \
  --split 1 --split 2 --split 3 --rps 5 --rps 15 --rps 30 \
  --repetitions 2 --output runs/sweep
```

Set `E_RATE` and `PD_RATE` to measured requests per second per instance. Set
`TTFT_MS` and `TPOT_MS` to the latency budgets for this experiment.

```bash
python cluster.py bench --gpus 0,1,2,3 --workload workload-mmmu \
  --e-rate "${E_RATE:?}" --pd-rate "${PD_RATE:?}" --rps 15 --rps 30 \
  --ttft-ms "${TTFT_MS:?}" --tpot-ms "${TPOT_MS:?}" \
  --repetitions 2 --output runs/comparison
```

`--split` specifies the number of encoder GPUs. Supplying rates compares the
equal split with the planner's recommendation. When they coincide, the runner
measures that allocation once per rate and repetition. Each repetition uses
the same seeded Poisson schedule and image order across splits. The split order
reverses on alternating repetitions. Per-request IDs provide cold encoder
inputs even when replay repeats the same images.
Each wave uses a new consecutive block of loopback ports starting from
`--base-port`, with one port per active worker. For concurrent runs on the
same node, choose non-overlapping ranges.

Outputs include the plan, workload digest, worker logs, per-request timing and
wave summaries. Throughput counts completions inside the timed window. Latency
percentiles use completed requests with the respective metric. The fraction
within both latency budgets uses all measured arrivals, including errors,
unfinished work and client drops in the denominator. Undefined TPOT does not
qualify. A nonzero client-drop count indicates the load driver hit its in-flight
cap. Use separate campaign criteria for model quality and hardware acceptance.

The launcher follows the [vLLM v0.22.0 E+PD interface](https://docs.vllm.ai/en/v0.22.0/examples/disaggregated/disaggregated_encoder/)
and [multimodal request IDs](https://docs.vllm.ai/en/v0.22.0/features/multimodal_inputs/#cached-inputs).
