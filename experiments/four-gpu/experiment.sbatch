#!/usr/bin/env bash
#SBATCH --job-name=epd-method
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --time=02:00:00
#SBATCH --output=epd-method-%j.out

set -euo pipefail
root="${SLURM_SUBMIT_DIR:?Submit from the method experiment directory}"
source_dir="${EPD_SOURCE:?Set EPD_SOURCE to the vLLM trace deployment directory}"
calibration="${EPD_CALIBRATION:?Set EPD_CALIBRATION to the frozen JSON file}"
image="${EPD_SIF:?Set EPD_SIF to the prepared Apptainer image}"
export HF_HOME="${EPD_HF_CACHE:?Set EPD_HF_CACHE to the prepared Hugging Face cache}"
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_TELEMETRY=1 VLLM_NO_USAGE_STATS=1
export VLLM_ENABLE_CUDA_COMPATIBILITY=0 APPTAINERENV_VLLM_ENABLE_CUDA_COMPATIBILITY=0
if [[ "${EPD_METHOD_IN_CONTAINER:-0}" != 1 ]]; then
    source /etc/profile.d/modules.sh
    module load apptainer
    mkdir -p "$root/apptainer-tmp"
    export APPTAINER_TMPDIR="$root/apptainer-tmp"
    calibration_dir=$(dirname "$calibration")
    mmmu_bind=()
    if [[ -n "${EPD_MMMU_SOURCE:-}" ]]; then
        mmmu_bind=(--bind "$EPD_MMMU_SOURCE:$EPD_MMMU_SOURCE:ro")
    fi
    exec apptainer exec --nv \
        --bind "$root:$root" \
        --bind "$source_dir:$source_dir:ro" \
        --bind "$calibration_dir:$calibration_dir:ro" \
        --bind "$HF_HOME:$HF_HOME" \
        "${mmmu_bind[@]}" \
        "$image" env EPD_METHOD_IN_CONTAINER=1 bash "$root/experiment.sbatch"
fi

package=$(python3 -c 'import importlib.util; print(next(iter(importlib.util.find_spec("vllm").submodule_search_locations)))')
overlay=$(mktemp -d)
trap 'rm -rf -- "$overlay"' EXIT
mkdir "$overlay/vllm"
cp -as "$package/." "$overlay/vllm/"
cp --remove-destination -r "$source_dir/overrides/vllm/." "$overlay/vllm/"
export PYTHONPATH="$overlay${PYTHONPATH:+:$PYTHONPATH}"
gpus="${EPD_GPUS:-4}"
python3 - "$gpus" <<'PY'
import sys

import torch
import vllm

devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
if len(devices) != int(sys.argv[1]) or any("H800" not in device for device in devices):
    raise RuntimeError(f"Expected the requested number of H800 GPUs, found {devices}")
print(vllm.__version__, torch.__version__, torch.version.cuda, devices)
print(torch.ones(1, device="cuda"))
PY
cd "$root"
options=()
if [[ -n "${EPD_MMMU_REPLAY:-}" ]]; then
    options=(--all-splits --sample-rate 5)
    workload="$EPD_MMMU_REPLAY"
elif [[ -n "${EPD_MMMU_SOURCE:-}" ]]; then
    cohort="cohort-${SLURM_JOB_ID:?}"
    python3 "$root/mmmu.py" --source "$EPD_MMMU_SOURCE" \
        --output "$cohort" --calibration "$calibration"
    options=(--all-splits --sample-rate 5)
    workload="$cohort/replay.json"
else
    workload="$root/workload.json"
fi
python3 "$root/experiment.py" \
    --source "$source_dir" \
    --calibration "$calibration" \
    --planner "$root/planner.py" \
    --workload "$workload" \
    --output "$root/results-${SLURM_JOB_ID:?}" \
    --base-port "$((22000 + (SLURM_JOB_ID % 500) * 50))" \
    --gpus "$gpus" "${options[@]}"
