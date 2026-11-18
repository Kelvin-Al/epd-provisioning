#!/usr/bin/env bash
#SBATCH --job-name=epd-ec-audit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --time=01:15:00
#SBATCH --output=epd-ec-audit-%j.out

set -euo pipefail
root="${SLURM_SUBMIT_DIR:?}"
source_dir="${EPD_SOURCE:?}"
calibration="${EPD_CALIBRATION:?}"
cohort="${EPD_COHORT_ROOT:?}"
image="${EPD_SIF:?}"
export HF_HOME="${EPD_HF_CACHE:?}"
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_TELEMETRY=1 VLLM_NO_USAGE_STATS=1
export VLLM_ENABLE_CUDA_COMPATIBILITY=0 APPTAINERENV_VLLM_ENABLE_CUDA_COMPATIBILITY=0
if [[ "${EPD_AUDIT_IN_CONTAINER:-0}" != 1 ]]; then
    source /etc/profile.d/modules.sh
    module load apptainer
    mkdir -p "$root/apptainer-tmp"
    export APPTAINER_TMPDIR="$root/apptainer-tmp"
    cal_dir=$(dirname "$calibration")
    exec apptainer exec --nv \
        --bind "$root:$root" --bind "$source_dir:$source_dir:ro" \
        --bind "$cal_dir:$cal_dir:ro" --bind "$cohort:$cohort:ro" \
        --bind "$HF_HOME:$HF_HOME" \
        "$image" env EPD_AUDIT_IN_CONTAINER=1 bash "$root/audit.sbatch"
fi

package=$(python3 -c 'import importlib.util; print(next(iter(importlib.util.find_spec("vllm").submodule_search_locations)))')
overlay=$(mktemp -d)
trap 'rm -rf -- "$overlay"' EXIT
mkdir "$overlay/vllm"
cp -as "$package/." "$overlay/vllm/"
cp --remove-destination -r "$source_dir/overrides/vllm/." "$overlay/vllm/"
export PYTHONPATH="$overlay${PYTHONPATH:+:$PYTHONPATH}"
python3 - <<'PY'
import torch
import vllm

devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
if len(devices) != 4 or any("H800" not in device for device in devices):
    raise RuntimeError(f"Expected four H800 GPUs, found {devices}")
print(vllm.__version__, torch.__version__, torch.version.cuda, devices)
print(torch.ones(1, device="cuda"))
PY
cd "$cohort"
entry=audit.py
if [[ "${EPD_COLD_SWEEP:-0}" == 1 ]]; then
    entry=cold_sweep.py
fi
python3 "$root/$entry" --overlay "$overlay" --source "$source_dir" \
    --replay "$cohort/cohort/replay.json" --calibration "$calibration" \
    --output "$root/results-${SLURM_JOB_ID:?}" \
    --port "$((34000 + (SLURM_JOB_ID % 400) * 30))"
