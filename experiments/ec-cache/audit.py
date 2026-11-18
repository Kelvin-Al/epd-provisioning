"""Cold and prefilled EC replay on the same four H800 GPUs."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from experiment import central_metrics, custom_image_runner, digest, read, resolve_cohort, validate_corpus, write


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Expected one occurrence of {old!r}")
    return source.replace(old, new)


def instrument(overlay):
    def edit(name, update):
        path = overlay / "vllm" / name
        source = update(path.read_text(encoding="utf-8"))
        compile(source, str(path), "exec")
        if path.is_symlink():
            path.unlink()
        path.write_text(source, encoding="utf-8")

    def profiling(source):
        base = source + '''

_AUDIT_FILE = os.environ.get("EPD_EC_AUDIT_FILE")


@cache
def _audit_fd():
    return os.open(_AUDIT_FILE, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)


def audit(stage, **fields):
    if _AUDIT_FILE:
        event = {"stage": stage, "time_ns": time.monotonic_ns(), **fields}
        os.write(_audit_fd(), (json.dumps(event, separators=(",", ":")) + "\\n").encode())
'''
        return replace_once(
            base, "    events = _gpu_events.get()\n    if events is None:\n        yield\n        return\n",
            '    events = _gpu_events.get()\n    if events is None:\n        yield\n'
            '        audit("visual_forward", items=num_items)\n        return\n',
        )

    def scheduler(source):
        marker = "                if self.encoder_cache_manager.check_and_update_cache(request, i):\n"
        source = replace_once(
            source, marker,
            marker + '                    if self.is_mm_encoder_only:\n'
            '                        audit("resident", request=request.request_id, image=i)\n',
        )
        marker = "                external_load_encoder_input.append(i)\n"
        source = replace_once(
            source, marker,
            marker + '                if self.is_mm_encoder_only:\n'
            '                    audit("ec_hit", request=request.request_id, image=i)\n',
        )
        marker = "            encoder_inputs_to_schedule.append(i)\n"
        source = replace_once(
            source, marker,
            marker + '            if self.is_mm_encoder_only:\n'
            '                audit("ec_miss", request=request.request_id, image=i)\n',
        )
        # Place the import before the engine imports, after any future import.
        return replace_once(
            source, "from vllm.distributed.ec_transfer.ec_connector.base import",
            "from vllm.epd_profiling import audit\nfrom vllm.distributed.ec_transfer.ec_connector.base import",
        )

    edit("epd_profiling.py", profiling)
    edit("v1/core/sched/scheduler.py", scheduler)


def runner_for(args):
    original = args.source / "run_epd_superpod.py"
    path = custom_image_runner(original, args.output / "runner.py")
    source = path.read_text(encoding="utf-8")
    source = replace_once(
        source, '    cache = args.output / "ec-cache"\n    cache.mkdir()\n',
        '    shared_cache = os.environ.get("EPD_SHARED_EC_CACHE")\n'
        '    cache = Path(shared_cache) if shared_cache else args.output / "ec-cache"\n'
        '    cache.mkdir(exist_ok=bool(shared_cache))\n',
    )
    source = replace_once(
        source, '            if encoder:\n                env["VLLM_MEDIA_LOADING_THREAD_COUNT"] = "1"\n',
        '            if encoder:\n                env["VLLM_MEDIA_LOADING_THREAD_COUNT"] = "1"\n'
        '                env["EPD_EC_AUDIT_FILE"] = str(args.output / "ec-audit.jsonl")\n',
    )
    source = replace_once(
        source, "        shutil.rmtree(cache)\n",
        '        if not shared_cache:\n            shutil.rmtree(cache)\n',
    )
    path.write_text(source, encoding="utf-8")
    return path


def count_audit(path, start, end):
    first_decisions = {}
    forwards = items = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        t = row["time_ns"] / 1e9
        stage = row["stage"]
        if stage == "visual_forward":
            if start <= t <= end:
                forwards += 1
                items += row["items"]
        else:
            first_decisions.setdefault((row["request"], row["image"]), (t, stage))
    counts = Counter(stage for t, stage in first_decisions.values() if start <= t <= end)
    total = counts["ec_hit"] + counts["ec_miss"] + counts["resident"]
    if not total:
        raise ValueError("No encoder decisions in measurement window")
    return {
        "visual_forwards": forwards,
        "visual_items": items,
        "first_image_decisions": total,
        "ec_hits": counts["ec_hit"],
        "ec_misses": counts["ec_miss"],
        "resident_hits": counts["resident"],
        "ec_hit_fraction": counts["ec_hit"] / total,
    }


def launch(args, config, replay, dataset, runner, label, rate, port, cache=None):
    waves = args.output / f"{label}-waves.json"
    capture = args.output / label
    write(waves, [{**replay, "dataset_path": str(dataset), "rate": rate, "name": "wave"}])
    env = dict(os.environ)
    env.pop("EPD_SHARED_EC_CACHE", None)
    if cache is not None:
        env["EPD_SHARED_EC_CACHE"] = str(cache)
    subprocess.run(
        [
            sys.executable, str(runner), "--repo", str(args.source),
            "--output", str(capture), "--waves", str(waves),
            "--encoders", "1", "--pd", "3", "--encoder-max-seqs", "1",
            "--model", config["deployment"]["model"], "--pd-kv-bytes",
            str(config["deployment"]["kv_cache_memory_bytes"]), "--pd-chunk",
            str(config["pd"]["chunk_tokens"]), "--base-port", str(port),
        ],
        env=env, check=True,
    )
    return capture


def run(args):
    replay = read(args.replay)
    config = read(args.calibration)
    args.output.mkdir(parents=True)
    dataset = resolve_cohort(replay, args.output)
    runner = runner_for(args)
    order = ("cold", "warm", "warm", "cold")
    write(args.output / "plan.json", {
        "rate_rps": 34.27, "prime_rps": 12, "gpus": 4, "split": [1, 3],
        "replay": replay, "order": order, "baseline_boundary_rps": 27.42584047132238,
        "runner_sha256": digest(runner), "calibration_sha256": digest(args.calibration),
        "audit_sha256": digest(Path(__file__)),
    })
    seed_cache = args.output / "prefilled-ec"
    prime = launch(args, config, replay, dataset, runner, "prime", 12, args.port, seed_cache)
    sample = read(prime / "wave.json")
    validate_corpus(sample, sample, replay["prompts"])
    prefilled_entries = sum(1 for _ in seed_cache.glob("*/encoder_cache.safetensors"))
    if not prefilled_entries:
        raise ValueError("No persisted embeddings after the prime")
    result = {
        "status": "running", "prefilled_entries": prefilled_entries, "runs": [],
        "prime": count_audit(prime / "ec-audit.jsonl", float("-inf"), float("inf")),
    }
    for index, condition in enumerate(order, 1):
        port = args.port + index * 5
        capture = launch(
            args, config, replay, dataset, runner, f"{index}-{condition}", 34.27, port,
            seed_cache if condition == "warm" else None,
        )
        bench = read(capture / "wave.json")
        validate_corpus(bench, sample, replay["prompts"])
        starts = bench["start_times"]
        first, last = min(starts), max(starts)
        begin, end = (first + p * (last - first) for p in (0.2, 0.8))
        path = capture / "ec-audit.jsonl"
        full_wave = count_audit(path, float("-inf"), float("inf"))
        if full_wave["first_image_decisions"] != replay["prompts"]:
            raise ValueError("Expected one first encoder decision per served image request")
        result["runs"].append({
            "condition": condition, "order": index, "completed": bench["completed"],
            "failed": bench["failed"], "p90_ttft_ms": bench["p90_ttft_ms"],
            "central": count_audit(path, begin, end),
            "full_wave": full_wave,
            **central_metrics(
                capture / "wave-metrics.jsonl", bench, read(capture / "wave-window.json"),
                1, 4, port,
            ),
        })
        write(args.output / "results.json", result)
    result["status"] = "completed"
    write(args.output / "results.json", result)
    shutil.rmtree(seed_cache)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay", type=Path, required=True)
    for name in ("source", "replay", "calibration", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--port", type=int, default=34000)
    args = parser.parse_args()
    instrument(args.overlay)
    run(args)
