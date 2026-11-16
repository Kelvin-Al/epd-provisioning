"""Run a frozen EPD sizing comparison against an equal split."""

import argparse
from collections import Counter
from hashlib import sha256
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys


CAPACITY_PATH = Path("examples/disaggregated/disaggregated_encoder/epd_capacity")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def execute(*args):
    subprocess.run(list(map(str, args)), check=True)


def custom_image_runner(original, destination):
    """Add a dataset option to the prepared upstream runner without editing it."""
    source = original.read_text(encoding="utf-8")
    start = source.index('                    "--dataset-name",\n')
    end = source.index('                    "--ignore-eos",\n', start)
    original_arguments = source[start:end]
    if '"random-mm"' not in original_arguments or '"--random-range-ratio"' not in original_arguments:
        raise ValueError("Unsupported vLLM benchmark argument layout")
    replacement = (
        '                    *(["--dataset-name", "custom_image",\n'
        '                       "--dataset-path", wave["dataset_path"],\n'
        '                       "--custom-output-len", str(wave["output_len"]),\n'
        '                       "--custom-ensure-client-side-data", "--no-oversample"]\n'
        '                      if "dataset_path" in wave else [\n'
        + original_arguments
        + '                      ]),\n'
    )
    script = source[:start] + replacement + source[end:]
    scrape = "                    log.flush()\n                    time.sleep(2)"
    if script.count(scrape) != 1:
        raise ValueError("Unsupported worker-scrape loop")
    script = script.replace(scrape, scrape.replace("sleep(2)", "sleep(1)"))
    compile(script, str(destination), "exec")
    destination.write_text(script, encoding="utf-8")
    return destination


def resolve_cohort(replay, output):
    source = Path(replay["dataset_path"]).resolve()
    if digest(source) != replay["cohort_sha256"]:
        raise ValueError("The MMMU request manifest changed")
    image_hash = sha256()
    rows = []
    for line in source.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        image, = (Path(name).resolve() for name in row["image_files"])
        if not image.is_relative_to(source.parent):
            raise ValueError("Cohort image is outside the cohort directory")
        image_hash.update(image.read_bytes())
        row["image_files"] = [str(image)]
        rows.append(json.dumps(row, ensure_ascii=False))
    if len(rows) != replay["prompts"] or image_hash.hexdigest() != replay["image_bytes_sha256"]:
        raise ValueError("The MMMU image cohort changed")
    path = output / "runtime-cohort.jsonl"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def benchmark(args, config, name, encoders, waves, base_port, trace=None):
    waves_path = args.output / f"{name}-waves.json"
    if args.runtime_dataset is not None:
        waves = [{**wave, "dataset_path": str(args.runtime_dataset)} for wave in waves]
    write(waves_path, waves)
    target = args.output / name
    execute(
        sys.executable, args.runner,
        "--repo", args.source, "--output", target, "--waves", waves_path,
        "--encoders", encoders, "--pd", args.gpus - encoders,
        "--encoder-max-seqs", 1, "--model", config["deployment"]["model"],
        "--pd-kv-bytes", config["deployment"]["kv_cache_memory_bytes"],
        "--pd-chunk", config["pd"]["chunk_tokens"], "--base-port", base_port,
        *(("--workload-trace", trace) if trace else ()),
    )
    return target


def validate_corpus(result, sample, prompts):
    if result["failed"] or result["completed"] != prompts:
        raise ValueError("Replay did not complete the entire corpus")
    for key in ("input_lens", "output_lens"):
        if len(result[key]) != prompts or result[key] != sample[key]:
            raise ValueError(f"Replay {key} does not match the sampled corpus in request order")


def central_metrics(path, bench, window, encoders, gpus, base_port):
    """Measure all workers over the same central portion of actual arrivals."""
    starts = bench["start_times"]
    first, last = min(starts), max(starts)
    if not window["begin_ns"] / 1e9 <= first < last <= window["end_ns"] / 1e9:
        raise ValueError("Client timestamps are outside the runner's monotonic window")
    start = first + 0.2 * (last - first)
    end = first + 0.8 * (last - first)
    ports = list(range(base_port + 1, base_port + gpus + 1))
    samples = {port: [] for port in ports}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        completed = waiting = 0.0
        for metric in row["metrics"]:
            if metric.startswith("vllm:request_success_total{") and any(
                f'finished_reason="{reason}"' in metric
                for reason in ("stop", "length", "repetition")
            ):
                completed += float(metric.rsplit(maxsplit=1)[-1])
            if metric.startswith("vllm:num_requests_waiting{"):
                waiting += float(metric.rsplit(maxsplit=1)[-1])
        samples[row["port"]].append((row["monotonic_ns"] / 1e9, completed, waiting))
    # Linear interpolation handles staggered scrapes without moving the window.
    def count_at(rows, when):
        for (t0, c0, _), (t1, c1, _) in zip(rows, rows[1:]):
            if t0 <= when <= t1:
                return c0 + (c1 - c0) * (when - t0) / (t1 - t0)
        raise ValueError("Worker counters do not bracket the central arrival window")

    rates, peaks = [], []
    for rows in samples.values():
        middle = [waiting for t, _, waiting in rows if start <= t <= end]
        if len(middle) < 3 or any(
            t1 <= t0 or c1 < c0 for (t0, c0, _), (t1, c1, _) in zip(rows, rows[1:])
        ):
            raise ValueError("Missing central scrapes or a worker counter reset")
        rates.append((count_at(rows, end) - count_at(rows, start)) / (end - start))
        peaks.append(max(middle))
    return {
        "actual_offer_rps": (len(starts) - 1) / (last - first),
        "central_seconds": end - start,
        "encoder_completed_rps": sum(rates[:encoders]),
        "pd_completed_rps": sum(rates[encoders:]),
        "encoder_peak_waiting_per_worker": max(peaks[:encoders]),
        "pd_peak_waiting_per_worker": max(peaks[encoders:]),
    }


def freeze(args, native, frozen, workload, replay, settings):
    estimate = native["workload_estimate"]
    input_hashes = {"model_sha256": digest(frozen), "workload_sha256": digest(workload)}
    if any(estimate[key] != value for key, value in input_hashes.items()):
        raise ValueError("vLLM recommendation used different frozen inputs")
    spec = importlib.util.spec_from_file_location("release_planner", args.planner)
    planner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(planner)
    even = args.gpus // 2
    e_rate = estimate["e_rate_upper_bound_rps"]
    pd_rate = estimate["pd_rate_rps"]
    decision = planner.provision(args.gpus, e_rate, pd_rate, incumbent=even)
    recommended = decision["encoder_gpus"]
    baseline_rps = min(even * e_rate, even * pd_rate)
    capacity_rps = decision["boundary_rps"]
    rates = {
        "low": round(0.75 * baseline_rps, 3),
        "between": round((baseline_rps + capacity_rps) / 2, 3),
        "overload": round(1.25 * capacity_rps, 3),
    }
    waves = [{**replay, "name": label, "rate": rate} for label, rate in rates.items()]
    minimum_seconds = 15 if "dataset_path" in replay else 24
    if min(rates.values()) <= 0 or replay["prompts"] / max(rates.values()) < minimum_seconds:
        raise ValueError("Planned arrival window is too short for worker counter sampling")
    counts = {"even": even, "method": recommended}
    orders = [["even", "method"], ["method", "even"]] if recommended != even else []
    if args.all_splits:
        counts = {f"{e}e": e for e in range(1, args.gpus)}
        order = list(counts)
        orders = [order, list(reversed(order))]
    return {
        **settings,
        "calibration_input_hashes": input_hashes,
        "public_planner_sha256": digest(args.planner),
        "native_plan_sha256": digest(args.output / "native-plan.json"),
        "rates_are_model_estimates": True,
        "stage_rate_inputs": estimate,
        "decision": decision,
        "encoder_counts": counts,
        "predicted_even_rps": baseline_rps,
        "waves": waves,
        "orders": orders,
    }


def compare(plan, observations):
    pairs = []
    for round_number in range(1, len(plan["orders"]) + 1):
        for wave in plan["waves"]:
            rows = {
                row["encoder_gpus"]: row["pd_completed_rps"] for row in observations
                if row["round"] == round_number and row["wave"] == wave["name"]
            }
            if set(rows) != set(plan["encoder_counts"].values()):
                continue
            recommended = plan["decision"]["encoder_gpus"]
            baseline = rows[plan["gpus"] // 2]
            observed = rows[recommended]
            best = max(rows.values())
            difference = observed - baseline
            pairs.append({
                "round": round_number,
                "wave": wave["name"],
                "offered_rps": wave["rate"],
                "pd_completion_difference_rps": difference,
                "pd_completion_difference_percent": 100 * difference / baseline if baseline else None,
                "recommended_encoder_gpus": recommended,
                "recommended_rank": 1 + sum(rate > observed for rate in rows.values()),
                "best_encoder_gpu_counts": [split for split, rate in rows.items() if rate == best],
                "gap_to_best_percent": 100 * (best - observed) / best if best else None,
            })
    return pairs


def run(args):
    replay = read(args.workload)
    config = read(args.calibration)
    args.output.mkdir(parents=True)
    frozen = args.output / "calibration.json"
    frozen.write_bytes(args.calibration.read_bytes())
    runner = args.source / "run_epd_superpod.py"
    args.runner = (
        custom_image_runner(runner, args.output / "custom-image-runner.py")
        if "dataset_path" in replay else runner
    )
    args.runtime_dataset = resolve_cohort(replay, args.output) if "dataset_path" in replay else None
    settings = {
        "gpus": args.gpus, "sampling_rps": args.sample_rate, "deployment": config["deployment"],
        "pd_chunk_tokens": config["pd"]["chunk_tokens"], "encoder_max_seqs": 1,
        "replay": replay, "runner_sha256": digest(args.runner),
        "upstream_runner_sha256": digest(runner),
        "method_runner_sha256": digest(Path(__file__)),
        "primary_metric": "PD successful completions per second in central 60% of actual arrivals",
    }
    write(args.output / "design.json", settings)
    trace = args.output / "sampled-requests.jsonl"
    even = args.gpus // 2
    sample_dir = benchmark(
        args, config, "sample", even,
        [{**replay, "name": "sample", "rate": args.sample_rate}], args.base_port, trace,
    )
    sample = read(sample_dir / "sample.json")
    validate_corpus(sample, sample, replay["prompts"])
    workload = args.output / "workload.json"
    execute(
        sys.executable, args.source / CAPACITY_PATH / "workload_trace.py",
        "--trace", trace, "--output", workload, "--min-completed", replay["prompts"],
    )
    observed = read(workload)
    if len(observed["encoder"]) != replay["prompts"] or any(
        Counter(observed["pd"][key]) != Counter(sample[key])
        for key in ("input_lens", "output_lens")
    ):
        raise ValueError("Proxy trace does not match native client usage")
    native_path = args.output / "native-plan.json"
    execute(
        "vllm", "bench", "epd-plan", "--num-gpus", args.gpus,
        "--frozen-model", frozen, "--workload", workload, "--output-json", native_path,
    )
    plan = freeze(args, read(native_path), frozen, workload, replay, settings)
    write(args.output / "plan.json", plan)
    observations = []
    summary = {"status": "same_split" if not plan["orders"] else "running", "runs": observations}
    write(args.output / "results.json", summary)
    launch_number = 0
    for round_number, order in enumerate(plan["orders"], 1):
        for arm in order:
            launch_number += 1
            encoders = plan["encoder_counts"][arm]
            port = args.base_port + launch_number * (args.gpus + 1)
            capture = benchmark(args, config, f"round-{round_number}-{arm}", encoders, plan["waves"], port)
            for wave in plan["waves"]:
                name = wave["name"]
                bench = read(capture / f"{name}.json")
                validate_corpus(bench, sample, replay["prompts"])
                observations.append({
                    "round": round_number, "arm": arm, "wave": name,
                    "encoder_gpus": encoders, "pd_gpus": args.gpus - encoders,
                    "offered_rps": wave["rate"], "completed": bench["completed"],
                    "failed": bench["failed"], "request_throughput_rps": bench["request_throughput"],
                    "p90_ttft_ms": bench["p90_ttft_ms"], "p90_tpot_ms": bench["p90_tpot_ms"],
                    **central_metrics(
                        capture / f"{name}-metrics.jsonl", bench,
                        read(capture / f"{name}-window.json"), encoders, args.gpus, port,
                    ),
                })
                summary["pairs"] = compare(plan, observations)
                write(args.output / "results.json", summary)
    if plan["orders"]:
        summary["status"] = "completed"
        write(args.output / "results.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "calibration", "planner", "workload", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--sample-rate", type=float, default=2)
    parser.add_argument("--base-port", type=int, default=22340)
    parser.add_argument("--all-splits", action="store_true")
    args = parser.parse_args()
    if args.gpus < 4 or args.gpus % 2:
        parser.error("An even GPU budget of at least four is required")
    if not math.isfinite(args.sample_rate) or args.sample_rate <= 0:
        parser.error("The sampling rate must be finite and positive")
    last_port = (2 * (args.gpus - 1) if args.all_splits else 4) * (args.gpus + 1) + args.gpus
    if not 1024 <= args.base_port <= 65535 - last_port:
        parser.error("Base port must leave room for all port blocks")
    for name in ("source", "calibration", "planner", "workload", "output"):
        setattr(args, name, getattr(args, name).resolve())
    run(args)


if __name__ == "__main__":
    main()
