"""Compare all four-GPU splits with per-request EC keys and an empty store."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

from audit import count_audit, instrument, replace_once, runner_for
from experiment import central_metrics, digest, read, resolve_cohort, validate_corpus, write


RATE = 34.27
ORDERS = ((1, 2, 3), (3, 2, 1))
PROXY_PATH = Path("examples/disaggregated/disaggregated_encoder/disagg_epd_proxy.py")


def make_runner(args):
    proxy = args.output / "cold-proxy.py"
    source = (args.source / PROXY_PATH).read_text(encoding="utf-8")
    source = replace_once(
        source, '    mm_items = extract_mm_items(orig_request)\n',
        '    mm_items = extract_mm_items(orig_request)\n'
        '    for idx, item in enumerate(mm_items):\n'
        '        item["uuid"] = hashlib.sha256(f"cold:{req_id}:{idx}".encode()).hexdigest()\n',
    )
    original_path = args.source / PROXY_PATH
    compile(source, str(original_path), "exec")
    proxy.write_text(
        f"import sys\nsys.path.insert(0, {str(original_path.parent)!r})\n"
        f"__file__ = {str(original_path)!r}\n"
        f"exec(compile({source!r}, __file__, 'exec'))\n", encoding="utf-8",
    )
    runner = runner_for(args)
    original = runner.read_text(encoding="utf-8")
    updated = replace_once(
        original,
        '    proxy = args.repo / (\n        "examples/disaggregated/disaggregated_encoder/disagg_epd_proxy.py"\n    )',
        f"    proxy = Path({str(proxy)!r})",
    )
    updated = replace_once(
        updated, 'str(args.output / "ec-audit.jsonl")',
        'str(args.output / f"ec-audit-{port}.jsonl")',
    )
    updated = replace_once(
        updated, '"--custom-ensure-client-side-data", "--no-oversample"]',
        '"--custom-ensure-client-side-data", "--no-oversample",\n'
        '                       "--num-prompts", str(prompts), "--seed", str(wave.get("seed", args.seed))]',
    )
    compile(updated, str(runner), "exec")
    runner.write_text(updated, encoding="utf-8")
    return runner, proxy


def launch(args, config, replay, dataset, runner, label, encoders, port):
    pd = 4 - encoders
    waves = args.output / f"{label}-waves.json"
    capture = args.output / label
    write(waves, [{**replay, "dataset_path": str(dataset), "rate": RATE, "name": "wave"}])
    env = dict(os.environ)
    env.pop("EPD_SHARED_EC_CACHE", None)
    subprocess.run(
        [
            sys.executable, str(runner), "--repo", str(args.source),
            "--output", str(capture), "--waves", str(waves),
            "--encoders", str(encoders), "--pd", str(pd), "--encoder-max-seqs", "1",
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
    runner, proxy = make_runner(args)
    write(args.output / "plan.json", {
        "offered_rps": RATE, "gpus": 4, "encoder_counts_by_round": ORDERS,
        "ec_policy": "fresh workers, empty EC directory, request-scoped media keys",
        "replay": replay, "deployment": config["deployment"],
        "pd_chunk_tokens": config["pd"]["chunk_tokens"],
        "calibration_sha256": digest(args.calibration),
        "runner_sha256": digest(runner), "proxy_sha256": digest(proxy),
        "audit_sha256": digest(Path(__file__).with_name("audit.py")),
        "sweep_sha256": digest(Path(__file__)),
    })
    results = {"status": "running", "runs": []}
    reference = None
    index = 0
    for round_num, counts in enumerate(ORDERS, 1):
        for encoders in counts:
            port = args.port + index * 5
            index += 1
            capture = launch(
                args, config, replay, dataset, runner,
                f"round-{round_num}-{encoders}e", encoders, port,
            )
            bench = read(capture / "wave.json")
            if reference is None:
                reference = bench
            validate_corpus(bench, reference, replay["prompts"])
            starts = bench["start_times"]
            first, last = min(starts), max(starts)
            begin, end = (first + p * (last - first) for p in (0.2, 0.8))
            events = capture / "ec-audit.jsonl"
            per_worker = sorted(capture.glob("ec-audit-*.jsonl"))
            if len(per_worker) != encoders:
                raise ValueError("Missing encoder audit streams")
            events.write_bytes(b"".join(path.read_bytes() for path in per_worker))
            full = count_audit(events, float("-inf"), float("inf"))
            if (
                full["first_image_decisions"] != replay["prompts"]
                or full["visual_items"] != replay["prompts"]
                or full["ec_hits"] or full["resident_hits"]
            ):
                raise ValueError(f"Wave did not encode every image without reuse: {full}")
            results["runs"].append({
                "round": round_num, "encoder_gpus": encoders, "pd_gpus": 4 - encoders,
                "completed": bench["completed"], "failed": bench["failed"],
                "p90_ttft_ms": bench["p90_ttft_ms"], "p90_tpot_ms": bench["p90_tpot_ms"],
                "central": count_audit(events, begin, end), "full_wave": full,
                **central_metrics(
                    capture / "wave-metrics.jsonl", bench, read(capture / "wave-window.json"),
                    encoders, 4, port,
                ),
            })
            write(args.output / "results.json", results)
    results["status"] = "completed"
    write(args.output / "results.json", results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("overlay", "source", "replay", "calibration", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    instrument(args.overlay)
    run(args)
