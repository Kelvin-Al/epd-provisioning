"""Deploy and benchmark vLLM E+PD on one allocated NVIDIA Linux node."""

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
from contextlib import contextmanager, suppress
from pathlib import Path

from cluster_data import DATASETS, load_workload, prepare
from cluster_load import compare, request, wave
from planner import provision

VLLM_VERSION = "0.22.0"
HOST = "127.0.0.1"


def positive(value):
    number = float(value)
    if number <= 0 or not math.isfinite(number):
        raise argparse.ArgumentTypeError("expected a positive finite number")
    return number


def count(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def devices(raw, visible):
    chosen = [part.strip() for part in raw.split(",")]
    if len(chosen) != len(set(chosen)) or any(
        not re.fullmatch(r"\d+|GPU-[0-9a-fA-F-]+|MIG-[0-9a-fA-F-]+", part)
        for part in chosen
    ):
        raise ValueError("GPU list must contain distinct CUDA ordinals or full GPU/MIG UUIDs")
    if visible is not None and not set(chosen) <= {part.strip() for part in visible.split(",")}:
        raise ValueError("GPU selection must stay within the inherited CUDA_VISIBLE_DEVICES")
    return chosen


def worker_command(executable, args, port, cache, producer):
    role = "ec_producer" if producer else "ec_consumer"
    config = {
        "ec_connector": "ECExampleConnector", "ec_role": role,
        "ec_connector_extra_config": {"shared_storage_path": str(cache)},
    }
    command = [
        executable, "serve", args.model, "--host", HOST, "--port", str(port),
        "--enforce-eager", "--no-enable-prefix-caching", "--max-model-len", str(args.context),
        "--allowed-local-media-path", str(args.workload.resolve()), "--ec-transfer-config",
        json.dumps(config), "--limit-mm-per-prompt", json.dumps({"image": args.max_images}),
        "--gpu-memory-utilization", str(args.e_memory if producer else args.pd_memory),
        "--max-num-seqs", str(args.max_seqs), "--no-enable-log-requests",
    ]
    if producer:
        command.extend(("--max-num-batched-tokens", str(args.e_batch_tokens)))
        if args.encoder_only:
            command.append("--mm-encoder-only")
    if args.model_revision:
        command.extend(("--revision", args.model_revision))
    return command


@contextmanager
def workers(executable, args, gpus, encoders, cache, directory):
    processes = []
    endpoints = []
    try:
        for index, gpu in enumerate(gpus):
            port = args.base_port + index
            with socket.socket() as probe:
                probe.bind((HOST, port))
            with (directory / f"worker-{index:02d}.log").open("wb") as log:
                process = subprocess.Popen(
                    worker_command(executable, args, port, cache, index < encoders),
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, VLLM_NO_USAGE_STATS="1"),
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            processes.append(process)
            endpoints.append(f"http://{HOST}:{port}")
        yield endpoints[:encoders], endpoints[encoders:], processes
    finally:
        for process in processes:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
        for process in processes:
            with suppress(subprocess.TimeoutExpired):
                process.wait(timeout=5)
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)


async def ready(session, endpoints, processes, timeout):
    from aiohttp import ClientError

    pending = set(endpoints)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while pending and loop.time() < deadline:
        for index, process in enumerate(processes):
            if process.poll() is not None:
                raise RuntimeError(f"worker {index} exited ({process.returncode}); see worker log")
        for url in list(pending):
            try:
                async with session.get(url + "/health", timeout=2) as response:
                    if response.status == 200:
                        pending.remove(url)
            except (ClientError, asyncio.TimeoutError):
                continue
        if pending:
            await asyncio.sleep(0.5)
    if pending:
        raise RuntimeError(f"{len(pending)} workers did not become healthy; see worker logs")


async def run_wave(executable, args, gpus, encoders, workload, directory, rate, seed):
    from aiohttp import ClientSession, ClientTimeout, TCPConnector

    with tempfile.TemporaryDirectory(prefix="epd-ec-", dir=args.cache_root) as temporary:
        cache = Path(temporary).resolve()
        with workers(executable, args, gpus, encoders, cache, directory) as deployment:
            e, pd, processes = deployment
            async with ClientSession(
                timeout=ClientTimeout(total=args.request_timeout),
                connector=TCPConnector(limit=0),
            ) as session:
                await ready(session, e + pd, processes, args.startup_timeout)
                # Confirm each worker participates and the producer creates connector data.
                for i in range(max(len(e), len(pd))):
                    await request(
                        session, e[i % len(e)], pd[i % len(pd)] if pd else None,
                        args.model, workload[0], min(args.max_tokens, 4), f"smoke-{i}", cache,
                    )
                if rate is None:
                    return {"health": "ok", "encoder_cache_written": True}
                summary, rows = await wave(
                    session, e, pd or None, workload, args, rate, seed, cache
                )
                if any(process.poll() is not None for process in processes):
                    raise RuntimeError("a worker exited during measurement; see worker logs")
                with (directory / "requests.jsonl").open("w", encoding="utf-8") as stream:
                    for row in rows:
                        stream.write(json.dumps(row, allow_nan=False) + "\n")
                return summary


def run(args):
    if sys.platform != "linux":
        raise ValueError("deployment requires NVIDIA Linux")
    visible = os.getenv("CUDA_VISIBLE_DEVICES")
    if not args.gpus and visible is None:
        raise ValueError("set --gpus or CUDA_VISIBLE_DEVICES to the GPUs allocated to this job")
    gpus = devices(args.gpus or visible, visible)
    if len(gpus) < (1 if args.e_only else 2) or args.base_port + len(gpus) - 1 > 65535:
        raise ValueError("allocate at least two GPUs and a valid consecutive port range")
    if (args.ttft_ms is None) != (args.tpot_ms is None):
        raise ValueError("supply both latency limits together")
    if (args.e_rate is None) != (args.pd_rate is None) or (args.e_rate and args.splits):
        raise ValueError("use both stage rates, or give explicit splits")
    if args.e_only and (args.splits or args.e_rate):
        raise ValueError("encoder-only measurement uses the first allocated GPU")
    if args.e_memory > 1 or args.pd_memory > 1:
        raise ValueError("GPU memory utilization must be at most 1")
    if args.command == "bench" and not args.rps:
        raise ValueError("bench needs at least one --rps")
    executable = shutil.which("vllm")
    if not executable or importlib.metadata.version("vllm") != VLLM_VERSION:
        raise ValueError(f"activate the environment containing vLLM {VLLM_VERSION}")
    workload = load_workload(args.workload)
    if any(len(images) > args.max_images for _, images in workload):
        raise ValueError("raise --max-images or prepare a workload with fewer images per request")
    plan = provision(len(gpus), args.e_rate, args.pd_rate) if args.e_rate else None
    splits = args.splits or (
        [len(gpus) // 2, plan["encoder_gpus"]] if plan else [1]
    )
    if not args.e_only and any(not 0 < n < len(gpus) for n in splits):
        raise ValueError("each split must leave at least one GPU for both E and PD")
    if args.e_only:
        gpus = gpus[:1]
    args.output.mkdir()
    identity = hashlib.sha256((args.workload / "workload.json").read_bytes())
    for path in sorted({p for _, images in workload for p in images}):
        identity.update(hashlib.sha256(path.read_bytes()).digest())
    write_json(args.output / "plan.json", {
        "workload_sha256": identity.hexdigest(), "model": args.model,
        "model_revision": args.model_revision, "vllm": VLLM_VERSION, "gpus": len(gpus),
        "splits": list(dict.fromkeys(splits)), "provision": plan, "rps": args.rps,
        "seed": args.seed, "repetitions": args.repetitions, "warmup_s": args.warmup,
        "duration_s": args.duration, "max_tokens": args.max_tokens, "e_only": args.e_only,
        "ttft_ms": args.ttft_ms, "tpot_ms": args.tpot_ms, "max_inflight": args.max_inflight,
        "context": args.context, "e_memory": args.e_memory, "pd_memory": args.pd_memory,
        "e_batch_tokens": args.e_batch_tokens, "max_seqs": args.max_seqs,
        "encoder_only_flag": args.encoder_only, "request_timeout_s": args.request_timeout,
    })
    records = []
    repeats = args.repetitions if args.command == "bench" else 1
    for repetition in range(repeats):
        order = splits if repetition % 2 == 0 else list(reversed(splits))
        for encoders in dict.fromkeys(order):
            for rate in args.rps or [None]:
                directory = args.output / f"wave-{len(records) + 1:03d}"
                directory.mkdir()
                result = asyncio.run(
                    run_wave(
                        executable, args, gpus, encoders, workload, directory,
                        rate, args.seed + repetition,
                    )
                )
                result.update(
                    encoder_gpus=encoders, pd_gpus=len(gpus) - encoders,
                    offered_rps=rate, seed=args.seed + repetition,
                )
                records.append(result)
                write_json(directory / "summary.json", result)
                write_json(args.output / "summary.json", {
                    "waves": records, "comparisons": compare(records) if rate else [],
                })
                print(json.dumps(result, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    data = commands.add_parser("prepare", help="materialize image requests and dataset revision")
    data.add_argument("--dataset", choices=DATASETS, default="mmmu")
    data.add_argument("--revision")
    data.add_argument("--split")
    data.add_argument("--subset", action="append")
    data.add_argument("--samples", type=count, default=200)
    data.add_argument("--shuffle-buffer", type=count, default=128)
    data.add_argument("--max-pixels", type=count, default=2073600)
    data.add_argument("--max-images", type=count, default=5)
    data.add_argument("--seed", type=int, default=17)
    data.add_argument("--output", type=Path, required=True)
    for name in ("smoke", "bench"):
        command = commands.add_parser(name)
        command.add_argument("--workload", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
        command.add_argument("--model-revision", help="model commit for reproducible comparisons")
        command.add_argument("--gpus", help="comma-separated, defaults to CUDA_VISIBLE_DEVICES")
        command.add_argument("--split", dest="splits", action="append", type=count)
        command.add_argument("--e-rate", type=positive)
        command.add_argument("--pd-rate", type=positive)
        command.add_argument("--e-only", action="store_true", help="measure encoder with one GPU")
        command.add_argument("--encoder-only", action="store_true", help="enable mm-encoder-only")
        command.add_argument("--rps", action="append", type=positive)
        command.add_argument("--warmup", type=positive, default=15)
        command.add_argument("--duration", type=positive, default=60)
        command.add_argument("--repetitions", type=count, default=1)
        command.add_argument("--max-inflight", type=count, default=1024)
        command.add_argument("--max-tokens", type=count, default=32)
        command.add_argument("--max-images", type=count, default=5)
        command.add_argument("--max-seqs", type=count, default=128)
        command.add_argument("--e-memory", type=positive, default=0.01)
        command.add_argument("--pd-memory", type=positive, default=0.8)
        command.add_argument("--context", type=count, default=8192)
        command.add_argument("--e-batch-tokens", type=count, default=114688)
        command.add_argument("--base-port", type=count, default=18080)
        command.add_argument("--startup-timeout", type=positive, default=900)
        command.add_argument("--request-timeout", type=positive, default=120)
        command.add_argument("--cache-root", type=Path, default=Path(tempfile.gettempdir()))
        command.add_argument("--ttft-ms", type=positive)
        command.add_argument("--tpot-ms", type=positive)
        command.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: signal.raise_signal(signal.SIGINT))
    try:
        if args.command == "prepare":
            from httpx import HTTPError

            try:
                prepare(args)
            except HTTPError as exc:
                parser.exit(2, f"Dataset download failed ({type(exc).__name__})\n")
        else:
            from aiohttp import ClientError

            if args.command == "smoke" and args.rps:
                raise ValueError("use bench to submit a load wave")
            try:
                run(args)
            except ClientError as exc:
                parser.exit(2, f"Worker request failed ({type(exc).__name__}); see worker logs\n")
    except (OSError, ValueError, KeyError, RuntimeError, ImportError) as exc:
        parser.exit(2, f"{type(exc).__name__}: {exc}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Interrupted\n")


if __name__ == "__main__":
    main()
