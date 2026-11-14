"""Replay an open-loop workload through dedicated E and PD endpoints."""

import asyncio
import json
import math
import random
import statistics
import time
import uuid

CHAT = "/v1/chat/completions"


async def request(session, encoder, decoder, model, row, max_tokens, request_id, cache):
    text, paths = row
    images = [
        {"type": "image_url", "image_url": {"url": path.as_uri()},
         "uuid": f"{request_id}-{index}"}
        for index, path in enumerate(paths)
    ]
    started = time.monotonic()
    async with session.post(encoder + CHAT, json={
        "model": model, "max_tokens": 1, "temperature": 0,
        "messages": [{"role": "user", "content": images}],
    }) as response:
        response.raise_for_status()
        await response.read()
    encoded = time.monotonic()
    for image in images:
        artifact = cache / image["uuid"] / "encoder_cache.safetensors"
        if not artifact.is_file():
            raise ValueError("encoder did not publish the expected cache entry")
    if decoder is None:
        return {"completed_at": encoded, "encoder_s": encoded - started}

    first = last = count = None
    finished = False
    async with session.post(decoder + CHAT, json={
        "model": model, "max_tokens": max_tokens, "temperature": 0,
        "stream": True, "stream_options": {"include_usage": True},
        "messages": [{"role": "user", "content": [*images, {"type": "text", "text": text}]}],
        "chat_template_kwargs": {"enable_thinking": False},
    }) as response:
        response.raise_for_status()
        async for line in response.content:
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                finished = True
                break
            event = json.loads(data)
            if event.get("error"):
                raise ValueError("generation stream reported an error")
            for choice in event.get("choices", []):
                delta = choice.get("delta") or {}
                if delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content"):
                    last = time.monotonic()
                    if first is None:
                        first = last
            if event.get("usage"):
                count = event["usage"]["completion_tokens"]
    if not finished or first is None or type(count) is not int or count < 1:
        raise ValueError("incomplete generation stream or missing token usage")
    return {
        "completed_at": time.monotonic(), "encoder_s": encoded - started, "tokens": count,
        "ttft_ms": (first - started) * 1000,
        "tpot_ms": (last - first) * 1000 / (count - 1) if count > 1 else None,
    }


def arrivals(rate, horizon, seed):
    rng = random.Random(seed)
    offset = rng.expovariate(rate)
    while offset < horizon:
        yield offset, rng.getrandbits(64), rng.getrandbits(64)
        offset += rng.expovariate(rate)


def summarize(rows, start, end, ttft_limit=None, tpot_limit=None):
    measured = [row for row in rows if start <= row["scheduled_at"] < end]
    completed = [row for row in measured if row["status"] == "completed"]
    completions = sum(
        row["status"] == "completed" and start <= row["completed_at"] < end for row in rows
    )
    result = {
        "offered": len(measured), "completed": len(completed),
        "failed": sum(row["status"] == "failed" for row in measured),
        "unfinished": sum(row["status"] == "unfinished" for row in measured),
        "client_dropped": sum(row["status"] == "client_dropped" for row in measured),
        "completions_in_window": completions, "throughput_rps": completions / (end - start),
    }
    for key in ("ttft_ms", "tpot_ms", "launch_delay_ms"):
        values = sorted(row[key] for row in completed if row.get(key) is not None)
        result[f"p90_{key}"] = values[math.ceil(len(values) * 0.9) - 1] if values else None
        result[f"p99_{key}"] = values[math.ceil(len(values) * 0.99) - 1] if values else None
    if ttft_limit is not None:
        qualified = sum(
            row.get("ttft_ms", math.inf) <= ttft_limit
            and row.get("tpot_ms") is not None and row["tpot_ms"] <= tpot_limit
            for row in completed
        )
        result["within_both_latency_limits"] = qualified
        result["within_both_fraction"] = qualified / len(measured) if measured else None
    return result


async def wave(session, encoders, decoders, workload, args, rate, seed, cache):
    from aiohttp import ClientError

    rows, active = [], set()
    namespace = uuid.uuid4().hex
    start = time.monotonic()
    cutoff = start + args.warmup + args.duration

    async def execute(row, index, e, pd):
        row["launch_delay_ms"] = max(0.0, (time.monotonic() - row["scheduled_at"]) * 1000)
        try:
            result = await request(
                session, e, pd, args.model, workload[index % len(workload)],
                args.max_tokens, f"{namespace}-{index}", cache,
            )
            if result["completed_at"] < cutoff:
                if "ttft_ms" in result:
                    result["ttft_ms"] += row["launch_delay_ms"]
                row.update(result, status="completed")
        except (OSError, ValueError, KeyError, ClientError, asyncio.TimeoutError) as exc:
            if time.monotonic() < cutoff:
                row.update(status="failed", error=type(exc).__name__)

    try:
        for index, (offset, e_pick, pd_pick) in enumerate(
            arrivals(rate, args.warmup + args.duration, seed)
        ):
            scheduled = start + offset
            await asyncio.sleep(max(0.0, scheduled - time.monotonic()))
            row = {"index": index, "scheduled_at": scheduled, "status": "unfinished"}
            rows.append(row)
            if len(active) >= args.max_inflight:
                row["status"] = "client_dropped"
                continue
            if time.monotonic() >= cutoff:
                continue
            e = encoders[e_pick % len(encoders)]
            pd = decoders[pd_pick % len(decoders)] if decoders else None
            task = asyncio.create_task(execute(row, index, e, pd))
            active.add(task)
            task.add_done_callback(active.discard)
        await asyncio.sleep(max(0.0, cutoff - time.monotonic()))
    finally:
        pending = list(active)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
    summary = summarize(rows, start + args.warmup, cutoff, args.ttft_ms, args.tpot_ms)
    for row in rows:
        row["scheduled_at"] -= start
        if "completed_at" in row:
            row["completed_at"] -= start
    return summary, rows


def compare(records):
    groups = {}
    for run in records:
        groups.setdefault((run["encoder_gpus"], run["offered_rps"]), []).append(run)
    return [
        {
            "encoder_gpus": encoders, "offered_rps": rate, "waves": len(group),
            "median_throughput_rps": statistics.median(row["throughput_rps"] for row in group),
            "client_dropped": sum(row["client_dropped"] for row in group),
        }
        for (encoders, rate), group in sorted(groups.items())
    ]
