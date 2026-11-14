"""Finite-horizon CPU simulation of the paper's effective FIFO tandem model."""

import math
import random
from bisect import bisect_left
from dataclasses import dataclass

from planner import decision_interval, positive_rate


@dataclass(frozen=True)
class Request:
    arrival: float
    e_seconds: float
    pd_seconds: float
    e_route: float
    pd_route: float


def poisson_trace(offered_rps, horizon, e_rate, pd_rate, seed):
    """Use constant services and common arrival/routing randomness across splits."""
    positive_rate(offered_rps, "offered rate")
    positive_rate(e_rate, "E rate")
    positive_rate(pd_rate, "PD rate")
    if not math.isfinite(horizon) or horizon <= 0:
        raise ValueError("simulation horizon must be finite and positive")
    arrivals = random.Random(seed)
    routing = random.Random(seed ^ 0x9E3779B9)
    time = arrivals.expovariate(offered_rps)
    trace = []
    while time < horizon:
        trace.append(Request(time, 1 / e_rate, 1 / pd_rate, routing.random(), routing.random()))
        time += arrivals.expovariate(offered_rps)
    return trace


def departures(trace, gpus, encoders):
    """Return E and PD completion times in request order.

    PD receives jobs in E completion order. Carry each job's original service
    requirements and independently sampled destination marks into that order.
    """
    decision_interval(gpus, encoders)
    e_ready = [0.0] * encoders
    e_done = []
    for request in trace:
        worker = int(request.e_route * encoders)
        e_ready[worker] = max(request.arrival, e_ready[worker]) + request.e_seconds
        e_done.append(e_ready[worker])

    pd_count = gpus - encoders
    pd_ready = [0.0] * pd_count
    pd_done = [0.0] * len(trace)
    for i in sorted(range(len(trace)), key=e_done.__getitem__):
        worker = int(trace[i].pd_route * pd_count)
        pd_ready[worker] = max(e_done[i], pd_ready[worker]) + trace[i].pd_seconds
        pd_done[i] = pd_ready[worker]
    return e_done, pd_done


def measure(trace, gpus, encoders, warmup, duration):
    """Count arrivals and completions in [warmup, warmup + duration)."""
    if not math.isfinite(warmup) or warmup < 0:
        raise ValueError("warmup must be finite and nonnegative")
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be finite and positive")
    end = warmup + duration
    if not math.isfinite(end) or end <= warmup:
        raise ValueError("measurement interval cannot be represented")
    e_done, pd_done = departures(trace, gpus, encoders)
    arrivals = [request.arrival for request in trace]
    e_sorted, pd_sorted = sorted(e_done), sorted(pd_done)
    start_a, end_a = (bisect_left(arrivals, t) for t in (warmup, end))
    start_e, end_e = (bisect_left(e_sorted, t) for t in (warmup, end))
    start_pd, end_pd = (bisect_left(pd_sorted, t) for t in (warmup, end))
    return {
        "metrics": {"throughput_rps": (end_pd - start_pd) / duration},
        "counts": {
            "arrivals": end_a - start_a,
            "e_completions": end_e - start_e,
            "pd_completions": end_pd - start_pd,
            "unfinished_at_start": start_a - start_pd,
            "unfinished_at_end": end_a - end_pd,
            "e_unfinished_at_end": end_a - end_e,
            "pd_unfinished_at_end": end_e - end_pd,
        },
        "checks": {
            "valid_simulation": (
                bool(trace) and end_pd > start_pd
                and all(
                    math.isfinite(pd) and request.arrival <= e <= pd
                    for request, e, pd in zip(trace, e_done, pd_done)
                )
                and start_pd <= start_e <= start_a and end_pd <= end_e <= end_a
            )
        },
    }
