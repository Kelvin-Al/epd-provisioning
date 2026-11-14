"""Fit encoder occupancy curves from caller-supplied measurement summaries."""

from math import isfinite
from statistics import fmean

SERIAL_DEGREES = {"H": 1, "M": 1, "C": 1, "X": 2, "G": 2, "R": 1}
SATURATED_DEGREES = {"H": 0, "M": 1, "C": 0, "X": 2, "G": 2, "R": 0}


def token_count(value):
    if type(value) is not int or value < 0:
        raise ValueError("visual_tokens must be a nonnegative integer")
    return value


def milliseconds(value):
    if type(value) not in (int, float) or not isfinite(value) or value < 0:
        raise ValueError("duration must be finite, nonnegative and in milliseconds")
    return value


def fit_encoder(data):
    """Fit separate strict-serial spans and saturated active resource times."""
    from numpy.polynomial import Polynomial

    anchors = data["anchors"]
    tokens = [token_count(row["visual_tokens"]) for row in anchors]
    if len(set(tokens)) < 3 or len(set(tokens)) != len(tokens):
        raise ValueError("supply one row per anchor, with at least three distinct token counts")
    parallelism = {
        lane: milliseconds(data["parallelism"][lane]) for lane in SATURATED_DEGREES
    }
    if not all(parallelism.values()):
        raise ValueError("all effective lane parallelisms must be positive")
    for row in anchors:
        for key in ("serial_ms", "saturated_ms"):
            spans = {lane: milliseconds(row[key][lane]) for lane in SERIAL_DEGREES}
            if spans["G"] > spans["X"]:
                raise ValueError(f"{key}: the GPU interval must fit inside the worker interval")

    def fit(key, degrees):
        return {
            lane: Polynomial.fit(
                tokens, [row[key][lane] for row in anchors], degree
            ).convert().coef.tolist()
            for lane, degree in degrees.items()
        }

    return {
        "visual_token_range": [min(tokens), max(tokens)],
        "serial_ms": fit("serial_ms", SERIAL_DEGREES),
        "saturated_ms": fit("saturated_ms", SATURATED_DEGREES),
        "parallelism": parallelism,
    }


def encoder_demands(model, visual_tokens):
    """Predict disjoint serial elapsed time and normalized resource demands."""
    visual_tokens = token_count(visual_tokens)
    lo, hi = model["visual_token_range"]
    if not lo <= visual_tokens <= hi:
        raise ValueError(f"visual_tokens={visual_tokens} outside calibrated range [{lo}, {hi}]")

    def predict(key, lane):
        coeffs = model[key][lane]
        value = 0.0
        for coefficient in reversed(coeffs):
            value = value * visual_tokens + coefficient
        # Least-squares fits can produce negative demands. Require recalibration
        # instead of silently treating a negative fit as a zero-time resource.
        return milliseconds(value)

    serial = {lane: predict("serial_ms", lane) for lane in SERIAL_DEGREES}
    active = {lane: predict("saturated_ms", lane) for lane in SATURATED_DEGREES}
    if any(
        times["G"] > times["X"] + 1e-9 * max(1.0, times["X"])
        for times in (serial, active)
    ):
        raise ValueError("fitted GPU interval exceeds fitted worker cycle")
    return (
        sum(serial[lane] for lane in ("H", "M", "C", "X", "R")),
        {lane: active[lane] / model["parallelism"][lane] for lane in active},
    )


def workload_rates(model, requests, pd_rate=None):
    """Average each resource before taking the workload-level bottleneck."""
    if not requests:
        raise ValueError("workload must contain at least one request")
    profiles = [
        encoder_demands(model, request["visual_tokens"]) for request in requests
    ]
    means = {lane: fmean(demands[lane] for _, demands in profiles) for lane in SATURATED_DEGREES}
    bottleneck = max(means, key=means.__getitem__)
    if means[bottleneck] <= 0:
        raise ValueError("mean encoder bottleneck demand must be positive")
    if pd_rate is None:
        pd_ms = fmean(milliseconds(request["pd_demand_ms"]) for request in requests)
        if pd_ms <= 0:
            raise ValueError("mean PD demand must be positive")
        pd_rate = 1000 / pd_ms
    return {
        "e_rate_upper_bound_rps": 1000 / means[bottleneck],
        "pd_rate_rps": pd_rate,
        "encoder_bottleneck_lane": bottleneck,
        "mean_normalized_lane_ms": means,
        "mean_serial_e_ms": fmean(serial for serial, _ in profiles),
        "request_count": len(requests),
    }
