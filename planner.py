"""Integer E:PD provisioning for constant effective per-instance rates."""

from fractions import Fraction
from math import ceil, floor, isfinite


def positive_rate(value, name):
    if (
        type(value) not in (int, float)
        or not isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be finite and positive")
    return Fraction(str(value))


def decision_interval(gpus, encoders):
    """Inclusive bounds for the E/PD rate ratio; None means no upper bound."""
    if type(gpus) is not int or type(encoders) is not int or not 1 <= encoders < gpus:
        raise ValueError("split must have at least one GPU per pool")
    lower = Fraction(gpus - encoders - 1, encoders)
    upper = None if encoders == 1 else Fraction(gpus - encoders, encoders - 1)
    return lower, upper


def provision(gpus, e_rate, pd_rate, *, incumbent=None, target=None, rate_bounds=None):
    """Optimize capacity, supplied TTFT limits, or the lower corner of a rate box.

    Rate bounds have the form ((E lower, E upper), (PD lower, PD upper)).
    Nominal rates are retained for diagnostics when optimizing the lower corner.
    """
    if type(gpus) is not int or gpus < 2:
        raise ValueError("gpus must be an integer >= 2")
    if incumbent is not None:
        decision_interval(gpus, incumbent)
    nominal_e = positive_rate(e_rate, "E rate")
    nominal_pd = positive_rate(pd_rate, "PD rate")
    e, pd = nominal_e, nominal_pd
    if rate_bounds is not None:
        (e_low, e_high), (pd_low, pd_high) = (
            tuple(positive_rate(value, "rate bound") for value in interval)
            for interval in rate_bounds
        )
        if not (e_low <= nominal_e <= e_high and pd_low <= nominal_pd <= pd_high):
            raise ValueError("each rate interval must contain its nominal rate")
        e, pd = e_low, pd_low

    crossing = gpus * pd / (e + pd)
    candidates = sorted(n for n in {floor(crossing), ceil(crossing)} if 1 <= n < gpus)
    capacities = {n: min(n * e, (gpus - n) * pd) for n in candidates}
    boundary = max(capacities.values())
    optima = [n for n, capacity in capacities.items() if capacity == boundary]
    selected = incumbent if incumbent in optima else optima[0]
    lower, upper = decision_interval(gpus, selected)
    ratio = e / pd
    margins = []
    if lower:
        margins.append(ratio / lower)
    if upper is not None:
        margins.append(upper / ratio)
    result = {
        "encoder_gpus": selected,
        "pd_gpus": gpus - selected,
        "optimal_encoder_gpu_counts": optima,
        "fractional_encoder_gpus": float(crossing),
        "fractional_boundary_rps": float(gpus * e * pd / (e + pd)),
        "boundary_rps": float(boundary),
        "nominal_boundary_at_selected_split_rps": float(
            min(selected * nominal_e, (gpus - selected) * nominal_pd)
        ),
        "stage_rates_rps": {"e": float(e), "pd": float(pd)},
        "rate_ratio": float(ratio),
        "optimal_ratio_interval": [float(lower), float(upper) if upper is not None else None],
        "multiplicative_ratio_margin": float(min(margins)) if margins else None,
        "bottleneck": (
            "balanced" if selected * e == (gpus - selected) * pd
            else "encoder" if selected * e < (gpus - selected) * pd
            else "pd"
        ),
    }
    if target is not None:
        load = positive_rate(target, "target")
        result["target"] = {
            "offered_rps": float(load),
            "strictly_below_boundary": load < boundary,
            "minimum_gpus_for_strict_target": floor(load / e) + floor(load / pd) + 2,
        }
    if rate_bounds is not None:
        low_ratio, high_ratio = e_low / pd_high, e_high / pd_low
        result["uncertainty"] = {
            "ratio_range": [float(low_ratio), float(high_ratio)],
            "selected_split_optimal_throughout": (
                low_ratio >= lower and (upper is None or high_ratio <= upper)
            ),
        }
    return result
