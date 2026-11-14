"""CLI for EPD planning, encoder calibration, and an InferRSI benchmark."""

import argparse
import json
import sys
import unittest
from pathlib import Path

from calibration import fit_encoder, workload_rates
from planner import positive_rate, provision
from simulation import measure, poisson_trace


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def emit(data, output=None):
    rendered = json.dumps(data, indent=2, allow_nan=False) + "\n"
    if output is None:
        sys.stdout.write(rendered)
    else:
        output.write_text(rendered, encoding="utf-8")


def plan(args):
    inputs = None
    if args.encoder_model:
        if args.e_rate is not None or args.workload is None:
            raise ValueError("use --encoder-model and --workload together, without --e-rate")
        inputs = workload_rates(
            read_json(args.encoder_model), read_json(args.workload), args.pd_rate
        )
        e_rate = inputs["e_rate_upper_bound_rps"]
        pd_rate = inputs["pd_rate_rps"]
    else:
        if args.workload is not None or args.e_rate is None or args.pd_rate is None:
            raise ValueError("supply both --e-rate and --pd-rate, or a model and workload")
        e_rate, pd_rate = args.e_rate, args.pd_rate

    basis = "capacity_at_supplied_effective_rates"
    if inputs is not None:
        basis = "planning_proxy_using_encoder_resource_load_upper_bound"
    if args.ttft_rates:
        if args.rate_bounds:
            raise ValueError("select either supplied TTFT limits or worst-case capacity")
        if inputs is not None:
            raise ValueError("TTFT planning requires explicit effective --e-rate and --pd-rate")
        if any(
            positive_rate(limit, "TTFT rate") > positive_rate(capacity, "capacity rate")
            for limit, capacity in zip(args.ttft_rates, (e_rate, pd_rate))
        ):
            raise ValueError("a TTFT limit cannot exceed its supplied stage capacity rate")
        e_rate, pd_rate = args.ttft_rates
        basis = "boundary_at_supplied_ttft_limits"
    if args.rate_bounds:
        if inputs is not None:
            raise ValueError("worst-case planning requires explicit effective stage rates")
        basis = "worst_case_capacity_over_supplied_rate_rectangle"
    result = provision(
        args.gpus, e_rate, pd_rate, incumbent=args.incumbent, target=args.target_rps,
        rate_bounds=(args.rate_bounds[:2], args.rate_bounds[2:]) if args.rate_bounds else None,
    )
    result["basis"] = basis
    if inputs is not None:
        result["workload_estimate"] = inputs
    emit(result, args.output)


def simulate(args):
    planned = provision(args.gpus, args.e_rate, args.pd_rate)
    encoders = (
        args.gpus // 2 if args.policy == "equal" else planned["encoder_gpus"]
    )
    trace = poisson_trace(
        args.offered_rps, args.warmup + args.duration, args.e_rate, args.pd_rate, args.seed
    )
    result = measure(trace, args.gpus, encoders, args.warmup, args.duration)
    result.update({
        "scope": "synthetic_effective_fifo_tandem",
        "policy": args.policy,
        "encoder_gpus": encoders,
        "pd_gpus": args.gpus - encoders,
        "capacity_prediction_rps": min(
            encoders * args.e_rate, (args.gpus - encoders) * args.pd_rate
        ),
        "workload": {
            "offered_rps": args.offered_rps, "seed": args.seed, "warmup_s": args.warmup,
            "duration_s": args.duration, "e_rate_rps": args.e_rate, "pd_rate_rps": args.pd_rate,
        },
    })
    emit(result, args.output)


def add_rates(command, *, required):
    command.add_argument("--gpus", type=int, required=True)
    command.add_argument("--e-rate", type=float, required=required)
    command.add_argument("--pd-rate", type=float, required=required)
    command.add_argument("--output", type=Path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    planning = commands.add_parser("plan", help="compute an E:PD split")
    add_rates(planning, required=False)
    planning.add_argument("--encoder-model", type=Path)
    planning.add_argument("--workload", type=Path)
    planning.add_argument("--incumbent", type=int, help="current E GPU count, kept on a tie")
    planning.add_argument("--target-rps", type=float)
    planning.add_argument("--ttft-rates", type=float, nargs=2, metavar=("E", "PD"))
    planning.add_argument(
        "--rate-bounds", type=float, nargs=4, metavar=("E_LO", "E_HI", "PD_LO", "PD_HI")
    )
    calibration = commands.add_parser("calibrate", help="fit lane models from anchor summaries")
    calibration.add_argument("--input", required=True, type=Path)
    calibration.add_argument("--output", required=True, type=Path)
    simulation = commands.add_parser("simulate", help="write an InferRSI benchmark result")
    add_rates(simulation, required=True)
    simulation.add_argument("--policy", choices=("equal", "provisioned"), required=True)
    simulation.add_argument("--offered-rps", type=float, required=True)
    simulation.add_argument("--seed", type=int, required=True)
    simulation.add_argument("--warmup", type=float, default=30.0, help="seconds")
    simulation.add_argument("--duration", type=float, default=120.0, help="seconds")
    commands.add_parser("validate", help="run CPU correctness tests")
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            plan(args)
        elif args.command == "calibrate":
            try:
                fitted = fit_encoder(read_json(args.input))
            except ModuleNotFoundError as exc:
                if exc.name != "numpy":
                    raise
                raise ValueError(
                    "calibration requires NumPy (python -m pip install numpy)"
                ) from exc
            emit(fitted, args.output)
        elif args.command == "simulate":
            simulate(args)
        else:
            suite = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parent))
            result = unittest.TextTestRunner().run(suite)
            return 0 if result.wasSuccessful() else 1
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
