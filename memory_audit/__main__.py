"""Command-line interface: python -m memory_audit <command> [options].

  plan              show how many cases and model calls a config implies
  doctor            check dependencies and that the model server is reachable
  run               run (or resume) every case of a config
  ablate            re-answer a finished run under each other reasoning mode
  collect           merge a model's shards into one run with a report
  compare           compare collected model runs on their common eligible facts
  collect-ablation  compare reasoning modes on their common eligible cases
  report            rebuild summary.json/.csv from a run's observations.json
"""
import argparse
import json
from pathlib import Path

from .config import expected_rows, load_config
from .data import population


def plan(config):
    facts = len(population(config["facts"], config["seeds"][0], config["public_controls"]))
    cases = facts * len(config["seeds"]) * len(config["backends"])
    states = expected_rows(config)
    return {"cases": cases, "evaluated_states_per_case": states,
            "attack_and_utility_calls": cases * states * (max(config["budgets"]) + 1),
            "additional_memory_and_controller_calls": "variable",
            "fixture": config["backends"] == ["fixture"], "model": config["llm"].get("model")}


def build_parser():
    parser = argparse.ArgumentParser(prog="python -m memory_audit",
                                     description="Audit whether deleting LLM-agent memory removes the information.")
    commands = parser.add_subparsers(dest="command", required=True)

    def command(name, help_text):
        return commands.add_parser(name, help=help_text, description=help_text)

    sub = command("plan", "Show how many cases and model calls a config implies.")
    sub.add_argument("--config", required=True)
    sub = command("doctor", "Check dependencies and that the model server lists the configured model.")
    sub.add_argument("--config", required=True)
    sub.add_argument("--offline", action="store_true", help="check dependencies only")
    sub = command("run", "Run (or resume) every case of a config.")
    sub.add_argument("--config", required=True)
    sub.add_argument("--output", required=True, help="output directory, e.g. runs/gemma/shards/0")
    sub.add_argument("--workers", type=int, default=1, help="cases run in parallel (match the server's max sequences)")
    sub = command("ablate", "Re-answer a finished run under each other reasoning mode.")
    sub.add_argument("--baseline", required=True, help="finished run (one shard) to replay")
    sub.add_argument("--output", required=True)
    sub.add_argument("--workers", type=int, default=1)
    sub = command("collect", "Merge a model's shards (<output>/shards/*) into one run with a report.")
    sub.add_argument("--config", required=True, help="the model's full config, e.g. configs/full-runs/gemma.json")
    sub.add_argument("--output", required=True)
    sub = command("compare", "Compare collected model runs on the facts valid for every model and backend.")
    sub.add_argument("--run", nargs=2, action="append", required=True, metavar=("NAME", "PATH"))
    sub.add_argument("--output", required=True)
    sub = command("collect-ablation", "Compare reasoning modes on the cases valid in every mode.")
    sub.add_argument("--config", required=True, help="e.g. configs/ablation/gemma.json")
    sub.add_argument("--baseline", required=True, help="the model's sharded main run")
    sub.add_argument("--output", required=True, help="the ablation run written by `ablate`")
    sub = command("report", "Rebuild summary.json/.csv from a run's observations.json.")
    sub.add_argument("--output", required=True)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.command == "plan":
            print(json.dumps(plan(load_config(args.config)), indent=2))
        elif args.command == "doctor":
            from .doctor import check
            print(json.dumps(check(load_config(args.config), offline=args.offline), indent=2))
        elif args.command == "run":
            from .runner import run
            run(load_config(args.config), args.output, args.workers)
        elif args.command == "ablate":
            from .runner import ablate
            ablate(args.baseline, args.output, args.workers)
        elif args.command == "collect":
            from .aggregate import collect
            collect(load_config(args.config), args.output)
        elif args.command == "compare":
            runs = dict(args.run)
            if len(runs) != len(args.run) or len(runs) < 2:
                parser.error("Provide at least two uniquely named model runs")
            from .aggregate import compare
            compare(runs, args.output)
        elif args.command == "collect-ablation":
            from .aggregate import collect_ablation
            collect_ablation(load_config(args.config), args.baseline, args.output)
        elif args.command == "report":
            from .report import write_report
            root = Path(args.output)
            config = json.loads((root / "manifest.json").read_text())["config"]
            write_report(root, json.loads((root / "observations.json").read_text()), config)
    except (ValueError, RuntimeError, ImportError, OSError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
