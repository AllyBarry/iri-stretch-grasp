#!/usr/bin/env python3
'''
Prints a simulation event log (events.jsonl), optionally filtered.

    log_view.py RUN_DIR_OR_FILE                       # everything
    log_view.py RUN --kind command result             # just requests and outcomes
    log_view.py RUN --source grasp --data             # grasps, with their full data
    log_view.py RUN --op 4                            # one operation: command, steps, result
    log_view.py RUN --grep feta
    log_view.py --latest                              # the most recent run
'''
import argparse
import json
import sys
from pathlib import Path

from stretch_sim.events import KINDS, read_jsonl
from stretch_sim.run_sim import default_log_root

_COLORS = {"command": "34", "result": "32", "step": "90", "state": "35", "violation": "31;1", "info": "0", "bt": "33", "belief": "36"}


def find_log(path: str = None, latest: bool = False) -> Path:
    if latest or path is None:
        runs = sorted(default_log_root().glob("stretch_sim-*/events.jsonl"))
        if not runs:
            sys.exit(f"No runs found in {default_log_root()}")
        return runs[-1]
    p = Path(path)
    return p / "events.jsonl" if p.is_dir() else p


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", nargs="?", help="A run directory or an events.jsonl file")
    parser.add_argument("--latest", action="store_true", help="Use the most recent run")
    parser.add_argument("--kind", nargs="+", choices=KINDS, help="Only these kinds of event")
    parser.add_argument("--source", nargs="+", help="Only these sources, e.g. navigate grasp detect")
    parser.add_argument("--op", type=int, help="Only events of this operation id")
    parser.add_argument("--grep", help="Only events whose text or data contains this (case-insensitive)")
    parser.add_argument("--data", action="store_true", help="Print each event's data too")
    parser.add_argument("--summary", action="store_true", help="Count events by kind and source")
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args(argv)

    path = find_log(args.log, args.latest)
    events = read_jsonl(path)
    if args.kind:
        events = [e for e in events if e.kind in args.kind]
    if args.source:
        events = [e for e in events if e.source in args.source]
    if args.op is not None:
        events = [e for e in events if e.op_id == args.op]
    if args.grep:
        q = args.grep.lower()
        events = [e for e in events if q in e.format().lower() or q in json.dumps(e.data).lower()]

    color = sys.stdout.isatty() and not args.no_color
    print(f"# {path} ({len(events)} events)")
    if args.summary:
        counts = {}
        for e in events:
            counts[(e.kind, e.source)] = counts.get((e.kind, e.source), 0) + 1
        for (kind, source), n in sorted(counts.items()):
            print(f"{kind:<10} {source:<16} {n}")
        return 0
    for e in events:
        line = e.format()
        print(f"\033[{_COLORS.get(e.kind, '0')}m{line}\033[0m" if color else line)
        if args.data and e.data:
            print("    " + json.dumps(e.data, indent=2).replace("\n", "\n    "))
    return 0


if __name__ == "__main__":
    sys.exit(main())
