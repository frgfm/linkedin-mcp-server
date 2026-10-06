#!/usr/bin/env python3
"""Compatibility entry point for the pre-rename policy-trace tests.

The canonical checker moved to ``check_policy_traces.py`` with the
``scraping/`` to ``linkedin/`` rename. Keep this shim until the downstream
tests no longer import the old path.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    module = importlib.import_module("tests.scraping.policy_scenarios")
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--output", type=Path)
    args = parser.parse_args()
    generated = asyncio.run(module.build_policy_traces())
    if args.output is not None:
        fixture_root = Path(module.TRACE_ROOT).resolve()
        if args.output.resolve().is_relative_to(fixture_root.parent):
            parser.error(
                "refusing to write generated output inside canonical fixture directory: "
                f"{args.output}"
            )
        if args.output.exists():
            parser.error(f"refusing to overwrite generated output: {args.output}")
        args.output.mkdir(parents=True)
        for name, trace in sorted(generated.items()):
            (args.output / name).write_text(
                module.canonical_json(trace), encoding="utf-8"
            )
        print(args.output)
        return 0
    difference = module.policy_trace_diff(generated)
    if difference:
        sys.stderr.write(difference)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
