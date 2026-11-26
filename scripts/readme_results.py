"""Write the README's best-pipeline section from `results/best-pipeline.json`.

Usage: `python3 scripts/readme_results.py` to update it, `--check` to only compare (exit 1 if stale).
"""

from __future__ import annotations

import csv
import json
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
START, END = "<!-- best-pipeline:start -->", "<!-- best-pipeline:end -->"
REQUIRED = ("task", "status", "date", "model", "variables", "runs", "metric", "steps", "source",
            "best", "before", "why", "caveat", "code")


def lower(text: str) -> str:
    return text[:1].lower() + text[1:]


def render() -> str:
    best = json.loads((ROOT / "results" / "best-pipeline.json").read_text(encoding="utf-8"))
    missing = [key for key in REQUIRED if key not in best]
    if missing:
        raise SystemExit(f"results/best-pipeline.json lacks {missing}")
    with (ROOT / best["source"]).open(encoding="utf-8") as handle:
        rows = {row["setting"]: row for row in csv.DictReader(handle)}
    new, old = rows[best["best"]], rows[best["before"]]
    nodes = [f'  s{i}["{"<br/>".join(textwrap.wrap(step, 28))}"]' for i, step in enumerate(best["steps"], 1)]
    chain = " --> ".join(f"s{i}" for i in range(1, len(best["steps"]) + 1))
    gain, low, high = (float(new[key]) for key in ("vs_fixed_gain", "vs_fixed_low", "vs_fixed_high"))
    note = (f"Gain {gain:+.2f} (95% interval {low:+.2f} to {high:+.2f}) on {best['variables']} variables, "
            f"same model (`{best['model']}`) and scorer, {best['runs']} runs each. {best['why']} "
            f"{best['status'].capitalize()}: {lower(best['caveat'])} "
            f"Data: [`{best['source']}`]({best['source']}), code: [`{best['code']}`]({best['code']}).")
    return "\n".join([
        START, "", "```mermaid", "flowchart LR", *nodes, f"  {chain}", "```", "",
        f"| Pipeline | {best['metric']} |", "|---|---:|",
        f"| Before: {lower(old['name'])} | {float(old['close_f1']):.2f} |",
        f"| **Best so far: {lower(new['name'])}** | **{float(new['close_f1']):.2f}** |", "",
        note, "", END])


def main() -> int:
    text = README.read_text(encoding="utf-8")
    head, rest = text.split(START, 1)
    current, tail = rest.split(END, 1)
    fresh = render()
    if "--check" in sys.argv:
        if START + current + END != fresh:
            print("README.md best-pipeline section is stale: run python3 scripts/readme_results.py")
            return 1
        return 0
    README.write_text(head + fresh + tail, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
