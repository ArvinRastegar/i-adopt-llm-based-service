"""Build figure.html (the chart, a table twin and the notes) from numbers.json.

One row per model, one panel per measure, one colour per model in every panel, every bar
starting at zero, each measure on its own axis (never two scales on one plot).

Run after compute_numbers.py:  python3 render_figure.py
Open figure.html in a browser. figure.html?png&theme=light hides the tables and forces the light
theme, for a clean screenshot (the theme otherwise follows the system setting).
"""

from __future__ import annotations

import html
import json
from collections.abc import Callable
from pathlib import Path
from string import Template

HERE = Path(__file__).resolve().parent

# key, short name, where it ran (row labels)
MODELS = (
    ("reranker", "Reranker", "0.6B model · Mac GPU"),
    ("qwen38", "Qwen3.8-27B", "27B chat model · PSNC"),
    ("jev", "Jev", "Jev 1.13 · OpenRouter"),
)

WIDTH, HEIGHT = 1180, 324
LABEL_X = 24
PANEL_X = (222, 542, 862)
BAR_W = 200
VALUE_DX = 214
PANEL_W = 296
ROW_Y = (102, 162, 222)
BAR_H = 22
GRID_TOP = 72
AXIS_Y = 254

PANELS = (
    {"title": "Accuracy", "hint": "higher is better", "vmax": 1.0,
     "desc": "share of the 284 slots linked correctly",
     "ticks": ((0, "0%"), (0.25, "25%"), (0.5, "50%"), (0.75, "75%"), (1.0, "100%")),
     "key": ("bar: main run, same 284 slots", "line: 95% margin of error")},
    {"title": "Cost", "hint": "lower is better", "vmax": 0.15,
     "desc": "to link the whole list once",
     "ticks": ((0, "$0"), (0.05, "$0.05"), (0.10, "$0.10"), (0.15, "$0.15")),
     "key": ("bar: median of the 15 OpenRouter providers", "line: cheapest to priciest provider")},
    {"title": "Latency", "hint": "lower is better", "vmax": 240.0,
     "desc": "time to link the whole list once, in sequence",
     "ticks": ((0, "0 s"), (60, "60"), (120, "120"), (180, "180"), (240, "240 s")),
     "key": ("bar: main run, calls added up", "line: range over repeat runs")},
)


def esc(text: object) -> str:
    """Escape text for an HTML element or attribute.

    Args: text: anything printable.
    Returns: the escaped string.
    Raises: none.
    Side effects: none.
    """
    return html.escape(str(text), quote=True)


def pct(x: float) -> str:
    """Format a share as a percentage with one decimal.

    Args: x: a share, 0 to 1.
    Returns: for example "75.0%".
    Raises: none.
    Side effects: none.
    """
    return f"{x * 100:.1f}%"


def usd(x: float) -> str:
    """Format dollars with four decimals, the precision these small amounts need.

    Args: x: an amount in USD.
    Returns: for example "$0.0117".
    Raises: none.
    Side effects: none.
    """
    return f"${x:.4f}"


def int_comma(x: int) -> str:
    """Format an integer with thousands separators.

    Args: x: an integer.
    Returns: for example "565,176".
    Raises: none.
    Side effects: none.
    """
    return f"{x:,}"


def span(low: float, high: float, fmt: Callable[[float], str]) -> str:
    """Write a range as "low to high".

    Args: low, high: the ends; fmt: formats one end.
    Returns: the text.
    Raises: none.
    Side effects: none.
    """
    return f"{fmt(low)} to {fmt(high)}"


def build_cells(n: dict) -> list[list[dict]]:
    """Work out, per panel and per model, the bar, the line, the labels and the tooltip text.

    Args: n: the contents of numbers.json.
    Returns: three lists (accuracy, cost, latency) of three cells, in MODELS order; a cell holds
    `value` (None when there is nothing to draw), `lo`/`hi` for the line, `main`, `sub` and `tip`.
    Raises: KeyError when numbers.json lacks a field.
    Side effects: none.
    """
    acc, lat, cost = n["accuracy"], n["latency"], n["cost"]
    slots = acc["reachable_slots"]
    accuracy = []
    for key, name, _ in MODELS:
        d = acc[key]
        hits = d["receipt"]["numerator"]
        lo, hi = d["ci95"]
        tip = (f"{name}: {hits} of {slots} slots linked correctly = {pct(d['share'])}. "
               f"95% margin of error {pct(lo)} to {pct(hi)} (resampling the {acc['terms']} terms, "
               "since slots of one term are not independent).")
        if "other_orders" in d:
            others = [v for k, v in d["other_orders"].items() if k != n["meta"]["primary_pass"]]
            tip += (" The same list in the other three candidate orders: "
                    + ", ".join(str(v) for v in others) + " correct.")
        accuracy.append({"value": d["share"], "lo": lo, "hi": hi, "main": pct(d["share"]),
                         "sub": f"{hits} of {slots}", "tip": tip})

    r, q, j = lat["reranker"], lat["qwen38"], lat["jev"]
    latency = [
        {"value": r["seconds"], "lo": r["range"][0], "hi": r["range"][1],
         "main": f"{r['seconds']:.0f} s", "sub": f"{r['seconds_per_slot']:.2f} s per slot",
         "tip": (f"Reranker: {r['seconds']:.0f} s to score the whole list "
                 f"({r['lists_scored']} slots, {int_comma(r['candidates_scored'])} candidates, "
                 f"batches of {r['batch_size']}) on this Mac's GPU, including loading the model. "
                 f"Two earlier runs of the same job took {r['run_seconds'][0]:.0f} s and "
                 f"{r['run_seconds'][1]:.0f} s.")},
        {"value": q["seconds"], "lo": q["range"][0], "hi": q["range"][1],
         "main": f"{q['seconds']:.1f} s", "sub": f"{q['mean_call_seconds']:.2f} s per call",
         "tip": (f"Qwen3.8-27B: {q['seconds']:.1f} s = the time of {lat['list_requests']} calls "
                 f"added up, sent one at a time (mean {q['mean_call_seconds']:.3f} s per call). "
                 f"The other three candidate orders: {q['range'][0]:.1f} to {q['range'][1]:.1f} s. "
                 "Served by PSNC, not by OpenRouter.")},
        {"value": j["seconds"], "lo": j["range"][0], "hi": j["range"][1],
         "main": f"{j['seconds']:.1f} s", "sub": f"{j['mean_call_seconds']:.2f} s per call",
         "tip": (f"Jev: {j['seconds']:.1f} s = the time of {lat['list_requests']} calls added up "
                 f"(mean {j['mean_call_seconds']:.3f} s per call). Range over the four candidate "
                 f"orders {j['range'][0]:.1f} to {j['range'][1]:.1f} s. The run sent "
                 f"{j['workers']} calls at a time, which took about "
                 f"{j['wall_seconds_per_list_as_run']:.0f} s of wall-clock time per list.")},
    ]

    o = cost["qwen38"]["openrouter"]
    lo_p, hi_p = o["cheapest"], o["priciest"]
    spend = [
        {"value": None, "main": "—", "sub": "not priced",
         "tip": ("Reranker: no per-call price. It runs on this Mac, so there is no bill "
                 "(electricity and hardware are not counted).")},
        {"value": cost["qwen38"]["usd_per_list"], "lo": lo_p["usd"], "hi": hi_p["usd"],
         "main": usd(cost["qwen38"]["usd_per_list"]), "sub": "median provider",
         "tip": (f"Qwen3.8-27B was free at PSNC. Priced as if bought on OpenRouter: "
                 f"{int_comma(cost['qwen38']['input_tokens_per_list'])} input and "
                 f"{cost['qwen38']['output_tokens_per_list']} output tokens per list. "
                 f"Median of the {o['providers_offering_logprobs']} providers that offer "
                 f"log-probabilities: {usd(o['median_usd'])} ({o['median_provider']}). "
                 f"Cheapest {usd(lo_p['usd'])} ({lo_p['provider']}, {lo_p['quantization']}), "
                 f"priciest {usd(hi_p['usd'])} ({hi_p['provider']}). "
                 f"Prices read {n['meta']['prices_read_on']}.")},
        {"value": cost["jev"]["usd_per_list"], "main": usd(cost["jev"]["usd_per_list"]),
         "sub": "billed",
         "tip": (f"Jev: {usd(cost['jev']['usd_per_list'])} per list = "
                 f"{int_comma(cost['jev']['input_tokens_per_list'])} input tokens at "
                 f"${cost['jev']['usd_per_million_input']:.3f} per million (output is free). "
                 "This is what OpenRouter billed for the first pass; the other orders cost "
                 "the same.")},
    ]
    return [accuracy, spend, latency]


def bar_path(x: float, y: float, w: float, h: float, r: float = 4.0) -> str:
    """Draw a bar that is square at the baseline and rounded at its data end.

    Args: x, y: top-left corner; w, h: width and height; r: corner radius of the data end.
    Returns: an SVG path.
    Raises: none.
    Side effects: none.
    """
    r = min(r, w, h / 2)
    return (f"M{x:.1f},{y:.1f}h{w - r:.1f}a{r:.1f},{r:.1f} 0 0 1 {r:.1f},{r:.1f}"
            f"v{h - 2 * r:.1f}a{r:.1f},{r:.1f} 0 0 1 {-r:.1f},{r:.1f}h{-(w - r):.1f}z")


def whisker(x_lo: float, x_hi: float, y: float) -> str:
    """Draw a range line with end caps, over a halo in the surface colour so it shows on a bar.

    Args: x_lo, x_hi: the ends; y: the height of the line.
    Returns: SVG for the halo and the line.
    Raises: none.
    Side effects: none.
    """
    cap = 5
    d = (f"M{x_lo:.1f},{y:.1f}H{x_hi:.1f}M{x_lo:.1f},{y - cap:.1f}V{y + cap:.1f}"
         f"M{x_hi:.1f},{y - cap:.1f}V{y + cap:.1f}")
    return f'<path class="halo" d="{d}"/><path class="whisker" d="{d}"/>'


def cell_svg(panel: int, row: int, cell: dict, vmax: float, label: str) -> str:
    """Draw one model's mark in one panel: hover area, bar, line, value and a note beneath it.

    Args: panel, row: the panel and the model's row; cell: from `build_cells`; vmax: the axis end;
    label: the accessible name.
    Returns: an SVG group (focusable, with its tooltip text in `data-tip`).
    Raises: KeyError when the cell lacks `value`, `main`, `sub` or `tip`.
    Side effects: none.
    """
    x0, y = PANEL_X[panel], ROW_Y[row]
    parts = [f'<g class="cell m{row}" tabindex="0" role="group" aria-label="{esc(label)}" '
             f'data-tip="{esc(cell["tip"])}">',
             f'<rect class="hit" x="{x0 - 8}" y="{y - 29}" width="{PANEL_W + 12}" height="58" '
             'rx="8"/>']
    if cell["value"] is None:
        parts.append(f'<text class="absent" x="{x0 + 10}" y="{y - 2}">no per-call price:</text>'
                     f'<text class="absent" x="{x0 + 10}" y="{y + 14}">runs on the Mac</text>')
    else:
        width = cell["value"] / vmax * BAR_W
        parts.append(f'<path class="bar" d="{bar_path(x0, y - BAR_H / 2, width, BAR_H)}"/>')
        if cell.get("lo") is not None:
            parts.append(whisker(x0 + cell["lo"] / vmax * BAR_W,
                                 x0 + cell["hi"] / vmax * BAR_W, y))
    parts.append(f'<text class="val" x="{x0 + VALUE_DX}" y="{y - 1}">{esc(cell["main"])}</text>'
                 f'<text class="valsub" x="{x0 + VALUE_DX}" y="{y + 15}">{esc(cell["sub"])}</text>')
    parts.append("</g>")
    return "".join(parts)


def chart_svg(n: dict) -> str:
    """Draw the whole chart: row labels, three panels with axes, marks and a key under each.

    Args: n: the contents of numbers.json.
    Returns: one inline SVG.
    Raises: KeyError when numbers.json lacks a field.
    Side effects: none.
    """
    cells = build_cells(n)
    parts = [f'<svg viewBox="0 0 {WIDTH} {HEIGHT}" role="img" '
             'aria-label="Accuracy, cost and latency of the reranker, Qwen3.8-27B and Jev. '
             'The numbers are in the tables below.">']
    for row, (_, name, where) in enumerate(MODELS):
        y = ROW_Y[row]
        parts.append(f'<g class="m{row}"><rect class="swatch" x="{LABEL_X}" y="{y - 13}" '
                     f'width="12" height="12" rx="3"/>'
                     f'<text class="rowname" x="{LABEL_X + 20}" y="{y - 2}">{esc(name)}</text>'
                     f'<text class="rowsub" x="{LABEL_X + 20}" y="{y + 15}">{esc(where)}</text></g>')
    for i, spec in enumerate(PANELS):
        x0 = PANEL_X[i]
        parts.append(f'<text class="title" x="{x0}" y="22">{esc(spec["title"])}</text>'
                     f'<text class="hint" x="{x0 + PANEL_W}" y="22" text-anchor="end">'
                     f'{esc(spec["hint"])}</text>'
                     f'<text class="desc" x="{x0}" y="41">{esc(spec["desc"])}</text>')
        for value, label in spec["ticks"]:
            x = x0 + value / spec["vmax"] * BAR_W
            parts.append(f'<line class="grid" x1="{x:.1f}" y1="{GRID_TOP}" x2="{x:.1f}" '
                         f'y2="{AXIS_Y}"/>'
                         f'<text class="tick" x="{x:.1f}" y="{AXIS_Y + 17}" text-anchor="middle">'
                         f'{esc(label)}</text>')
        parts.append(f'<line class="axis" x1="{x0}" y1="{GRID_TOP}" x2="{x0}" y2="{AXIS_Y}"/>'
                     f'<line class="axis" x1="{x0}" y1="{AXIS_Y}" x2="{x0 + BAR_W}" y2="{AXIS_Y}"/>')
        for row, (_, name, _) in enumerate(MODELS):
            parts.append(cell_svg(i, row, cells[i][row], spec["vmax"],
                                  f"{name}, {spec['title'].lower()}: {cells[i][row]['main']}"))
        parts.append(f'<text class="legend" x="{x0}" y="{AXIS_Y + 40}">{esc(spec["key"][0])}</text>'
                     f'<text class="legend" x="{x0}" y="{AXIS_Y + 55}">{esc(spec["key"][1])}</text>')
    parts.append("</svg>")
    return "".join(parts)


def table_row(cells: list[str], header: str | None = None) -> str:
    """Write one table row.

    Args: cells: the cell texts (already safe HTML); header: the row header, if any.
    Returns: the `<tr>` element.
    Raises: none.
    Side effects: none.
    """
    head = f'<th scope="row">{header}</th>' if header else ""
    return "<tr>" + head + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"


def table(caption: str, head: list[str], rows: list[str]) -> str:
    """Write a table in a horizontally scrollable wrapper.

    Args: caption: its caption; head: the column headings; rows: rows from `table_row`.
    Returns: the HTML.
    Raises: none.
    Side effects: none.
    """
    heads = "".join(f"<th>{h}</th>" for h in head)
    return (f'<div class="table-scroll"><table><caption>{caption}</caption>'
            f"<thead><tr>{heads}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>")


def correct_with_share(acc: dict, key: str) -> str:
    """Write a linker's correct count with its share.

    Args: acc: the `accuracy` block of numbers.json; key: the linker.
    Returns: for example "213 (75.0%)".
    Raises: KeyError for an unknown linker.
    Side effects: none.
    """
    d = acc[key]
    return f"{d['receipt']['numerator']} ({pct(d['share'])})"


def tables(n: dict) -> str:
    """Write the table twin: the same numbers as the chart, plus the whole experiments as run.

    Args: n: the contents of numbers.json.
    Returns: a section of HTML.
    Raises: KeyError when numbers.json lacks a field.
    Side effects: none.
    """
    acc, lat, cost, run = n["accuracy"], n["latency"], n["cost"], n["as_run"]
    slots, lists = acc["reachable_slots"], lat["list_requests"]
    o = cost["qwen38"]["openrouter"]
    r, q, j = lat["reranker"], lat["qwen38"], lat["jev"]
    per_list = [
        table_row([correct_with_share(acc, "reranker"), span(*acc["reranker"]["ci95"], pct),
                   "not priced", "—",
                   f"{r['seconds']:.0f} s ({r['range'][0]:.0f} to {r['range'][1]:.0f} over "
                   f"{len(r['run_seconds'])} runs)", f"{r['seconds_per_slot']:.2f} s per slot"],
                  "Reranker"),
        table_row([correct_with_share(acc, "qwen38"), span(*acc["qwen38"]["ci95"], pct),
                   f"{usd(cost['qwen38']['usd_per_list'])} (median)",
                   span(o["cheapest"]["usd"], o["priciest"]["usd"], usd),
                   f"{q['seconds']:.1f} s ({q['range'][0]:.1f} to {q['range'][1]:.1f} over 4 orders)",
                   f"{q['mean_call_seconds']:.3f} s per call"], "Qwen3.8-27B"),
        table_row([correct_with_share(acc, "jev"), span(*acc["jev"]["ci95"], pct),
                   f"{usd(cost['jev']['usd_per_list'])} (billed)", "one price",
                   f"{j['seconds']:.1f} s ({j['range'][0]:.1f} to {j['range'][1]:.1f} over 4 orders)",
                   f"{j['mean_call_seconds']:.3f} s per call"], "Jev"),
    ]
    thousand = 1000 / lists
    scaled = [
        table_row([usd(cost["qwen38"]["usd_per_list"] * thousand),
                   span(o["cheapest"]["usd"] * thousand, o["priciest"]["usd"] * thousand, usd)],
                  "Qwen3.8-27B (median, range)"),
        table_row([usd(cost["jev"]["usd_per_list"] * thousand), "one price"], "Jev (billed)"),
    ]
    rq, rj = run["qwen38"], run["jev"]
    as_run = [
        table_row(["1 scoring run, 286 lists", "—", "not priced",
                   f"{run['reranker']['seconds_summed']:.0f} s",
                   f"{run['reranker']['wall_seconds']:.0f} s"], "Reranker"),
        table_row([f"{int_comma(rq['calls'])} (4 candidate orders)", int_comma(rq["input_tokens"]),
                   f"free at PSNC; at OpenRouter {usd(rq['usd_openrouter_median'])} (median), "
                   f"{span(*rq['usd_openrouter_range'], usd)}",
                   f"{rq['seconds_summed']:.1f} s",
                   f"{rq['wall_seconds']:.1f} s ({rq['workers']} call at a time)"], "Qwen3.8-27B"),
        table_row([f"{int_comma(rj['calls'])} (4 candidate orders + a yes/no pass; "
                   "1 answer asked again)", int_comma(rj["input_tokens"]),
                   f"{usd(rj['usd_billed'])} billed", f"{rj['seconds_summed']:.1f} s",
                   f"{rj['wall_seconds']:.1f} s ({rj['workers']} calls at a time)"], "Jev"),
    ]
    return (
        '<section class="tables"><h2>The numbers behind the chart</h2>'
        f'<p class="small">One list = the {lists} slots that have candidates, linked once. Accuracy '
        f"counts the {slots} slots whose vocabulary the sources reach; the margin of error "
        f"resamples the {acc['terms']} terms.</p>"
        + table("For one list",
                ["", f"Correct of {slots}", "95% margin of error", "Cost per list",
                 "Cost range across providers", "Latency per list", "Per call"], per_list)
        + table("Cost per 1,000 slots (the same figures scaled)",
                ["", "Cost", "Range across providers"], scaled)
        + table("Whole experiments as they were run (not like for like: Jev's includes a yes/no "
                "pass that Qwen3.8-27B did not run)",
                ["", "Calls", "Input tokens", "Cost", "Latency of all calls added up",
                 "Wall-clock as run"], as_run)
        + "</section>")


def lede(n: dict) -> str:
    """Write the bottom line in plain words, with its numbers taken from numbers.json.

    Args: n: the contents of numbers.json.
    Returns: HTML for the opening paragraph.
    Raises: KeyError when numbers.json lacks a field.
    Side effects: none.
    """
    ratios = n["cost"]["ratios"]
    cheap = 1 / ratios["qwen38_cheapest_over_jev"]
    return (
        "<strong>Bottom line.</strong> Accuracy cannot be ranked: the margins of error overlap. "
        "Jev and Qwen3.8-27B are about equally fast. Cost depends on where Qwen3.8-27B is "
        "bought: at a typical OpenRouter provider it would cost about "
        f"{ratios['qwen38_median_over_jev']:.1f}× what Jev costs, and across providers it ranges "
        f"from {cheap:.0f}× cheaper than Jev to {ratios['qwen38_priciest_over_jev']:.0f}× dearer. "
        "The reranker has no per-call price but was more than twice as slow, on the Mac."
    )


def notes(n: dict) -> str:
    """Write the notes under the chart: what makes the comparison fair and what it cannot say.

    Args: n: the contents of numbers.json.
    Returns: an HTML list.
    Raises: KeyError when numbers.json lacks a field.
    Side effects: none.
    """
    acc, lat, cost = n["accuracy"], n["latency"], n["cost"]
    o = cost["qwen38"]["openrouter"]
    items = [
        f"<strong>Same test for all.</strong> The same {acc['reachable_slots']} slots, the same "
        "candidates and the same strict scoring (a Wikidata twin of a QUDT kind counts as wrong). "
        "The three accuracy ranges overlap, so they cannot be ranked.",
        f"<strong>One list</strong> is the {lat['list_requests']} slots that have candidates, "
        "linked once. For Jev and Qwen3.8-27B the bar is the first candidate order (the "
        "pre-specified main run) and the line shows the other three orders; for the reranker it is "
        "one scoring run, with two earlier runs of the same job.",
        "<strong>Latency</strong> adds up the time of every call, one at a time. Where it ran "
        "matters: the reranker on this Mac's GPU (including loading the model), Qwen3.8-27B on "
        f"PSNC's server, Jev through OpenRouter. The Jev run sent {lat['jev']['workers']} calls at a "
        f"time, about {lat['jev']['wall_seconds_per_list_as_run']:.0f} s of wall-clock time per "
        "list.",
        "<strong>Cost.</strong> Jev is what OpenRouter billed. Qwen3.8-27B was free at PSNC, so it "
        "is priced as if bought on OpenRouter, from the "
        f"{o['providers_offering_logprobs']} of {o['providers_listed']} providers that offer "
        "log-probabilities (this method needs them). "
        f"Prices were read on {n['meta']['prices_read_on']} and they change. The reranker has no "
        "per-call price.",
        "<strong>Post hoc.</strong> Put together after the results were known; nothing was re-run. "
        f"Jev's yes/no formulation is not shown ({acc['jev']['yes_no_pass']} of "
        f"{acc['reachable_slots']} correct).",
    ]
    return '<ul class="notes">' + "".join(f"<li>{i}</li>" for i in items) + "</ul>"


PAGE = Template("""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Accuracy, cost and latency of three linkers</title>
<style>
:root {
  color-scheme: light dark;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11, 11, 11, 0.10); --wash: rgba(11, 11, 11, 0.05);
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255, 255, 255, 0.10);
    --wash: rgba(255, 255, 255, 0.06); --s1: #3987e5; --s2: #d95926; --s3: #199e70;
  }
}
:root[data-theme="dark"] {
  --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --border: rgba(255, 255, 255, 0.10);
  --wash: rgba(255, 255, 255, 0.06); --s1: #3987e5; --s2: #d95926; --s3: #199e70;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink);
  font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1240px; margin: 0 auto; padding: 24px 16px 48px; }
h1 { font-size: 22px; line-height: 1.25; margin: 0 0 8px; font-weight: 650; }
h2 { font-size: 16px; margin: 28px 0 6px; }
.lede { color: var(--ink-2); max-width: 86ch; margin: 0 0 16px; font-size: 15px; }
.lede strong { color: var(--ink); }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
  padding: 14px 16px 10px; }
.chart-wrap { position: relative; }
.chart-scroll { overflow-x: auto; }
svg { display: block; width: 100%; min-width: 880px; height: auto; }
svg text { font-family: inherit; fill: var(--ink); }
.title { font-size: 15px; font-weight: 650; }
.hint, .desc, .rowsub, .valsub, .legend { font-size: 11.5px; }
.hint, .rowsub, .legend, .tick, .absent { fill: var(--muted); }
.desc, .valsub { fill: var(--ink-2); }
.rowname { font-size: 14px; font-weight: 650; }
.val { font-size: 15px; font-weight: 650; }
.tick { font-size: 11px; font-variant-numeric: tabular-nums; }
.absent { font-size: 12px; font-style: italic; }
.grid { stroke: var(--grid); stroke-width: 1; }
.axis { stroke: var(--axis); stroke-width: 1; }
.hit { fill: transparent; }
.cell { outline: none; }
.cell:hover .hit, .cell:focus-visible .hit { fill: var(--wash); }
.m0 .bar, .m0 .swatch { fill: var(--s1); }
.m1 .bar, .m1 .swatch { fill: var(--s2); }
.m2 .bar, .m2 .swatch { fill: var(--s3); }
.halo { stroke: var(--surface); stroke-width: 4; fill: none; stroke-linecap: round; }
.whisker { stroke: var(--ink-2); stroke-width: 1.5; fill: none; stroke-linecap: round; }
#tip { position: absolute; z-index: 5; max-width: 360px; background: var(--ink);
  color: var(--surface); padding: 8px 10px; border-radius: 8px; font-size: 12.5px;
  line-height: 1.4; pointer-events: none; box-shadow: 0 6px 20px rgba(0, 0, 0, 0.18); }
#tip[hidden] { display: none; }
.notes { color: var(--ink-2); font-size: 13px; margin: 10px 4px 4px; padding-left: 18px;
  max-width: 120ch; }
.notes li { margin: 3px 0; }
.notes strong { color: var(--ink); }
.small { color: var(--ink-2); font-size: 13px; margin: 0 0 8px; }
.table-scroll { overflow-x: auto; margin: 10px 0 16px; }
table { border-collapse: collapse; width: 100%; font-size: 13.5px; background: var(--surface);
  border: 1px solid var(--border); border-radius: 8px; }
caption { text-align: left; font-weight: 650; padding: 6px 2px; color: var(--ink); }
th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--grid);
  vertical-align: top; }
thead th { color: var(--ink-2); font-weight: 600; }
tbody th { font-weight: 650; white-space: nowrap; }
footer { margin-top: 22px; color: var(--muted); font-size: 12.5px; max-width: 110ch; }
.png main { padding: 16px; }
.png .tables, .png footer, .png #tip { display: none; }
</style>
</head>
<body>
<main>
  <h1>Accuracy, cost and latency of the three linkers</h1>
  <p class="lede">${lede}</p>
  <div class="card">
    <div class="chart-wrap"><div class="chart-scroll">${chart}</div><div id="tip" role="tooltip" hidden></div></div>
    ${notes}
  </div>
  ${tables}
  <footer>Made by compute_numbers.py and render_figure.py from the stored run files; numbers.json holds
  every number. A post hoc side analysis: it was not pre-specified and re-runs nothing.</footer>
</main>
<script>
(function () {
  var theme = /[?&]theme=(light|dark)/.exec(location.search);
  if (theme) { document.documentElement.setAttribute('data-theme', theme[1]); }
  var wrap = document.querySelector('.chart-wrap');
  var tip = document.getElementById('tip');
  function show(cell, x, y) {
    tip.textContent = cell.getAttribute('data-tip');
    tip.hidden = false;
    var box = wrap.getBoundingClientRect();
    var left = Math.min(Math.max(x - box.left + 14, 8), box.width - tip.offsetWidth - 8);
    var top = y - box.top + 18;
    if (y + 18 + tip.offsetHeight > window.innerHeight - 8) { top = y - box.top - tip.offsetHeight - 12; }
    tip.style.left = left + 'px';
    tip.style.top = top + 'px';
  }
  function hide() { tip.hidden = true; }
  document.querySelectorAll('.cell').forEach(function (cell) {
    cell.addEventListener('pointermove', function (e) { show(cell, e.clientX, e.clientY); });
    cell.addEventListener('pointerleave', hide);
    cell.addEventListener('focus', function () {
      var b = cell.getBoundingClientRect();
      show(cell, b.left + b.width / 2, b.top + b.height / 2);
    });
    cell.addEventListener('blur', hide);
  });
  if (location.search.indexOf('png') !== -1) { document.documentElement.classList.add('png'); }
})();
</script>
</body>
</html>
""")


def main() -> None:
    """Build figure.html from numbers.json.

    Args: none.
    Returns: None.
    Raises: OSError when numbers.json is missing (run compute_numbers.py first).
    Side effects: reads numbers.json; writes figure.html beside this file.
    """
    n = json.loads((HERE / "numbers.json").read_text(encoding="utf-8"))
    page = PAGE.substitute(lede=lede(n), chart=chart_svg(n), notes=notes(n), tables=tables(n))
    (HERE / "figure.html").write_text(page, encoding="utf-8")
    print(f"wrote {HERE / 'figure.html'}")


if __name__ == "__main__":
    main()
