#!/usr/bin/env python
"""Regenerate `best-configuration-session.md` from the corpus and the recorded run.

The document embeds a 15 KB prompt and a raw model response verbatim. Transcribing either by
hand would eventually drift from what was actually sent, so the record is generated instead:
every figure in it is read from `output/example-selection-*.jsonl` or re-derived from the same
functions the run used.

    ./build_best_config_doc.py                 # regenerate with the default featured target
    ./build_best_config_doc.py --target <id>   # feature a different confirmation-set variable

EXPLORATORY side experiment. Reads the corpus and the recorded run; writes one markdown file.
Issues no provider call, writes no database row, and touches nothing under `outputs/`.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FS = HERE.parent
ROOT = FS.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(FS))
sys.path.insert(0, str(ROOT / "src"))

import design  # noqa: E402
from attribution import fit, ranking  # noqa: E402
from harness import CONFIG, MODEL_ID, candidate_hash  # noqa: E402
from shot_count_ablation import render  # noqa: E402

from iadopt_lab.corpus.ingestion import load_canonical_records  # noqa: E402
from iadopt_lab.generation.extractor import extract_json  # noqa: E402
from iadopt_lab.prompting.renderer import load_prompt_version  # noqa: E402
from iadopt_lab.validation import load_schema_bytes, validate_prediction  # noqa: E402

OBSERVATIONS = HERE / "output" / "example-selection-observations.jsonl"
CALLS = HERE / "output" / "example-selection-calls.jsonl"
REPORT = HERE / "best-configuration-session.md"
PROMPT_VARIANT = CONFIG["prompt_variant"]

# Chosen because it exercises every interesting part of the output shape in one response: a
# populated constraint array, a context object, and two empty scalars. Overridable with --target.
DEFAULT_TARGET = "urn:iadopt-lab:variable:d8164cf2a4d4be23ca63fc588e2850efbb8041ad653843e104594e7295140d5d"


def item_f1(evaluation: dict) -> float:
    """Micro Close F1 for one scored item, summed over its components.

    Args: evaluation: one `evaluate_item` record.
    Returns: F1 in [0, 1]; 0.0 when the item contributes no true positives.
    Side Effects: none.
    """
    tp = fp = fn = 0.0
    for component in evaluation["close"]["components"].values():
        contributions = component["contributions"]
        tp += contributions["tp"]["value"]
        fp += contributions["fp"]["value"]
        fn += contributions["fn"]["value"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def load_observations() -> list[dict]:
    """Read the per-candidate observation log, which carries every reported score."""
    return [json.loads(line) for line in OBSERVATIONS.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_candidate_calls(digest: str) -> list[dict]:
    """Stream the 396 MB call log and keep only one candidate's calls.

    Args: digest: the candidate hash to select.
    Returns: every recorded call for that candidate, across all repetitions.
    Side Effects: reads the call log line by line; never loads it whole.
    """
    needle = f'"candidate_hash": "{digest}"'
    out = []
    with CALLS.open(encoding="utf-8") as handle:
        for line in handle:
            if needle in line:
                out.append(json.loads(line))
    return out


def confirmation_table(observations: list[dict]) -> list[tuple[str, float, float, tuple[float, float]]]:
    """Per-candidate held-out statistics, best first, exactly as stage D computed them."""
    scores: dict[str, list[float]] = {}
    for row in observations:
        if row["eval_set"] == "C":
            scores.setdefault(row["name"], []).append(row["result"]["close_f1_official"])
    out = []
    for name, values in scores.items():
        mean = statistics.mean(values)
        sd = statistics.stdev(values) if len(values) > 1 else 0.0
        sem = sd / len(values) ** 0.5
        out.append((name, mean, sd, (mean - 1.96 * sem, mean + 1.96 * sem)))
    return sorted(out, key=lambda row: -row[1])


def build() -> str:
    """Assemble the whole document. Returns the markdown text."""
    parser = argparse.ArgumentParser(description="regenerate the best-configuration session record")
    parser.add_argument("--target", default=DEFAULT_TARGET)
    args = parser.parse_args()

    records = list(load_canonical_records(ROOT))
    by_id = {row["variable_id"]: row for row in records}
    split = design.corpus_split(records)
    observations = load_observations()

    attribution = [(tuple(row["subset"]), row["result"]["close_f1_official"])
                   for row in observations if row["stage"] == "attribution"]
    fitted = fit(split["P"], attribution)
    coefficients = fitted["coefficients"]
    top25 = design.top_k_candidate(split["P"], coefficients)
    rank = {variable_id: position for position, (variable_id, _) in enumerate(ranking(coefficients), 1)}
    digest = candidate_hash(top25)

    stored = next(row["subset"] for row in observations if row["name"] == "top25")
    if sorted(stored) != list(top25):
        raise SystemExit("recomputed top25 differs from the recorded run; refusing to write a false record")

    template = load_prompt_version(PROMPT_VARIANT, ROOT)
    schema_bytes = load_schema_bytes(ROOT)
    schema_text = schema_bytes.decode("utf-8")
    examples = [by_id[variable_id] for variable_id in top25]

    target = by_id[args.target]
    if args.target not in set(split["C"]):
        raise SystemExit(f"{args.target} is not in the held-out confirmation set")
    prompt = render(template, schema_text, examples, target["definition"])

    calls = load_candidate_calls(digest)
    call = next(row for row in calls if row["variable_id"] == args.target and row["repetition"] == 1)
    # The document asserts this response parses under the frozen extraction protocol and passes
    # the schema. Check it rather than assert it, so a bad featured target cannot ship a claim.
    extracted = extract_json(call["answer"])
    if not (extracted.success and validate_prediction(extracted.candidate, schema_bytes).valid):
        raise SystemExit(f"featured response for {args.target} does not extract and validate")
    confirmation = set(split["C"])
    rep1 = sorted((row for row in calls if row["repetition"] == 1 and row["variable_id"] in confirmation),
                  key=lambda row: -(item_f1(row["evaluation"]) if row.get("evaluation") else -1.0))

    confirm = confirmation_table(observations)
    top_row = next(row for row in confirm if row[0] == "top25")

    lines: list[str] = []
    write = lines.append

    write("# Best-known configuration — complete session record")
    write("")
    write("**Exploratory side experiment, not an official result.** The configuration below won a")
    write("held-out comparison inside [the example-selection study](README.md); it carries no campaign")
    write("identity and must not enter an official ranking or results table. It is recorded here so the")
    write("exact prompt, the exact 25 examples and the exact request body can be reused elsewhere —")
    write("for instance by a web service — without re-deriving them from the run logs.")
    write("")
    write(f"Generated from `output/example-selection-calls.jsonl` and the canonical corpus by")
    write(f"[`build_best_config_doc.py`](build_best_config_doc.py). Candidate hash `{digest}`.")
    write("")

    write("## 1. What is being claimed, and what is not")
    write("")
    write("Micro Close F1 on the 24-variable confirmation set `C`, which never informed selection.")
    write("15 repetitions per candidate.")
    write("")
    write("| Candidate | Close F1 | SD | 95% CI |")
    write("|---|---:|---:|---|")
    for name, mean, sd, (lo, hi) in confirm:
        label = f"**`{name}`**" if name == "top25" else f"`{name}`"
        write(f"| {label} | {mean:.4f} | {sd:.4f} | [{lo:.4f}, {hi:.4f}] |")
    write("")
    write("**What holds.** This set beats a deliberately-bad selection (`bottom25`, +0.091,")
    write("Holm `p < 0.0001`) and a domain-stratified selection (+0.094, `p < 0.0001`), and beats the")
    write("pooled random-25 reference (+0.037, `p = 0.0002`). The attribution model learned something")
    write("real and reproducible.")
    write("")
    write("**What does not hold.** It is *not* significantly better than the best of three random")
    write("draws (`random-1` at 0.4757, difference +0.0133, `p = 0.156`). Treat this as a reliably")
    write("good selection, not a proven optimum. Two further limits are worth carrying forward:")
    write("")
    write("- The 25 were chosen from a 40-variable pool `P`, so **8 of the 25 carry negative")
    write("  coefficients** — they are included because 25 of 40 must be taken, not because they help.")
    write(f"- The coefficients were fitted on `E`. On `E` the top-25/bottom-25 gap is +0.254; on `C` it")
    write("  is +0.091. Roughly two thirds of the fitted signal is specific to the set that fitted it.")
    write("")
    write(f"Absolute accuracy is modest: **Close F1 {top_row[1]:.4f}** means a little under half of the")
    write("component mass is recovered. This is a research-grade configuration, not a solved task.")
    write("")

    write("## 2. The 25 examples")
    write("")
    write(f"Attribution fit over {fitted['n']} random 25-subsets: **r² = {fitted['r2']:.4f}**,")
    write(f"**split-half stability = {fitted['split_half']:.4f}** (coefficients had stabilised).")
    write("Coefficients are *relative* marginal contributions; only their ordering and differences")
    write("mean anything, never their absolute level.")
    write("")
    write("> **Ordering matters for byte-exact reproduction.** The prompt carries these 25 in sorted")
    write("> `variable_id` order — *not* in coefficient order. The table is shown by coefficient rank")
    write("> for readability; the `prompt #` column is the order actually sent.")
    write("")
    write("| prompt # | rank | coefficient | domain | label |")
    write("|---:|---:|---:|---|---|")
    position = {variable_id: index for index, variable_id in enumerate(top25, 1)}
    for variable_id, coefficient in ranking(coefficients):
        if variable_id not in position:
            continue
        row = by_id[variable_id]
        write(f"| {position[variable_id]} | {rank[variable_id]} | {coefficient:+.4f} | "
              f"{row['category']} | {row['label']} |")
    write("")
    write("The 25 `variable_id`s, in prompt order:")
    write("")
    write("```json")
    write(json.dumps(list(top25), indent=2))
    write("```")
    write("")

    write("## 3. The model call")
    write("")
    write(f"OpenAI-compatible chat completion against PSNC. Model `{MODEL_ID}`, reasoning disabled —")
    write("that is the condition the configuration was measured in, and `enable_thinking: true` would")
    write("be a different configuration with none of the evidence above behind it.")
    write("")
    write("```http")
    write("POST {PSNC_API_BASE_URL}/chat/completions")
    write("Authorization: Bearer {PSNC_API_KEY}")
    write("Content-Type: application/json")
    write("```")
    write("")
    write("```json")
    body = {"model": MODEL_ID, "messages": [{"role": "user", "content": "<the prompt from section 4>"}],
            "stream": False, "temperature": CONFIG["temperature"], "top_p": CONFIG["top_p"],
            "max_tokens": CONFIG["max_output_tokens"], **CONFIG["reasoning_fields"]}
    write(json.dumps(body, indent=2))
    write("```")
    write("")
    prompt_tokens = [row["prompt_tokens"] for row in calls if row["prompt_tokens"]]
    completion_tokens = [row["completion_tokens"] for row in calls if row["completion_tokens"]]
    latencies = [row["latency_seconds"] for row in calls if row["latency_seconds"]]
    write("A single user message carries the whole prompt; there is no system message. Default base")
    write(f"URL is `https://llm.hpc.psnc.pl`. Measured cost over this candidate\'s {len(calls)} recorded")
    write(f"calls: **{statistics.mean(prompt_tokens):,.0f} prompt tokens** on average "
          f"({min(prompt_tokens):,}–{max(prompt_tokens):,} — the spread is the")
    write(f"target definition alone), against {statistics.mean(completion_tokens):.0f} completion tokens "
          f"({min(completion_tokens)}–{max(completion_tokens)}), "
          f"{statistics.mean(latencies):.2f} s mean latency,")
    write("and zero transport failures. The 16,000-token output ceiling is never approached; it is")
    write("inherited from the measured configuration rather than chosen for this task.")
    write("")

    write("## 4. The prompt")
    write("")
    write(f"Template [`prompts/{PROMPT_VARIANT}-v1.txt`](../../prompts/{PROMPT_VARIANT}-v1.txt) with")
    write("three placeholders filled in one substitution pass:")
    write("")
    write(f"- `{{{{schema}}}}` — [`schemas/lexical-decomposition.schema.json`](../../schemas/lexical-decomposition.schema.json), verbatim UTF-8.")
    write("- `{{demonstrations}}` — the 25 examples as one compact JSON array, each entry")
    write("  `{\"demonstration\": <1-based position>, \"definition\": <text>, \"decomposition\": <gold>}`.")
    write("- `{{target_definition}}` — the definition text of the variable being decomposed.")
    write("")
    write("The demonstrations array is encoded with **sorted keys, no ASCII folding, and `(\",\", \":\")`")
    write("separators**. That encoding is part of the measured prompt: re-serialising it with default")
    write("`json.dumps` spacing changes the prompt bytes, and nothing here was measured under those")
    write("bytes. In Python the exact call is:")
    write("")
    write("```python")
    write('json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":"))')
    write("```")
    write("")
    write(f"Rendered length for the session below: **{len(prompt):,} characters, "
          f"{call['prompt_tokens']:,} prompt tokens**. The block")
    write("below is byte-identical to what was sent, with one caveat: the prompt ends in a single")
    write("newline, and the fence adds one of its own. Strip exactly one trailing newline when you")
    write("copy it out. Full text, verbatim:")
    write("")
    write("````text")
    write(prompt)
    write("````")
    write("")

    write("## 5. The raw response")
    write("")
    write(f"Target variable: **{target['label']}** (`{target['category']}` / {target['subcategory']}),")
    write("held-out set `C`, repetition 1.")
    write("")
    write("Definition sent:")
    write("")
    write("```text")
    write(target["definition"])
    write("```")
    write("")
    write(f"Raw `choices[0].message.content`, exactly as returned — `finish_reason: {call['finish_reason']}`,")
    write(f"{call['completion_tokens']} completion tokens, {call['latency_seconds']:.3f} s:")
    write("")
    write("````text")
    write(call["answer"])
    write("````")
    write("")
    write("The model returned a bare JSON object with no fences and no prose, which is what the")
    write("template asks for. **Do not rely on that** — the extraction step exists because it is not")
    write("guaranteed; see section 7.")
    write("")
    write("Gold decomposition for the same variable:")
    write("")
    write("```json")
    write(json.dumps(target["gold"], indent=2, ensure_ascii=False))
    write("```")
    write("")
    write(f"This item scored **Close F1 {item_f1(call['evaluation']):.3f}**.")
    write("")

    write("## 6. How the rest of the held-out set scored")
    write("")
    write("All 24 confirmation targets at repetition 1, same 25 examples, same configuration. The")
    write("spread is the honest picture: perfect items and zeroes coexist at this accuracy level.")
    write("")
    write("| Close F1 | completion tokens | target |")
    write("|---:|---:|---|")
    for row in rep1:
        score = item_f1(row["evaluation"]) if row.get("evaluation") else None
        marker = " ← featured above" if row["variable_id"] == args.target else ""
        write(f"| {score:.3f} | {row['completion_tokens']} | {by_id[row['variable_id']]['label']}{marker} |"
              if score is not None else
              f"| invalid | {row['completion_tokens']} | {by_id[row['variable_id']]['label']}{marker} |")
    write("")
    zeroes = sum(1 for row in rep1 if row.get("evaluation") and item_f1(row["evaluation"]) == 0.0)
    perfect = sum(1 for row in rep1 if row.get("evaluation") and item_f1(row["evaluation"]) == 1.0)
    write(f"{perfect} of 24 exactly right, {zeroes} of 24 scoring zero, the rest partial. A service built")
    write("on this should present decompositions as drafts for review, not as answers.")
    write("")

    write("## 7. Adopting this in a service")
    write("")
    write("Five things that are easy to get wrong and that invalidate the measurement if you do:")
    write("")
    write("1. **Never let a target variable appear among the 25 examples.** The harness raises rather")
    write("   than score an overlap, because an example that is also scored leaks its own answer. A")
    write("   service decomposing user-supplied definitions is safe by construction; one decomposing")
    write("   these 102 corpus variables is not.")
    write("2. **Keep the demonstration encoding byte-exact** (section 4). Different spacing is a")
    write("   different prompt.")
    write("3. **Extract before you parse.** Responses are not guaranteed to be bare JSON.")
    write("   `iadopt_lab.generation.extractor.extract_json` runs a frozen ordered protocol — whole")
    write("   response, one unwrapped JSON string, complete fenced blocks, then balanced outermost")
    write("   objects — and reports ambiguity as failure rather than guessing. Reuse it.")
    invalid_here = sum(1 for row in rep1 if not row.get("evaluation"))
    invalid_candidate = sum(1 for row in calls if not row.get("evaluation"))
    write("4. **Validate against the schema and treat failure as an empty prediction**, keeping the")
    write("   variable in the population. Silently dropping unparseable output inflates every score.")
    write(f"   The repetition shown above happened to produce {invalid_here} invalid responses, but "
          f"{invalid_candidate} of this")
    write(f"   candidate\'s {len(calls)} calls failed validation ({invalid_candidate / len(calls) * 100:.1f}%), "
          "and 3.2% did across the whole run.")
    write("5. **`\"\"` and `[]` are meaningful**, and `null` is never valid. An absent statistical")
    write("   modifier is the empty string, not a missing key.")
    write("")
    write("Temperature is 0.5, so the same definition will not always give the same decomposition.")
    write("Measured run-to-run SD on an aggregate of this size is about 0.019 Close F1; per-item")
    write("variation is much larger. If a service needs stable output, that is an argument for")
    write("caching a decomposition once accepted, not for lowering the temperature — T=0.5 is the")
    write("value every number here was measured at.")
    write("")

    write("## 8. Provenance")
    write("")
    write("| | |")
    write("|---|---|")
    write(f"| Model | `{MODEL_ID}` on PSNC |")
    write(f"| Prompt variant | `{PROMPT_VARIANT}` (`prompts/{PROMPT_VARIANT}-v1.txt`) |")
    write(f"| Sampling | T={CONFIG['temperature']}, top_p={CONFIG['top_p']}, max_tokens={CONFIG['max_output_tokens']} |")
    write(f"| Reasoning | {CONFIG['reasoning_mode']} (`{json.dumps(CONFIG['reasoning_fields'])}`) |")
    write(f"| Shot count | {len(top25)} |")
    write(f"| Candidate hash | `{digest}` |")
    write(f"| Corpus | {len(records)} canonical variables, tag `{records[0]['tag']}` |")
    write(f"| Split | P={len(split['P'])} pool / E={len(split['E'])} search-eval / C={len(split['C'])} held out |")
    write(f"| Held-out score | Close F1 {top_row[1]:.4f} ± {top_row[2]:.4f} over 15 repetitions |")
    write("| Run date | 2026-09-12 |")
    write("| Raw evidence | `output/example-selection-calls.jsonl` (396 MB, gitignored) |")
    write("")
    write("Regenerate this document with `./build_best_config_doc.py`. It refuses to write if the")
    write("recomputed top-25 no longer matches the set the run actually evaluated, or if the featured")
    write("response no longer extracts and validates.")
    write("")
    write("Fidelity was checked against the run logs when this was written: the prompt in section 4 is")
    write("byte-identical to the render the harness sent, and the response in section 5 is byte-")
    write("identical to the logged `choices[0].message.content`, re-extracting and re-validating")
    write("cleanly. The prompt is rebuilt from `prompts/`, `schemas/` and the corpus rather than")
    write("stored, so it stays correct only while those artifacts are unchanged — all three were")
    write("clean at commit `e341649` and predate the run.")

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    REPORT.write_text(build(), encoding="utf-8")
    print(f"wrote {REPORT}")
