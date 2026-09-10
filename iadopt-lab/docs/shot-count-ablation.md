# Shot-count ablation (exploratory side analysis)

**This is not part of the official experiment.** It is a curiosity-driven ablation with its
own runner, its own output directory and its own Excel workbook. Its results do not appear
in any official ranking, campaign, or results table, and it changed nothing about the
official grid, campaigns, task counts or conclusions. If you are reading this to understand
the official experiment, you are in the wrong document — see [campaign log](campaign-log.md).

## Scientific question

With everything else held at a model's best known reasoning-off configuration, how does
performance change as the number of few-shot examples goes 0 → 1 → 3 → 5 → 7 → 10?

The official experiment found shot count to be by far the strongest effect (mean Close F1
0.244 at 0 shots rising to 0.353 at 5) and, importantly, **the curve had not flattened at
5**. But 5 is the official grid's ceiling, not an optimum, so the official experiment could
not tell whether more examples keep helping. This ablation answers that.

## Models and why reasoning is off

`Qwen3.8-27B` and `DeepSeek-V4-Flash`, both on PSNC.

Reasoning is disabled because that is the condition in which their best configurations were
measured, and because holding it fixed is the point: the only variable that changes here is
shot count. It also keeps the ablation cheap and fast — PSNC reasoning-off calls run
0.7–3.3s, against 150–233s with reasoning on (D-044, D-045).

PSNC was chosen deliberately so this can run without competing with the main remaining
campaign, which is entirely OpenRouter.

## How the best configurations were selected

From the stored official results, not by guessing. Both come from completed campaign
`5cdc9417`, taken as each model's highest Close F1 among its 36 rankable reasoning-off
configurations:

| | Qwen3.8-27B | DeepSeek-V4-Flash |
|---|---|---|
| Prompt variant | matrix-decomposition | matrix-decomposition |
| Temperature / top_p | 0.5 / 1.0 | 0.5 / 1.0 |
| Max output tokens | 16,000 | 16,000 |
| Reasoning | `disabled` (`enable_thinking: false`) | `not_applicable` (emits none) |
| Official result | rank 1, Close F1 0.3912 | rank 2, Close F1 0.3899 |

Both winners share the same prompt and temperature, so the only differences between the two
models in this ablation are the model itself and its reasoning field. **No unexpected extra
dimension appeared**, so the design stays a clean 2 × 6 × 92.

## Shot pool and evaluation population

The corpus has 102 variables. Ten are reserved as the example pool, leaving **92 evaluated
at every shot level**. The population is identical across all six conditions, so it cannot
confound the trend.

The pool is deterministic and **nested** — lower shot counts are prefixes of the same
ordering, never independently resampled:

| # | Used at | Origin | Variable |
|---:|---|---|---|
| 1 | 1,3,5,7,10 | official demonstration #1 | Air daily maximum temperature at 1.7m |
| 2 | 3,5,7,10 | official demonstration #2 | Number of persons receiving welfare in a statistical unit |
| 3 | 3,5,7,10 | official demonstration #3 | Blood lactate concentration |
| 4 | 5,7,10 | official demonstration #4 | Circulation-Mode-in-Pipe |
| 5 | 5,7,10 | official demonstration #5 | Heat stress index |
| 6 | 7,10 | added, sorted `variable_id` | Primary function of building |
| 7 | 7,10 | added, sorted `variable_id` | Level of highest education |
| 8 | 10 | added, sorted `variable_id` | Overnight stays in 3-star hotel near the sea shore |
| 9 | 10 | added, sorted `variable_id` | Mass flux of carbon into soil from vegetation due to senescence |
| 10 | 10 | added, sorted `variable_id` | Feral-free enclosure area |

The five official demonstrations come first, in their official order, so the 1-, 3- and
5-shot conditions use exactly the examples the official experiment used. The remaining five
are selected by sorted `variable_id`, which is reproducible and independent of anything
measured. The 92-variable evaluation set is a strict subset of the official 97.

**Prompt fidelity is verified, not assumed:** at 0, 1, 3 and 5 shots the ablation renderer
produces byte-identical prompts to the official `render_base_prompt`. 7 and 10 extend the
same pattern with more entries in the same demonstrations array.

## Why it does not use the official campaign machinery

It cannot, without contaminating the official experiment. Three guards block it:

- `select_demonstrations` rejects any shot count outside `(0, 1, 3, 5)`.
- `render_base_prompt` requires the exact official five-variable demonstration prefix.
- The frozen demonstration pool has five members; this needs ten.

Every way around them means editing `src/`, `prompts/` or `data/manifests/` — all of which
feed the official `implementation` artifact hash. Changing that hash invalidates every
official plan and would force ~21,000 completed, paid tasks to be re-run. A standalone
runner was the smaller and safer choice.

## Isolation

| Concern | How it is isolated |
|---|---|
| Implementation hash | Runner lives in `experiments/`, which no artifact collector walks. Verified: the official hash was `ecfe44cb…` before and after. |
| Configuration | Reads nothing from `parameters.yml`; its configuration is fixed in the runner. Never writes it. |
| Database | Writes no campaign, run, task or evaluation row. Results go to a JSONL file. |
| Rankings | Results carry no campaign/run/task identity, so they cannot enter an official ranking even accidentally. |
| Outputs | Writes only `experiments/output/`, never `outputs/`. |
| Dependencies | `openpyxl` installed into the venv only; `uv.lock` and `pyproject.toml` unchanged, which matters because both feed the official hash. |
| Provider | PSNC, while the main remaining campaign is entirely OpenRouter. |

## Running and resuming

```bash
cd iadopt-lab
.venv/bin/python experiments/shot_count_ablation.py --plan-only   # counts only, no calls
.venv/bin/python experiments/shot_count_ablation.py               # run or resume
.venv/bin/python experiments/shot_count_ablation.py --limit 2     # small smoke test
.venv/bin/python experiments/build_ablation_excel.py              # rebuild the workbook
```

Resumability is by construction: every scored evaluation is appended to
`experiments/output/shot-count-ablation-results.jsonl` as it completes, and a re-run skips
any `(model, shot count, variable)` triple already present. An interruption costs nothing
and no provider call is repeated. Re-running after completion does nothing and prints
`already complete`.

Execution settings: concurrency 8, timeout 120s. Both are generous for reasoning-off PSNC
calls, which measured 0.7–3.3s; the long reasoning-enabled settings are deliberately not
used here.

## Results

1,104 evaluations, zero failed calls, 4 minutes 4 seconds.

Micro Close F1, aggregated exactly as the official experiment does:

| Shots | Qwen3.8-27B | DeepSeek-V4-Flash |
|---:|---:|---:|
| 0 | 0.227 | 0.264 |
| 1 | 0.264 | 0.265 |
| 3 | 0.348 | 0.322 |
| 5 | 0.362 | **0.360** |
| 7 | 0.370 | 0.349 |
| 10 | **0.433** | 0.356 |

**The two models react differently, which is the main finding.**

`Qwen3.8-27B` improves monotonically across the whole range and has still not plateaued at
10 shots. Its largest single gain is the last one, 7 → 10 (+0.063), which is bigger than
3 → 7 (+0.022). Whatever is happening for this model, ten examples is not enough to exhaust
it.

`DeepSeek-V4-Flash` saturates. It climbs to 0.360 at 5 shots and then stops: 7 shots is
*worse* (0.349) and 10 shots (0.356) is still below its 5-shot score. For this model the
official grid's ceiling of 5 happens to be about right.

Both models gain most between 1 and 3 shots, and both gain almost nothing from 0 to 1.

Treat this as exploratory. It is one repetition per cell over 92 variables with no
confidence intervals, so the small non-monotonic wobble in DeepSeek's 5 → 7 → 10 range
(0.360, 0.349, 0.356) is well within what noise could produce. The Qwen 7 → 10 jump is
large enough to be interesting but has not been repeated.

## Output

- **Excel:** `iadopt-lab/experiments/output/shot-count-ablation-results.xlsx`
  - *Summary* — one row per model × shot count, with Close/Exact precision, recall and F1,
    valid rate, failed calls, latency and token usage
  - *Per-variable* — all 1,104 evaluations, so you can see where extra shots helped or hurt
  - *Shot pool* — the ten reserved variables, their order and which levels use them
  - *Configuration* — the fixed configuration per model and the official result it came from
- **Raw:** `iadopt-lab/experiments/output/shot-count-ablation-results.jsonl` (~15 MB)

Both are under `experiments/output/`, which is not the official `outputs/` directory.
