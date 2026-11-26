# Pattern retrieval: results summary

**Exploratory side experiment, not an official result.** It belongs to `few-shot-selection/` and
appears in no official ranking, campaign or results table.

## Question

Does showing the model similar variables that are already decomposed improve its I-ADOPT
decompositions, and by how much? Do the choices around that (how many examples, which ones, in what
order) matter?

## What was compared

- 102 scientific variables, each with an expert decomposition into the six I-ADOPT parts.
- One model at one host: `qwen/qwen3.8-27b` through OpenRouter, reasoning off, 8-bit.
- Score: Close F1 from the frozen scorer (`iadopt-lab/src/iadopt_eval`), ten runs per setting.
- Retrieved examples come from the five corpus folds the target does not belong to, so a target
  never sees its own fold. Today's prompt keeps its fixed 25 examples whichever fold the target is
  in (a variable that is one of them is left out).
- Two settings are compared by the difference in pooled Close F1, with a 95% interval from
  resampling the variables. A difference counts as reliable only when that interval excludes zero.
- Examples are found by the cosine similarity of the definitions, using the embedding model the
  scorer uses. In every setting but the one listed as nearest first, the most similar example comes
  last in the prompt.

## Results

| Setting | Close F1 | SD over runs | vs today's prompt (95% interval) |
|---|---:|---:|---|
| Today's prompt: its own 25 examples | 0.593 | 0.012 | — |
| 10 nearest examples | 0.657 | 0.007 | +0.063 (+0.020, +0.107) |
| **25 nearest examples** | 0.678 | 0.011 | +0.084 (+0.045, +0.123) |
| 25 nearest, plus one re-ask of unusable answers | 0.681 | 0.007 | +0.088 (+0.049, +0.127) |
| 50 nearest examples | 0.683 | 0.011 | +0.089 (+0.051, +0.130) |
| All 85 other-fold examples | 0.684 | 0.014 | +0.091 (+0.053, +0.131) |
| 25 nearest by shared words (not embeddings) | 0.640 | 0.010 | +0.047 (+0.004, +0.089) |
| Best setting, chosen on the other five folds | 0.674 | — | +0.081 (+0.044, +0.119) |

The setting chosen on the other five folds scores 0.674, not 0.684, because 0.684 is the best of
eleven settings picked on the same variables; 0.674 is the fairer figure. The other four retrieval
variants (nearest first, today's 25 plus 10 nearest, spread out for diversity, picked by a draft's
labels) score 0.673, 0.683, 0.681, 0.684, none of them reliably better than 25 nearest. The one
setting that is reliably worse is the one that compares definitions by shared words
(-0.037, 95% interval -0.059 to -0.017 against 25 nearest).

By the corpus groups (25 example, 53 practice, 24 held-out variables) 25 nearest scores
0.723, 0.625 and 0.735.

Close F1 by part, 25 nearest against today's prompt:

| Part | Today's prompt | 25 nearest | Change |
|---|---:|---:|---:|
| Property | 0.697 | 0.741 | +0.043 |
| Object of interest | 0.658 | 0.720 | +0.061 |
| Matrix | 0.473 | 0.653 | +0.180 |
| Context object | 0.306 | 0.435 | +0.130 |
| Statistical modifier | 0.757 | 0.735 | -0.022 |
| Constraints | 0.497 | 0.603 | +0.106 |

Five of the six parts rise; the statistical modifier slips a little.

## What holds and what does not

- Retrieval beats today's fixed example set by +0.081 for the fairly chosen setting and by
  +0.084 for 25 nearest, both reliably.
- Choosing the setting on the other folds does not improve on 25 nearest: -0.003
  (-0.020 to +0.014).
- A re-ask of an unusable answer repaired 20 of 22 of them
  and raised the score by 0.004 (+0.001 to +0.007): what those
  answers cost, not an improvement of the method.
- In an earlier stage of the same experiment, given the experts' own answers to the six pattern
  decisions (property pattern, object shape, matrix, context object, statistical modifier, constraint
  types), the same model reached 0.747, from a baseline of 0.589 in that stage. Its own
  answers to those decisions added nothing. Most of the difference sits on the matrix and the
  constraint types.
- As a check announced before the run, though it uses the gold labels, the retrieval gain was split
  by whether the retrieved examples hold the target's exact gold labels. Among the 81 variables without
  a near-duplicate definition in another fold, the 44 whose 25 retrieved examples hold at least one
  gold label that today's 25 do not gained +0.154 (+0.094 to +0.215) with 25 nearest; the other 37
  gained -0.009 (-0.061 to +0.049). Retrieval helps mostly where the pool already holds the right
  labels.
- A Close F1 of 0.70 is not reached.

## Limits

- One model, one host, 8-bit, 102 variables of one corpus. No setting was tested on variables kept
  out of every decision, so these numbers describe this corpus and are not a claim about new variables.
- 21 of the 102 variables have a near-duplicate in another fold (a definition sharing half its words
  with it), which lifts every retrieval score.
- The score's acceptance threshold (cosine 0.80) has not been checked against human judgement.
- The plan, run log and reviews of this experiment are kept in the project's private record.

## Reproduce

From `iadopt-lab/`, with the lab's local embedding model in place and an OpenRouter key in the
repository's ignored `.env` (variable `OPENROUTER_API_KEY`):

    uv run python -I few-shot-selection/guided/rag_tuning.py --similarities
    uv run python -I few-shot-selection/guided/rag_tuning.py --run
    uv run python -I few-shot-selection/guided/rag_tuning.py --score
    uv run python -I few-shot-selection/guided/recompute_tuning.py

A run makes paid API calls. The stored answers behind these figures are not in the repository, so the
figures can be recomputed only on a machine that has them.
