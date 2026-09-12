"""Twenty checks that a live campaign is behaving the way the finished ones did.

Run every ten minutes by `ops/healthcheck.sh`. Each check states the observed baseline
it is judged against, so a failure says what changed rather than only that something
did. Thresholds sit well outside the range both completed campaigns produced: these
detect breakage, not mediocrity, because a model scoring badly is a result and a
campaign that has stopped scoring at all is an incident.

Checks skip rather than fail while a sample is too small; a fresh campaign should not
alarm anyone in its first minute.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from conftest import BASELINE, MIN_SAMPLE

# ----------------------------------------------------------------- liveness and progress


def test_the_supervisor_process_is_alive(health):
    """Without it nothing restarts the campaign after a recoverable failure."""
    assert health["supervisor_alive"], (
        "No supervisor is running. Restart with ops/run-campaign.sh; completed work is kept.")


def test_work_is_actually_in_flight(health):
    """Leases are the only proof that requests are being made right now.

    A campaign row can say `running` for days after its process died, so liveness has to
    be read from leases, not from the state column.
    """
    if health["settled"] >= health["total"]:
        pytest.skip("campaign has finished; no leases expected")
    assert health["leases"] > 0, (
        f"No task is leased, yet {health['total'] - health['settled']:,} remain. "
        f"Campaign state says '{health['campaign_state']}', which can be stale.")


def test_settled_work_advanced_since_the_previous_check(health):
    """Ten minutes of a healthy campaign always settles more tasks.

    This is the check that would have caught the whitespace deadlock: the supervisor was
    alive and retrying, the provider was ready, and nothing moved for two hours.
    """
    previous = health["previous"]
    if previous is None:
        pytest.skip("first run for this campaign; nothing to compare against")
    if previous["settled"] >= health["total"]:
        pytest.skip("already complete at the previous check")
    # Time-gated, because the intended cadence is ten minutes: comparing against a
    # snapshot taken seconds ago reports "no progress" for a campaign that is simply
    # mid-task, and a health check that cries wolf when run twice is worse than none.
    minutes = (datetime.now(UTC)
               - datetime.fromisoformat(previous["checked_at"])).total_seconds() / 60
    if minutes < 5:
        pytest.skip(f"only {minutes:.1f} min since the last check; needs 5")
    assert health["settled"] > previous["settled"], (
        f"No progress since {previous['checked_at']}: still {health['settled']:,}"
        f"/{health['total']:,}. Supervisor alive={health['supervisor_alive']}, "
        f"provider={health['provider']}, failure streak={health['failure_streak']}. "
        f"Read .runtime/campaign/attempt-*.log.")


def test_throughput_is_in_a_workable_band(health):
    """Far below the observed rate means something is throttling or retrying hard.

    Both completed campaigns sustained well over 1,000 attempts/hour at concurrency 8.
    """
    previous = health["previous"]
    if previous is None or health["settled"] >= health["total"]:
        pytest.skip("needs a previous check and unfinished work")
    minutes = (datetime.now(UTC) - datetime.fromisoformat(previous["checked_at"])).total_seconds() / 60
    if minutes < 5:
        pytest.skip(f"only {minutes:.1f} min since the last check")
    if minutes > 45:
        # The baseline predates a gap - the checks were paused, or the campaign was. A
        # rate averaged across that reports a stall that already ended; actual stoppage
        # is what the liveness and progress checks are for.
        pytest.skip(f"baseline is {minutes:.0f} min old and spans a stop; rate meaningless")
    rate = (health["settled"] - previous["settled"]) / minutes * 60
    assert rate >= 120, (
        f"Only {rate:.0f} tasks/hour over the last {minutes:.0f} min; both completed "
        f"campaigns sustained >1,000/hour. Check provider state and latency.")


def test_the_supervisor_is_not_stuck_in_a_failure_streak(health):
    """Repeated fast failures mean a deterministic fault no retry can clear."""
    assert health["failure_streak"] < 3, (
        f"{health['failure_streak']} consecutive supervisor failures; backoff is growing "
        f"toward 30 min. The reason is one line in .runtime/campaign/attempt-*.log.")


# ------------------------------------------------------------------------ provider health


def test_the_provider_is_not_paused(health):
    """A paused provider blocks every queued task (D-046)."""
    paused = [name for name, state in health["provider"].items() if state in {"paused", "failed"}]
    assert not paused, (
        f"Provider(s) {paused} are not serving. Reconciliation releases a pause on the "
        f"next resume; if it returns immediately the deployment is genuinely unhealthy.")


def test_pauses_are_not_recurring(health):
    """One pause self-heals; a stream of them is a deployment problem."""
    assert health["pauses_recent"] <= 3, (
        f"The provider has paused {health['pauses_recent']} times in the last hour "
        f"({health['pause_events']} over the whole campaign). Recovery is working, but "
        f"the underlying cause is not going away.")


def test_the_http_error_rate_matches_the_observed_band(health):
    """Baseline: 0.0 on PSNC, 0.046 on OpenRouter, which was mostly upstream 429s."""
    if health["responses"] < MIN_SAMPLE:
        pytest.skip(f"only {health['responses']} responses so far")
    assert health["http_error_rate"] <= 0.20, (
        f"{health['http_error_rate']:.1%} of responses are HTTP >=400, against a "
        f"{max(BASELINE['http_error_rate']):.1%} worst case in the completed campaigns.")


def test_latency_has_not_collapsed_into_timeouts(health):
    """The timeout is 120s; a mean anywhere near it means the run is barely moving."""
    if health["responses"] < MIN_SAMPLE:
        pytest.skip(f"only {health['responses']} responses so far")
    assert health["latency_mean"] <= 60, (
        f"Mean latency {health['latency_mean']:.1f}s against a 120s timeout "
        f"(completed campaigns: {BASELINE['latency_mean_s'][0]}-"
        f"{BASELINE['latency_mean_s'][1]}s, reasoning off). Timeouts are imminent.")


# -------------------------------------------------------------------------- output sanity


def test_the_valid_json_rate_has_not_collapsed(health):
    """Baseline 0.953 (PSNC) and 0.820 (OpenRouter). Below 0.50 is a prompt or model fault."""
    if health["validations"] < MIN_SAMPLE:
        pytest.skip(f"only {health['validations']} validations so far")
    assert health["valid_rate"] >= 0.50, (
        f"Only {health['valid_rate']:.1%} of answers validate, against "
        f"{min(BASELINE['valid_json_rate']):.1%} in the worst completed campaign.")


def test_every_model_is_producing_valid_output(health):
    """A single collapsed model is invisible in the campaign-wide average.

    In `844df00e` one model's scattered failures cost 33 of its 36 configurations while
    the overall numbers still looked healthy.
    """
    sampled = {model: stats for model, stats in health["per_model"].items()
               if stats["responses"] >= MIN_SAMPLE}
    if not sampled:
        pytest.skip("no model has enough validations yet")
    # 0.05, not something near the campaign average: a weak model returning malformed
    # JSON much of the time is a *result* this experiment exists to measure, and
    # meta-llama/llama-3.1-8b-instruct legitimately sits near 0.21 here. Only a model
    # producing essentially nothing usable is an incident.
    broken = {model: round(stats["valid"], 3) for model, stats in sampled.items()
              if stats["valid"] < 0.05}
    assert not broken, (
        f"These models have stopped producing usable output entirely: {broken}. "
        f"All models: { {m: round(s['valid'], 2) for m, s in sampled.items()} }")


def test_truncation_stays_negligible(health):
    """Both completed campaigns truncated zero times; the ceiling is 8,000 tokens.

    A truncated answer is a terminal failure that cannot be retried, so a rising rate
    directly erodes how many configurations end up rankable.
    """
    if health["responses"] < MIN_SAMPLE:
        pytest.skip(f"only {health['responses']} responses so far")
    assert health["truncation_rate"] <= 0.02, (
        f"{health['truncation_rate']:.2%} of responses hit the output ceiling, against "
        f"0% in both completed campaigns. The 8,000-token ceiling may be too low here.")


def test_predictions_fill_the_two_universal_fields(health):
    """Every gold decomposition has hasProperty and hasObjectOfInterest, all 97 of them.

    Predictions filled them 94.8-98.5% of the time. A collapse means the model has
    stopped decomposing and is emitting near-empty objects that still pass the schema.
    """
    if health["predictions"] < MIN_SAMPLE:
        pytest.skip(f"only {health['predictions']} predictions so far")
    assert health["property_filled"] >= 0.70 and health["object_filled"] >= 0.70, (
        f"hasProperty filled {health['property_filled']:.1%}, hasObjectOfInterest "
        f"{health['object_filled']:.1%}; both were >=93.6% in the completed campaigns.")


def test_empty_predictions_stay_rare(health):
    """Baseline 0.5% (PSNC) and 4.5% (OpenRouter) after three invalid attempts."""
    if health["predictions"] < MIN_SAMPLE:
        pytest.skip(f"only {health['predictions']} predictions so far")
    assert health["empty_rate"] <= 0.25, (
        f"{health['empty_rate']:.1%} of predictions are terminal-invalid (scored empty), "
        f"against {max(BASELINE['empty_prediction_rate']):.1%} in the worst completed run.")


def test_scores_are_in_range_and_not_degenerate(health):
    """F1 must stay in [0,1], and an all-zero distribution means scoring is broken.

    Zero-F1 share was 37.5% and 47.0%; universally zero would mean gold and prediction
    are no longer being compared to each other at all.
    """
    if health["scored"] < MIN_SAMPLE:
        pytest.skip(f"only {health['scored']} evaluations so far")
    assert 0.0 <= health["f1_min"] and health["f1_max"] <= 1.0, (
        f"Close F1 outside [0,1]: min {health['f1_min']}, max {health['f1_max']}.")
    assert health["zero_f1_share"] <= 0.90, (
        f"{health['zero_f1_share']:.1%} of evaluations score exactly zero, against "
        f"{max(BASELINE['zero_f1_share']):.1%} in the completed campaigns.")


# ----------------------------------------------------------------------------------- cost


def test_spend_is_below_the_cap(health):
    """Reaching the cap aborts the campaign, so this must never be a surprise."""
    if not health["cap"]:
        pytest.skip("this campaign has no monetary cap")
    assert health["spent"] < health["cap"] * 0.90, (
        f"${health['spent']:.2f} of ${health['cap']:.0f} spent (>90% of cap). "
        f"The campaign aborts at the cap.")


def test_cost_per_task_matches_the_estimate(health):
    """Measured $0.000197/task on OpenRouter; the estimate assumes $13.71 for 17,460."""
    if health["complete"] < MIN_SAMPLE:
        pytest.skip(f"only {health['complete']} tasks complete")
    per_task = health["spent"] / health["complete"]
    assert per_task <= 0.0040, (
        f"${per_task:.5f} per task, against ${BASELINE['cost_per_task_usd'][1]:.6f} "
        f"measured and ${13.71 / 17460:.6f} estimated. Token usage may be running away.")


def test_the_projected_total_stays_under_the_cap(health):
    """Extrapolating current spend must not exceed the cap before the last task."""
    if health["complete"] < MIN_SAMPLE or not health["cap"]:
        pytest.skip("not enough completed tasks, or no cap")
    projected = health["spent"] / health["complete"] * health["total"]
    assert projected <= health["cap"], (
        f"Projected total ${projected:.2f} exceeds the ${health['cap']:.0f} cap; the run "
        f"would abort around {health['cap'] / (health['spent'] / health['complete']):,.0f} tasks.")


# ------------------------------------------------------------------------------ integrity


def test_permanent_losses_stay_below_the_rankability_line(health):
    """Lost tasks are never retried and destroy whole configurations.

    In `844df00e`, 144 losses spread across one model left 3 of its 36 configurations
    rankable. 1% of the population is the point where that starts to bite.
    """
    if health["settled"] < MIN_SAMPLE:
        pytest.skip(f"only {health['settled']} tasks settled")
    share = health["lost"] / health["settled"]
    assert share <= 0.01, (
        f"{health['lost']} of {health['settled']:,} settled tasks are permanently lost "
        f"({share:.2%}): {health['ambiguous']} ambiguous, {health['failed']} failed. "
        f"Scattered losses cost whole configurations, not just tasks.")


def test_every_model_is_progressing(health):
    """Provider-fair scheduling should keep all five models moving together.

    One model starved of work distorts the comparison the experiment exists to make.
    """
    if health["complete"] < MIN_SAMPLE * 5:
        pytest.skip(f"only {health['complete']} tasks complete; too early to compare models")
    # No assertion on the model count: it is read from this campaign's own resolved runs,
    # so comparing it with a hard-coded 5 only asserted that this campaign is the one I
    # happened to write the number for - it fired on a perfectly healthy 3-model run.
    counts = {model: stats["complete"] for model, stats in health["per_model"].items()}
    stalled = {model: n for model, n in counts.items() if n == 0}
    assert not stalled, (
        f"These models have completed nothing while others progressed: {sorted(stalled)}. "
        f"All models: {counts}")
    # 0.15, measured: reasoning-on models run 7-34x slower per call (qwen3-32b averages
    # 44s against ministral's 1.3s), yet provider-fair scheduling still held the observed
    # min/max completion ratio at 0.51. Anything under 0.15 is starvation, not slowness.
    assert min(counts.values()) >= max(counts.values()) * 0.15, (
        f"Models are progressing very unevenly: {counts}. One is being starved or is "
        f"failing far more often than the rest.")
