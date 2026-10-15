"""The whole Jev run, offline: a fake Jev, the real committed baseline, every gate exercised."""

from __future__ import annotations

import csv
import functools
import json
import shutil
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from iadopt_linking import experiment
from iadopt_linking.artifacts import file_sha256
from iadopt_linking.cli import main
from iadopt_linking.errors import RunError, UsageError
from iadopt_linking.jev import MODEL, NONE_KEY, PASSES, PRIMARY_PASS, encode
from iadopt_linking.jev_experiment import (
    PUBLISHED_JEV,
    JevPaths,
    jev_evidence_identity,
    run_jev,
)
from iadopt_linking.jev_health import REQUIRED_JEV_CHECKS
from iadopt_linking.openrouter import DecisionsClient, Exchange
from iadopt_linking.provenance import now_utc

BASELINE = Path(__file__).resolve().parents[1] / "results" / "baseline-20261014"
PILOT = ["EWCT", "C8_CWDFraction"]
SNAPSHOT = "typesafe/jev-1.13-20260917"
KEY = "sk-or-v1-NEVER-WRITE-THIS-TO-DISK-9876543210"
CACHED_GOLD = functools.cache(experiment.load_gold_links)


@pytest.fixture(autouse=True)
def read_the_corpus_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(experiment, "load_gold_links", CACHED_GOLD)


def respond(body: dict[str, Any], *, model: str = SNAPSHOT, cost_scale: float = 1.0) -> str:
    """Answer a request the way a plausible Jev would: favour the candidate named like the term."""
    term = body["state"]["term"].lower()
    questions = body["questions"]
    answers: dict[str, Any] = {}
    if "entity" in questions:
        options = list(questions["entity"]["criteria"])
        real = [o for o in options if o != NONE_KEY]
        named = [o for o in real
                 if questions["entity"]["criteria"][o]["label"].lower() == term]
        favourite = (named or real)[0]
        probabilities = {o: 0.0 for o in options}
        rest = 0.3 / max(len(real) - 1, 1)
        for o in real:
            probabilities[o] = round(0.68 if o == favourite else rest, 2)
        probabilities[NONE_KEY] = 0.02
        top = max(probabilities, key=probabilities.get)
        n = len(options)
        answers["entity"] = {"type": "choice", "choice": top, "probabilities": probabilities,
                             "confidence": round((probabilities[top] - 1 / n) / (1 - 1 / n), 2)}
    else:
        for key, question in questions.items():
            same = question["instructions"]["candidate"]["label"].lower() == term
            answers[key] = {"type": "noul", "noul": 0.9 if same else 0.1}
    tokens = len(encode(body)) // 2
    return json.dumps({"id": "gen-dec-1", "model": model, "provider": "TypeSafe",
                       "answers": answers,
                       "usage": {"input_tokens": tokens, "output_tokens": 12,
                                 "cost": tokens * 4.2e-8 * cost_scale}})


class FakeJev:
    """A client that answers every request in process."""

    def __init__(self, **options: Any) -> None:
        self.options = options
        self.calls: list[dict[str, Any]] = []

    def settings(self) -> dict[str, Any]:
        return {"url": "fake"}

    def post(self, body: dict[str, Any]) -> Exchange:
        self.calls.append(body)
        return Exchange(200, respond(body, **self.options), None, 1, (), 0.01, 0.01, now_utc())


class Refusing(FakeJev):
    """Answers with a status instead of a result."""

    def __init__(self, status: int) -> None:
        super().__init__()
        self.status = status

    def post(self, body: dict[str, Any]) -> Exchange:
        self.calls.append(body)
        return Exchange(self.status, None, f"HTTP {self.status}: no", 1, (), 0.01, 0.01, now_utc())


class FlakyFor(FakeJev):
    """Fails the first N requests of one pass as an exhausted 503, then behaves."""

    def __init__(self, failures: int) -> None:
        super().__init__()
        self.failures = failures

    def post(self, body: dict[str, Any]) -> Exchange:
        self.calls.append(body)
        if self.failures > 0:
            self.failures -= 1
            return Exchange(503, None, "HTTP 503: busy", 4, ("HTTP 503; waited 2 s",) * 3,
                            1.0, 0.1, now_utc())
        return Exchange(200, respond(body), None, 1, (), 0.01, 0.01, now_utc())


def execute(out: Path, client: Any | None = None, **options: Any) -> dict[str, Any]:
    settings: dict[str, Any] = {"baseline_dir": BASELINE, "max_cost_usd": "1", "only": PILOT,
                                "workers": 3, "resamples": 100}
    settings.update(options)
    fake = client if client is not None else FakeJev()
    return run_jev(out, client_factory=lambda: fake, **settings)


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def stored(out: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (out / "responses.jsonl").read_text().splitlines()]


def test_a_pilot_run_writes_every_artifact_and_passes_every_gate(tmp_path: Path) -> None:
    out = tmp_path / "jev"
    fake = FakeJev()
    manifest = execute(out, fake)
    assert [c["name"] for c in manifest["checks"]] == list(REQUIRED_JEV_CHECKS)
    assert all(c["passed"] is True for c in manifest["checks"])
    assert set(manifest["files"]) == set(PUBLISHED_JEV)
    for name, expected in manifest["files"].items():
        assert file_sha256(out / name) == expected
    assert (out / "manifest.json").is_file() and manifest["pilot_variables"] == sorted(PILOT)
    assert manifest["model"]["requested"] == MODEL and manifest["model"]["served"] == [SNAPSHOT]
    assert len(stored(out)) == len(fake.calls) == manifest["spend"]["requests"] > 0
    assert manifest["evidence_code_sha256"] == jev_evidence_identity()
    journal = read(out / "provenance.json")
    assert journal["invocations"][-1]["outcome"] == "completed" and "device_check" not in journal


def test_every_pass_asks_every_slot_that_has_candidates(tmp_path: Path) -> None:
    out = tmp_path / "jev"
    fake = FakeJev()
    manifest = execute(out, fake)
    plan = read(out / "plan.json")
    slots = manifest["population"]["slots"] - plan["asked"]["empty_pools"]
    assert len(fake.calls) == slots * len(PASSES) == plan["tasks"]
    per_pass = {}
    for record in stored(out):
        per_pass[record["pass"]] = per_pass.get(record["pass"], 0) + 1
    assert per_pass == {spec.name: slots for spec in PASSES}


def test_the_tables_for_later_analysis_have_one_row_per_slot_pass_and_candidate(
        tmp_path: Path) -> None:
    out = tmp_path / "jev"
    execute(out)
    plan = read(out / "plan.json")
    slots = read(out / "manifest.json")["population"]["slots"]
    outcome_rows = (out / "outcomes.csv").read_text().splitlines()
    assert len(outcome_rows) == 1 + slots
    assert outcome_rows[0].startswith("slot_id,") and "jev_selected_probability" in outcome_rows[0]
    assert "query_key" in outcome_rows[0], "the grouping key of the term-level analysis is published"
    assert "qwen3_correct" in outcome_rows[0] and "jev_confidence" in outcome_rows[0]
    confidence_rows = (out / "confidence.csv").read_text().splitlines()
    assert len(confidence_rows) == 1 + slots * len(PASSES)
    asked = sum(1 for r in stored(out) if r["pass"] == PRIMARY_PASS)
    probabilities = (out / "probabilities.csv").read_text().splitlines()[1:]
    shown = sum(len(r["shown"]) for r in stored(out))
    assert len(probabilities) == shown and asked == slots - plan["asked"]["empty_pools"]


def test_the_report_has_the_baselines_sections_and_jevs_own(tmp_path: Path) -> None:
    out = tmp_path / "jev"
    execute(out)
    report = (out / "report.md").read_text()
    for heading in ("## Result", "## Jev against Qwen3 on the reachable slots",
                    "## Where the misses come from", "## Diagnostics", "## Which source wins",
                    "## Passes, order and chance", "## Jev's own `none_of_these`",
                    "## What the confidence is worth", "## Run health", "## Notes"):
        assert heading in report
    assert "## Slots only Jev got right" in (out / "comparison.md").read_text()


def test_a_second_run_asks_nothing_and_changes_no_evidence(tmp_path: Path) -> None:
    out = tmp_path / "jev"
    execute(out)
    before = {name: file_sha256(out / name)
              for name in ("responses.jsonl", "plan.json", "summary.json", "outcomes.csv",
                           "confidence.csv", "probabilities.csv")}

    def forbidden() -> Any:
        raise AssertionError("a finished folder must not build a client")

    manifest = run_jev(out, baseline_dir=BASELINE, max_cost_usd="1", only=PILOT, resamples=100,
                       client_factory=forbidden)
    after = {name: file_sha256(out / name) for name in before}
    assert before == after
    assert len(read(out / "provenance.json")["invocations"]) == 2
    assert all(c["passed"] for c in manifest["checks"])


def test_the_forced_choice_ignores_a_none_of_these_that_wins(tmp_path: Path) -> None:
    class Refuser(FakeJev):
        def post(self, body: dict[str, Any]) -> Exchange:
            self.calls.append(body)
            response = json.loads(respond(body))
            answer = response["answers"].get("entity")
            if answer is not None:
                answer["probabilities"] = {o: (0.98 if o == NONE_KEY else 0.0)
                                           for o in answer["probabilities"]}
                answer["probabilities"]["c01"] = 0.02
                answer["choice"], answer["confidence"] = NONE_KEY, 0.96
            return Exchange(200, json.dumps(response), None, 1, (), 0.01, 0.01, now_utc())

    out = tmp_path / "jev"
    execute(out, Refuser())
    rows = list(csv.DictReader((out / "outcomes.csv").open(encoding="utf-8")))
    asked = [r for r in rows if int(r["pool_size"]) > 0]
    assert asked and all(r["jev_own_choice"] == NONE_KEY for r in asked), "the refusal is recorded"
    # ... yet every slot with candidates still has a selection, because the choice is forced
    assert all(r["jev_selected_local_id"] for r in asked)
    abstention = read(out / "summary.json")["analysis"]["abstention"]
    assert abstention["answered"]["numerator"] == 0 and abstention["asked"] == len(asked)


def test_requests_that_fail_stop_the_run_and_the_next_run_asks_only_those(tmp_path: Path) -> None:
    out = tmp_path / "jev"
    flaky = FlakyFor(failures=3)
    with pytest.raises(RunError, match="3 of .* requests have no usable answer"):
        execute(out, flaky, workers=1)
    assert not (out / "manifest.json").exists()
    failed = [r for r in stored(out) if r["status"] != 200]
    assert len(failed) == 3 and all(r["response_text"] is None for r in failed)
    healthy = FakeJev()
    execute(out, healthy)
    assert len(healthy.calls) == 3
    assert (out / "manifest.json").is_file()
    journal = read(out / "provenance.json")
    assert journal["invocations"][0]["outcome"].startswith("stopped: RunError")


def test_a_refused_credential_stops_at_once_and_keeps_what_was_paid_for(tmp_path: Path) -> None:
    out = tmp_path / "jev"
    refusing = Refusing(402)
    with pytest.raises(RunError, match="refused the credential or the account"):
        execute(out, refusing, workers=1)
    assert len(refusing.calls) <= 2 and not (out / "manifest.json").exists()
    asking = read(out / "provenance.json")["invocations"][0]["asking"]
    assert asking["failed"] >= 1, "a stopped invocation still says what it did"


def test_a_cap_below_the_upper_bound_estimate_is_refused_before_any_call(tmp_path: Path) -> None:
    out = tmp_path / "jev"
    fake = FakeJev()
    with pytest.raises(RunError, match="exceeds the cap"):
        execute(out, fake, max_cost_usd="0.0000001")
    assert fake.calls == []


def test_a_billed_run_without_a_cap_is_refused(tmp_path: Path) -> None:
    fake = FakeJev()
    with pytest.raises(RunError, match="explicit spending cap"):
        execute(tmp_path / "jev", fake, max_cost_usd=None)
    assert fake.calls == []


@pytest.mark.parametrize("cap", ["0", "-1", "abc", "NaN"])
def test_an_unusable_cap_is_refused(tmp_path: Path, cap: str) -> None:
    with pytest.raises(RunError, match="cap"):
        execute(tmp_path / "jev", FakeJev(), max_cost_usd=cap)


def test_spend_above_the_cap_stops_the_run_and_stores_the_answers_already_paid_for(
        tmp_path: Path) -> None:
    out = tmp_path / "jev"
    dear = FakeJev(cost_scale=1000.0)       # the service charges far more than the estimate
    with pytest.raises(RunError, match="passed the cap"):
        execute(out, dear, max_cost_usd="0.05", workers=1)
    assert 0 < len(stored(out)) < read(out / "plan.json")["tasks"]


@pytest.mark.parametrize(("field", "value"), [
    ("prompt", {"fingerprint": "tampered"}),
    ("passes", [["choice-a1", "choice", "shuffle-a"]]),
    ("baseline", {"folder": "other"}),
    ("only", ["EWCT"]),
    ("evidence_code_sha256", "0" * 64),
    ("analysis", {}),
])
def test_a_folder_that_does_not_match_this_run_is_not_resumed(
        tmp_path: Path, field: str, value: Any) -> None:
    out = tmp_path / "jev"
    execute(out)
    plan = read(out / "plan.json")
    plan[field] = value
    (out / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(RunError, match="resuming would mix evidence"):
        execute(out)


def test_evidence_without_a_plan_or_a_journal_is_refused(tmp_path: Path) -> None:
    out = tmp_path / "jev"
    execute(out)
    (out / "plan.json").unlink()
    with pytest.raises(RunError, match="no plan.json"):
        execute(out)
    other = tmp_path / "other"
    execute(other)
    (other / "provenance.json").unlink()
    with pytest.raises(RunError, match="no provenance.json"):
        execute(other)


def test_a_changed_response_is_caught_by_its_hash(tmp_path: Path) -> None:
    out = tmp_path / "jev"
    execute(out)
    lines = (out / "responses.jsonl").read_text().splitlines()
    record = json.loads(lines[0])
    record["response_text"] = record["response_text"].replace("0.9", "0.1", 1) + " "
    lines[0] = json.dumps(record)
    (out / "responses.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(RunError, match="do not match their hash"):
        execute(out, FakeJev())


def test_two_model_snapshots_in_one_run_are_refused(tmp_path: Path) -> None:
    class Drifting(FakeJev):
        def post(self, body: dict[str, Any]) -> Exchange:
            self.calls.append(body)
            model = SNAPSHOT if len(self.calls) % 2 else "typesafe/jev-1.13-20261001"
            return Exchange(200, respond(body, model=model), None, 1, (), 0.01, 0.01, now_utc())

    with pytest.raises(RunError, match="one model snapshot served"):
        execute(tmp_path / "jev", Drifting())


def test_a_different_model_than_requested_is_refused(tmp_path: Path) -> None:
    with pytest.raises(RunError, match="one model snapshot served"):
        execute(tmp_path / "jev", FakeJev(model="typesafe/jev-1.14-20261014"))


def test_a_response_that_is_not_a_valid_answer_is_stored_and_stops_the_run(tmp_path: Path) -> None:
    class Garbled(FakeJev):
        def post(self, body: dict[str, Any]) -> Exchange:
            self.calls.append(body)
            return Exchange(200, '{"model": "x"}', None, 1, (), 0.01, 0.01, now_utc())

    out = tmp_path / "jev"
    garbled = Garbled()
    with pytest.raises(RunError, match="requests all failed"):
        execute(out, garbled, workers=1)
    records = stored(out)
    assert len(records) == len(garbled.calls) < read(out / "plan.json")["tasks"]
    assert all(r["parse_error"] and r["response_text"] for r in records)


def test_a_baseline_that_changed_after_its_run_is_refused(tmp_path: Path) -> None:
    copy = tmp_path / "baseline"
    shutil.copytree(BASELINE, copy)
    with (copy / "scores.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(RunError, match="changed after its run"):
        run_jev(tmp_path / "jev", baseline_dir=copy, max_cost_usd="1", only=PILOT,
                client_factory=FakeJev)


def test_an_unfinished_baseline_is_refused(tmp_path: Path) -> None:
    copy = tmp_path / "baseline"
    shutil.copytree(BASELINE, copy)
    (copy / "manifest.json").unlink()
    with pytest.raises(RunError, match="not a finished baseline"):
        run_jev(tmp_path / "jev", baseline_dir=copy, max_cost_usd="1", only=PILOT,
                client_factory=FakeJev)


def test_the_credential_never_reaches_a_file(tmp_path: Path) -> None:
    sent: list[dict[str, str]] = []

    def transport(url: str, headers: Any, body: bytes, timeout: float) -> tuple[int, dict, bytes]:
        sent.append(dict(headers))
        return 200, {}, respond(json.loads(body)).encode()

    out = tmp_path / "jev"
    run_jev(out, baseline_dir=BASELINE, max_cost_usd="1", only=PILOT, resamples=100,
            client_factory=lambda: DecisionsClient(KEY, transport=transport))
    assert sent and all(h["Authorization"] == f"Bearer {KEY}" for h in sent)
    for path in out.iterdir():
        assert KEY not in path.read_text(encoding="utf-8"), f"the key leaked into {path.name}"
    assert "Authorization" not in (out / "provenance.json").read_text()


def test_the_client_for_a_real_run_reads_its_key_from_the_credential_file(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = tmp_path / ".env"
    env.write_text(f"OPENROUTER_API_KEY={KEY}\n", encoding="utf-8")
    seen: list[str] = []

    def transport(url: str, headers: Any, body: bytes, timeout: float) -> tuple[int, dict, bytes]:
        seen.append(headers["Authorization"])
        return 200, {}, respond(json.loads(body)).encode()

    monkeypatch.setattr("iadopt_linking.jev_experiment.DecisionsClient",
                        lambda key: DecisionsClient(key, transport=transport))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    run_jev(tmp_path / "jev", baseline_dir=BASELINE, max_cost_usd="1", only=PILOT, resamples=100,
            env_file=str(env))
    assert set(seen) == {f"Bearer {KEY}"}


def test_a_missing_credential_stops_before_any_request(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    empty = tmp_path / "empty.env"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(RunError, match="OPENROUTER_API_KEY is not set"):
        run_jev(tmp_path / "jev", baseline_dir=BASELINE, max_cost_usd="1", only=PILOT,
                env_file=str(empty))


def test_workers_change_the_order_responses_arrive_in_but_not_a_single_number(
        tmp_path: Path) -> None:
    one, four = tmp_path / "one", tmp_path / "four"
    execute(one, workers=1)
    execute(four, workers=4)
    assert read(one / "summary.json") == read(four / "summary.json")
    assert (one / "outcomes.csv").read_text() == (four / "outcomes.csv").read_text()


def test_the_plan_command_prints_requests_and_calls_nothing(
        capsys: pytest.CaptureFixture) -> None:
    assert main(["jev-plan", "--baseline", str(BASELINE), "--only", ",".join(PILOT)]) == 0
    text = capsys.readouterr().out
    assert "estimate:" in text and "first choice request" in text and "first noul request" in text
    assert KEY not in text and "Authorization" not in text


def test_an_unknown_pass_is_a_usage_error(capsys: pytest.CaptureFixture) -> None:
    with pytest.raises(SystemExit):
        main(["jev-plan", "--baseline", str(BASELINE), "--passes", "choice-zzz"])
    assert "choice-zzz" in capsys.readouterr().err


def test_jev_paths_know_when_a_folder_is_new(tmp_path: Path) -> None:
    paths = JevPaths(tmp_path)
    assert not paths.has_evidence()
    (tmp_path / "responses.jsonl").write_text("{}\n")
    assert paths.has_evidence()


def test_passes_that_cannot_make_a_result_are_refused_before_any_money_is_spent(
        tmp_path: Path) -> None:
    fake = FakeJev()
    with pytest.raises(UsageError, match="primary pass"):
        execute(tmp_path / "a", fake, passes=["choice-a2-reversed"])
    with pytest.raises(UsageError, match="duplicates"):
        execute(tmp_path / "b", fake, passes=["choice-a1", "choice-a1"])
    assert fake.calls == [] and not (tmp_path / "a" / "responses.jsonl").exists()


def test_a_rejected_response_is_still_paid_for_and_counted(tmp_path: Path) -> None:
    class Inconsistent(FakeJev):
        """Names an option that does not have the highest probability in the first answer."""

        def post(self, body: dict[str, Any]) -> Exchange:
            self.calls.append(body)
            text = respond(body)
            if len(self.calls) == 1:
                data = json.loads(text)
                answer = data["answers"]["entity"]
                top = max(answer["probabilities"], key=answer["probabilities"].get)
                answer["choice"] = next(o for o in answer["probabilities"]
                                        if answer["probabilities"][o] < answer["probabilities"][top])
                text = json.dumps(data)
            return Exchange(200, text, None, 1, (), 0.01, 0.01, now_utc())

    out = tmp_path / "jev"
    with pytest.raises(RunError, match="1 of .* requests have no usable answer"):
        execute(out, Inconsistent(), workers=1)
    manifest = execute(out)
    records = stored(out)
    total = sum(Decimal(repr(json.loads(r["response_text"])["usage"]["cost"])).quantize(
        Decimal("1e-12")).normalize() for r in records if r["response_text"] is not None)
    spend = manifest["spend"]
    assert Decimal(spend["cost_usd"]) == total, "the rejected response was billed too"
    assert (spend["rejected_responses"], spend["responses_stored"], spend["requests"]) == (
        1, len(records), len(records) - 1)
    assert manifest["evidence_period"]["first_request"] == min(r["sent_at"] for r in records)
    answered = next(c for c in manifest["checks"] if c["name"] == "every request answered")
    assert "1 stored responses were rejected" in answered["detail"]
    assert any("rejected by the validity rule" in note for note in manifest["notes"])


def test_an_unexpected_exception_in_the_client_loses_no_answer_already_paid_for(
        tmp_path: Path) -> None:
    class Exploding(FakeJev):
        def post(self, body: dict[str, Any]) -> Exchange:
            self.calls.append(body)
            if len(self.calls) == 3:
                raise RuntimeError("detail that must not be stored")
            return Exchange(200, respond(body), None, 1, (), 0.01, 0.01, now_utc())

    out = tmp_path / "jev"
    exploding = Exploding()
    with pytest.raises(RunError, match="1 of .* requests have no usable answer"):
        execute(out, exploding, workers=1)
    records = stored(out)
    assert len(records) == len(exploding.calls) > 3
    failed = [r for r in records if r["status"] != 200]
    assert [r["error"] for r in failed] == ["unexpected RuntimeError while sending"]
    assert "detail that must not be stored" not in (out / "responses.jsonl").read_text()
    healthy = FakeJev()
    execute(out, healthy)
    assert len(healthy.calls) == 1


def test_every_request_that_needed_a_retry_is_counted(tmp_path: Path) -> None:
    class Slow(FakeJev):
        def post(self, body: dict[str, Any]) -> Exchange:
            self.calls.append(body)
            return Exchange(200, respond(body), None, 3,
                            ("HTTP 429, Retry-After 1; waited 1 s", "HTTP 503; waited 2 s"),
                            3.0, 0.1, now_utc())

    out = tmp_path / "jev"
    manifest = execute(out, Slow())
    requests = manifest["spend"]["requests"]
    assert manifest["spend"]["answered_after_retry"] == requests > 0
    assert manifest["spend"]["failed_attempts_repeated"] == 2 * requests
    answered = next(c for c in manifest["checks"] if c["name"] == "every request answered")
    assert f"{requests} needed an HTTP retry" in answered["detail"]
    assert read(out / "provenance.json")["invocations"][-1]["asking"]["retried"] == requests


def test_a_cost_the_arithmetic_cannot_hold_does_not_lose_the_other_answers(tmp_path: Path) -> None:
    class Absurd(FakeJev):
        def post(self, body: dict[str, Any]) -> Exchange:
            self.calls.append(body)
            text = respond(body)
            if len(self.calls) == 3:
                data = json.loads(text)
                data["usage"]["cost"] = 1e30
                text = json.dumps(data)
            return Exchange(200, text, None, 1, (), 0.01, 0.01, now_utc())

    out = tmp_path / "jev"
    absurd = Absurd()
    with pytest.raises(RunError, match="1 of .* requests have no usable answer"):
        execute(out, absurd, workers=1)
    records = stored(out)
    assert len(records) == len(absurd.calls) > 3
    flagged = [r for r in records if r["parse_error"]]
    assert len(flagged) == 1 and flagged[0]["response_text"] is not None
    assert "unexpected InvalidOperation while reading the response" in flagged[0]["parse_error"]
