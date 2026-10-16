"""The whole PSNC chat run, offline: a fake model, the real committed baseline, every gate exercised."""

from __future__ import annotations

import csv
import functools
import hashlib
import inspect
import json
import math
import re
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from iadopt_linking import experiment
from iadopt_linking.artifacts import file_sha256
from iadopt_linking.chat_choice import (
    CHAT_PASSES,
    MODEL,
    build_tasks,
    label_to_option,
    option_label,
)
from iadopt_linking.chat_experiment import (
    PSNC,
    PSNC_EVIDENCE_FILES,
    REQUIRED_CHAT_CHECKS,
    chat_checks,
    chat_notes,
    extra_analysis,
    load_reference,
    run_chat,
    tokens_of,
)
from iadopt_linking.chat_redact import RedactingChatClient
from iadopt_linking.cli import main
from iadopt_linking.errors import RunError, UsageError
from iadopt_linking.jev import NONE_KEY, encode
from iadopt_linking.jev_experiment import JEV_EVIDENCE_FILES, PUBLISHED_JEV, JevPaths, ask_phase
from iadopt_linking.jev_health import read_answers
from iadopt_linking.jev_inputs import load_baseline
from iadopt_linking.openrouter import Exchange
from iadopt_linking.provenance import now_utc
from iadopt_linking.recorder import JsonlStore, latest

RESULTS = Path(__file__).resolve().parents[1] / "results"
BASELINE = RESULTS / "baseline-20261014"
JEV_RUN = RESULTS / "jev-20261015"
PILOT = ["EWCT", "C8_CWDFraction"]
BUILD = "vllm-test-0.1-abcdef"
KEY = "psnc-NEVER-WRITE-THIS-TO-DISK-9876543210"
CACHED_GOLD = functools.cache(experiment.load_gold_links)


@pytest.fixture(autouse=True)
def read_the_corpus_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(experiment, "load_gold_links", CACHED_GOLD)


def respond(body: dict[str, Any], *, model: str = MODEL, build: str | None = BUILD,
            favourite: float = 0.7, none_wins: bool = False, message: dict | None = None,
            finish: str = "length", label_mass: float = 0.99) -> str:
    """Answer a request the way a plausible chat model would: favour the candidate named like the term."""
    user = body["messages"][1]["content"]
    term = user.splitlines()[0].removeprefix("term: ").lower()
    block = user.split("candidates:\n", 1)[1].split("\n\n", 1)[0]
    options = []
    for line in block.splitlines():
        label, _, rest = line.partition(". ")
        options.append((label, rest[1:].split('"', 1)[0] if rest.startswith('"') else rest))
    real = [(label, name) for label, name in options if label != "0"]
    named = [label for label, name in real if name.lower() == term]
    chosen = (named or [real[0][0]])[0]
    others = [label for label, _ in real if label != chosen]
    mass = {label: 0.0 for label, _ in options}
    for label in others:
        mass[label] = 0.25 / max(len(others), 1)
    mass[chosen] = favourite
    mass["0"] = 0.02
    if none_wins:
        mass = {label: 0.0 for label in mass}
        mass["0"], mass[real[0][0]] = 0.95, 0.03
    total = sum(mass.values())
    mass = {label: label_mass * p / total for label, p in mass.items()}
    tops = sorted(((t, p) for t, p in mass.items() if p > 0), key=lambda kv: -kv[1])[:19]
    tops.append(("The", max(1 - label_mass, 1e-6)))
    first = tops[0][0]
    entry = {"token": first, "bytes": [ord(first)], "logprob": math.log(tops[0][1]),
             "top_logprobs": [{"token": t, "bytes": [], "logprob": math.log(p)} for t, p in tops]}
    tokens = len(encode(body)) // 4
    data: dict[str, Any] = {
        "id": "chatcmpl-1", "model": model, "object": "chat.completion",
        "choices": [{"finish_reason": finish, "index": 0,
                     "message": {"content": first, "role": "assistant", **(message or {})},
                     "logprobs": {"content": [entry]}}],
        "usage": {"prompt_tokens": tokens, "completion_tokens": 1, "total_tokens": tokens + 1}}
    if build is not None:
        data["system_fingerprint"] = build
    return json.dumps(data)


class FakeChat:
    """A client that answers every request in process."""

    def __init__(self, **options: Any) -> None:
        self.options = options
        self.calls: list[dict[str, Any]] = []

    def settings(self) -> dict[str, Any]:
        return {"host": "fake"}

    def post(self, body: dict[str, Any]) -> Exchange:
        self.calls.append(body)
        return Exchange(200, respond(body, **self.options), None, 1, (), 0.01, 0.01, now_utc())


class Refusing(FakeChat):
    """Answers with a status instead of a result."""

    def __init__(self, status: int) -> None:
        super().__init__()
        self.status = status

    def post(self, body: dict[str, Any]) -> Exchange:
        self.calls.append(body)
        return Exchange(self.status, None, f"HTTP {self.status}: no", 1, (), 0.01, 0.01, now_utc())


class FlakyFor(FakeChat):
    """Fails the first N requests as an exhausted 503, then behaves."""

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
    settings: dict[str, Any] = {"baseline_dir": BASELINE, "max_requests": 100_000, "only": PILOT,
                                "workers": 1, "resamples": 100}
    settings.update(options)
    fake = client if client is not None else FakeChat()
    return run_chat(out, client_factory=lambda: fake, **settings)


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def stored(out: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (out / "responses.jsonl").read_text().splitlines()]


def test_a_pilot_run_writes_every_artifact_and_passes_every_gate(tmp_path: Path) -> None:
    out = tmp_path / "psnc"
    fake = FakeChat()
    manifest = execute(out, fake)
    assert [c["name"] for c in manifest["checks"]] == list(REQUIRED_CHAT_CHECKS)
    assert all(c["passed"] is True for c in manifest["checks"])
    assert set(manifest["files"]) == set(PUBLISHED_JEV)
    for name, expected in manifest["files"].items():
        assert file_sha256(out / name) == expected
    assert manifest["pilot_variables"] == sorted(PILOT) and manifest["experiment"] == (
        "psnc-entity-linking-v1")
    model = manifest["model"]
    assert model["requested"] == MODEL and model["served"] == [MODEL] and model["builds"] == [BUILD]
    assert model["providers"] == ["psnc"] and model["request_parameters"]["temperature"] == 0
    assert len(stored(out)) == len(fake.calls) == manifest["spend"]["requests"] > 0
    assert manifest["spend"]["cost_usd"] == "0" and "free access" in manifest["spend"]["billing"]
    journal = read(out / "provenance.json")
    assert journal["invocations"][-1]["outcome"] == "completed"
    assert journal["invocations"][-1]["max_requests"] == 100_000
    assert journal["invocations"][-1]["client"] == {"host": "fake"}


def test_the_evidence_is_the_jev_evidence_plus_the_two_chat_modules() -> None:
    assert PSNC_EVIDENCE_FILES == (*JEV_EVIDENCE_FILES, "iadopt_linking/chat.py",
                                   "iadopt_linking/chat_choice.py")


def test_every_pass_asks_every_slot_that_has_candidates(tmp_path: Path) -> None:
    out = tmp_path / "psnc"
    fake = FakeChat()
    manifest = execute(out, fake)
    plan = read(out / "plan.json")
    slots = manifest["population"]["slots"] - plan["asked"]["empty_pools"]
    assert len(fake.calls) == slots * len(CHAT_PASSES) == plan["tasks"]
    per_pass: dict[str, int] = {}
    for record in stored(out):
        per_pass[record["pass"]] = per_pass.get(record["pass"], 0) + 1
    assert per_pass == {spec.name: slots for spec in CHAT_PASSES}
    assert all(call["chat_template_kwargs"] == {"enable_thinking": False} for call in fake.calls)


def test_the_tables_for_later_analysis_have_one_row_per_slot_pass_and_candidate(
        tmp_path: Path) -> None:
    out = tmp_path / "psnc"
    execute(out)
    slots = read(out / "manifest.json")["population"]["slots"]
    outcome_rows = (out / "outcomes.csv").read_text().splitlines()
    assert len(outcome_rows) == 1 + slots
    header = outcome_rows[0]
    assert header.startswith("slot_id,") and "llm_selected_probability" in header
    assert "query_key" in header and "qwen3_correct" in header and "llm_confidence" in header
    assert "jev_" not in header and "pointwise" not in header
    confidence_rows = (out / "confidence.csv").read_text().splitlines()
    assert len(confidence_rows) == 1 + slots * len(CHAT_PASSES) and "llm_own_choice" in confidence_rows[0]
    probabilities = (out / "probabilities.csv").read_text().splitlines()[1:]
    shown = sum(len(r["shown"]) for r in stored(out))
    assert len(probabilities) == shown


def test_the_report_names_the_model_and_has_the_baselines_sections_and_its_own(
        tmp_path: Path) -> None:
    out = tmp_path / "psnc"
    execute(out)
    report = (out / "report.md").read_text()
    for heading in ("## Result", f"## {MODEL} against Qwen3 on the reachable slots",
                    "## Where the misses come from", "## Diagnostics", "## Which source wins",
                    "## Passes, order and chance", f"## {MODEL}'s own `none_of_these`",
                    "## What the confidence is worth", "## Run health", "## Notes"):
        assert heading in report
    assert f"## Slots only {MODEL} got right" in (out / "comparison.md").read_text()
    for jev_only in ("TypeSafe", "OpenRouter", "typed-decision", "usage.cost", "Jev's", "Jev top-1",
                     "rounded to two decimals", "documents Jev"):
        assert jev_only not in report, f"{jev_only!r} describes Jev, not this model"
    assert "not billed to the owner" in report and "log-probabilities" in report
    assert f"answers from {MODEL} served by PSNC" in report
    assert "own-choice probability" in report


def test_a_second_run_asks_nothing_and_changes_no_evidence(tmp_path: Path) -> None:
    out = tmp_path / "psnc"
    execute(out)
    before = {name: file_sha256(out / name)
              for name in ("responses.jsonl", "plan.json", "summary.json", "outcomes.csv",
                           "confidence.csv", "probabilities.csv")}

    def forbidden() -> Any:
        raise AssertionError("a finished folder must not build a client")

    manifest = run_chat(out, baseline_dir=BASELINE, max_requests=None, only=PILOT, resamples=100,
                        client_factory=forbidden)
    after = {name: file_sha256(out / name) for name in before}
    assert before == after
    assert len(read(out / "provenance.json")["invocations"]) == 2
    assert all(c["passed"] for c in manifest["checks"])


def test_the_forced_choice_ignores_a_none_of_these_that_wins(tmp_path: Path) -> None:
    out = tmp_path / "psnc"
    execute(out, FakeChat(none_wins=True))
    rows = list(csv.DictReader((out / "outcomes.csv").open(encoding="utf-8")))
    asked = [r for r in rows if int(r["pool_size"]) > 0]
    assert asked and all(r["llm_own_choice"] == NONE_KEY for r in asked), "the refusal is recorded"
    assert all(r["llm_selected_local_id"] for r in asked), "the choice is still forced"
    abstention = read(out / "summary.json")["analysis"]["abstention"]
    assert abstention["answered"]["numerator"] == 0 and abstention["asked"] == len(asked)


def test_requests_that_fail_stop_the_run_and_the_next_run_asks_only_those(tmp_path: Path) -> None:
    out = tmp_path / "psnc"
    flaky = FlakyFor(failures=3)
    with pytest.raises(RunError, match="3 of .* requests have no usable answer"):
        execute(out, flaky)
    assert not (out / "manifest.json").exists()
    failed = [r for r in stored(out) if r["status"] != 200]
    assert len(failed) == 3 and all(r["response_text"] is None for r in failed)
    healthy = FakeChat()
    execute(out, healthy)
    assert len(healthy.calls) == 3 and (out / "manifest.json").is_file()
    assert read(out / "provenance.json")["invocations"][0]["outcome"].startswith("stopped: RunError")


@pytest.mark.parametrize("status", [401, 402, 403])
def test_a_refused_credential_stops_at_once_and_keeps_what_was_stored(
        tmp_path: Path, status: int) -> None:
    out = tmp_path / "psnc"
    refusing = Refusing(status)
    with pytest.raises(RunError, match="refused the credential or the account"):
        execute(out, refusing)
    assert len(refusing.calls) <= 2 and not (out / "manifest.json").exists()
    asking = read(out / "provenance.json")["invocations"][0]["asking"]
    assert asking["failed"] >= 1, "a stopped invocation still says what it did"


def test_a_run_without_a_request_limit_is_refused_before_any_call(tmp_path: Path) -> None:
    fake = FakeChat()
    with pytest.raises(RunError, match="explicit request limit"):
        execute(tmp_path / "psnc", fake, max_requests=None)
    assert fake.calls == []


def test_a_limit_below_what_is_pending_is_refused_before_any_call(tmp_path: Path) -> None:
    fake = FakeChat()
    with pytest.raises(RunError, match="request limit is 5"):
        execute(tmp_path / "psnc", fake, max_requests=5)
    assert fake.calls == []


def test_the_limit_also_holds_while_asking_and_what_was_asked_is_kept(tmp_path: Path) -> None:
    # run_chat refuses a limit below what is pending, so the limit inside the asking phase is a
    # second line of defence; reach it by calling the phase directly.
    from decimal import Decimal

    from iadopt_linking.chat_choice import build_tasks
    from iadopt_linking.chat_experiment import PSNC

    baseline = load_baseline(BASELINE, only=PILOT)
    tasks, _ = build_tasks(baseline, CHAT_PASSES)
    paths = JevPaths(tmp_path / "psnc")
    paths.root.mkdir()
    fake = FakeChat()
    with pytest.raises(RunError, match="request limit of 7 was reached"):
        ask_phase(paths, tasks, fake, workers=1, cap_usd=None, spent_before=Decimal(0), exp=PSNC,
                  max_requests=7)
    assert len(fake.calls) == 7 == len(stored(paths.root)), "what was asked is stored"


@pytest.mark.parametrize(("field", "value"), [
    ("prompt", {"fingerprint": "tampered"}),
    ("passes", [["choice-a1", "choice", "shuffle-a"]]),
    ("baseline", {"folder": "other"}),
    ("only", ["EWCT"]),
    ("evidence_code_sha256", "0" * 64),
    ("analysis", {}),
    ("model", {"requested": "other"}),
    ("reference", {"folder": "other"}),
    ("corpus", {"release": "another corpus release"}),
    ("run_id", "another run name"),
])
def test_a_folder_that_does_not_match_this_run_is_not_resumed(
        tmp_path: Path, field: str, value: Any) -> None:
    out = tmp_path / "psnc"
    execute(out)
    plan = read(out / "plan.json")
    plan[field] = value
    (out / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(RunError, match="resuming would mix evidence"):
        execute(out)


def test_evidence_without_a_plan_or_a_journal_is_refused(tmp_path: Path) -> None:
    out = tmp_path / "psnc"
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
    out = tmp_path / "psnc"
    execute(out)
    lines = (out / "responses.jsonl").read_text().splitlines()
    record = json.loads(lines[0])
    record["response_text"] = record["response_text"].replace("0.", "1.", 1) + " "
    lines[0] = json.dumps(record)
    (out / "responses.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(RunError, match="do not match their hash"):
        execute(out, FakeChat())


def test_two_server_builds_in_one_run_are_refused(tmp_path: Path) -> None:
    class Drifting(FakeChat):
        def post(self, body: dict[str, Any]) -> Exchange:
            self.options = {"build": BUILD if len(self.calls) % 2 == 0 else "another-build"}
            return super().post(body)

    with pytest.raises(RunError, match="one model snapshot served"):
        execute(tmp_path / "psnc", Drifting())


def test_another_model_answering_is_refused(tmp_path: Path) -> None:
    with pytest.raises(RunError, match="one model snapshot served"):
        execute(tmp_path / "psnc", FakeChat(model="Qwen3-Coder-Next"))


def test_a_response_with_reasoning_text_is_stored_as_a_failure_not_scored(tmp_path: Path) -> None:
    out = tmp_path / "psnc"
    with pytest.raises(RunError, match="first 10 requests all failed .*reasoned"):
        execute(out, FakeChat(message={"reasoning_content": "let me think"}))
    records = stored(out)
    assert records and all("reasoned" in (r["parse_error"] or "") for r in records)
    assert all(r["response_text"] for r in records), "the response is kept as evidence"


def test_a_response_with_nothing_on_the_labels_is_stored_as_a_failure(tmp_path: Path) -> None:
    out = tmp_path / "psnc"
    with pytest.raises(RunError, match="first 10 requests all failed .*of the first token"):
        execute(out, FakeChat(label_mass=0.2))
    assert all("option labels" in (r["parse_error"] or "") for r in stored(out))


def test_responses_without_token_usage_are_stored_as_failures_and_stop_the_run(
        tmp_path: Path) -> None:
    class NoUsage(FakeChat):
        def post(self, body: dict[str, Any]) -> Exchange:
            data = json.loads(respond(body))
            data["usage"] = {"total_tokens": 3}
            return Exchange(200, json.dumps(data), None, 1, (), 0.01, 0.01, now_utc())

    out = tmp_path / "psnc"
    with pytest.raises(RunError, match="first 10 requests all failed .*usage.prompt_tokens"):
        execute(out, NoUsage())
    assert stored(out) and all("usage.prompt_tokens" in (r["parse_error"] or "") for r in stored(out))


def test_the_tokens_a_stored_response_reports_are_read_whether_or_not_it_was_valid() -> None:
    body = {"messages": [{}, {"content": "term: a\n\ncandidates:\n1. \"a\"\n"
                                         "0. none of these: x\n\nA"}]}
    assert tokens_of(respond(body)) is not None
    assert tokens_of('{"usage": {"prompt_tokens": 1}}') is None and tokens_of(None) is None
    assert tokens_of("not json") is None and tokens_of("[" * 100_000) is None


def test_workers_change_the_order_responses_arrive_in_but_not_a_single_number(
        tmp_path: Path) -> None:
    one = tmp_path / "one"
    many = tmp_path / "many"
    execute(one, FakeChat(), workers=1)
    execute(many, FakeChat(), workers=4)
    for name in ("summary.json", "outcomes.csv", "confidence.csv", "probabilities.csv"):
        assert (one / name).read_text() == (many / name).read_text(), name


def test_the_credential_never_reaches_a_file(tmp_path: Path) -> None:
    sent: list[dict[str, str]] = []

    def transport(url: str, headers: Any, body: bytes, timeout: float) -> tuple[int, dict, bytes]:
        sent.append(dict(headers))
        return 200, {}, respond(json.loads(body)).encode()

    out = tmp_path / "psnc"
    run_chat(out, baseline_dir=BASELINE, max_requests=100_000, only=PILOT, resamples=100,
             client_factory=lambda: RedactingChatClient(KEY, transport=transport))
    assert sent and all(h["Authorization"] == f"Bearer {KEY}" for h in sent)
    for path in out.iterdir():
        assert KEY not in path.read_text(encoding="utf-8"), f"the key leaked into {path.name}"
    assert "Authorization" not in (out / "provenance.json").read_text()
    assert "llm.hpc.psnc.pl" in (out / "provenance.json").read_text()


def test_the_client_for_a_real_run_reads_its_key_from_the_credential_file(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = tmp_path / ".env"
    env.write_text(f"PSNC_API_KEY={KEY}\n", encoding="utf-8")
    seen: list[str] = []

    def transport(url: str, headers: Any, body: bytes, timeout: float) -> tuple[int, dict, bytes]:
        seen.append(f"{url}|{headers['Authorization']}")
        return 200, {}, respond(json.loads(body)).encode()

    monkeypatch.setattr("iadopt_linking.chat_experiment.RedactingChatClient",
                        lambda key, base_url: RedactingChatClient(
                            key, base_url=base_url, transport=transport))
    monkeypatch.delenv("PSNC_API_KEY", raising=False)
    monkeypatch.delenv("PSNC_API_BASE_URL", raising=False)
    run_chat(tmp_path / "psnc", baseline_dir=BASELINE, max_requests=100_000, only=PILOT,
             resamples=100, env_file=str(env))
    assert set(seen) == {f"https://llm.hpc.psnc.pl/v1/chat/completions|Bearer {KEY}"}


def test_a_missing_credential_stops_before_any_request(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PSNC_API_KEY", raising=False)
    empty = tmp_path / "empty.env"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(RunError, match="PSNC_API_KEY is not set"):
        run_chat(tmp_path / "psnc", baseline_dir=BASELINE, max_requests=100_000, only=PILOT,
                 env_file=str(empty))


def test_an_unknown_pass_a_duplicate_or_a_missing_primary_pass_is_refused_before_any_call(
        tmp_path: Path) -> None:
    fake = FakeChat()
    with pytest.raises(UsageError, match="No such pass"):
        execute(tmp_path / "a", fake, passes=["noul-n1"])
    with pytest.raises(UsageError, match="duplicates"):
        execute(tmp_path / "b", fake, passes=["choice-a1", "choice-a1"])
    with pytest.raises(UsageError, match="primary pass"):
        execute(tmp_path / "c", fake, passes=["choice-a2-reversed"])
    assert fake.calls == []


def test_a_model_without_a_measured_reasoning_control_is_refused(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match="no measured control"):
        execute(tmp_path / "psnc", FakeChat(), model="GLM-5.2")


# --- the command line ---

def test_psnc_plan_calls_nothing_and_shows_a_request(
        capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a plan must not build a client")

    monkeypatch.setattr("iadopt_linking.chat_experiment.RedactingChatClient", boom)
    assert main(["psnc-plan", "--baseline", str(BASELINE), "--only", ",".join(PILOT)]) == 0
    text = capsys.readouterr().out
    assert "without candidates (not asked)" in text and '"requests": ' in text
    assert '"enable_thinking": false' in text and '"model": "Qwen3.8-27B"' in text


def test_psnc_run_needs_a_request_limit_and_a_baseline() -> None:
    with pytest.raises(SystemExit):
        main(["psnc-run", "--baseline", str(BASELINE), "--out", "x"])
    with pytest.raises(SystemExit):
        main(["psnc-run", "--out", "x", "--max-requests", "5"])
    with pytest.raises(SystemExit):
        main(["psnc-run", "--baseline", str(BASELINE), "--out", "x", "--max-requests", "0"])


def test_psnc_run_through_the_command_line(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeChat()
    monkeypatch.setattr("iadopt_linking.cli.run_chat", functools.partial(
        run_chat, client_factory=lambda: fake, resamples=100))
    out = tmp_path / "psnc"
    assert main(["psnc-run", "--baseline", str(BASELINE), "--out", str(out), "--max-requests",
                 "100000", "--only", ",".join(PILOT), "--passes", "choice-a1,choice-a3"]) == 0
    assert {r["pass"] for r in stored(out)} == {"choice-a1", "choice-a3"}
    assert read(out / "plan.json")["passes"] == [["choice-a1", "choice", "shuffle-a"],
                                                 ["choice-a3", "choice", "shuffle-b"]]


# --- the reference run (Jev) and the full population ---

@pytest.fixture(scope="module")
def full_run(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """One complete fake run over all 294 slots, compared with the committed Jev run."""
    out = tmp_path_factory.mktemp("psnc-full") / "psnc"
    experiment_patch = pytest.MonkeyPatch()
    experiment_patch.setattr(experiment, "load_gold_links", CACHED_GOLD)
    try:
        run_chat(out, baseline_dir=BASELINE, max_requests=100_000, workers=1, resamples=100,
                 client_factory=lambda: FakeChat(), reference=JEV_RUN)
    finally:
        experiment_patch.undo()
    yield out


def test_a_full_run_is_compared_with_the_committed_jev_run(full_run: Path) -> None:
    summary = read(full_run / "summary.json")
    reference = summary["analysis"]["reference"]
    assert reference["folder"] == "jev-20261015" and reference["experiment"] == "jev-entity-linking-v1"
    assert reference["top1"]["reachable"] == {"numerator": 213, "denominator": 284}
    assert reference["top1"]["all"] == {"numerator": 213, "denominator": 294}
    paired = reference["primary_vs_reference"]["reachable"]
    assert paired["both"] + paired["only_first"] + paired["only_second"] + paired["neither"] == 284
    own = summary["primary"]["populations"]["reachable"]["top1"]
    assert paired["both"] + paired["only_first"] == own["numerator"]
    assert paired["both"] + paired["only_second"] == 213
    assert reference["primary_vs_reference"]["reachable"]["terms"]["terms"] == 145
    plan = read(full_run / "plan.json")
    jev_files = read(JEV_RUN / "manifest.json")["files"]
    assert plan["reference"] == {
        "folder": "jev-20261015", "experiment": "jev-entity-linking-v1",
        "responses_sha256": jev_files["responses.jsonl"], "plan_sha256": jev_files["plan.json"],
        "outcomes_sha256": jev_files["outcomes.csv"]}
    report = (full_run / "report.md").read_text()
    assert f"## {MODEL} against Jev on the same slots (pre-specified)" in report
    assert "213/284 = 0.750" in report
    comparison = (full_run / "comparison.md").read_text()
    assert f"## {MODEL} against Jev on the same slots (pre-specified)" in comparison


def test_a_full_run_reads_the_baseline_the_way_the_jev_run_did(full_run: Path) -> None:
    summary = read(full_run / "summary.json")
    assert summary["baseline"]["populations"]["reachable"]["top1"] == {
        "numerator": 194, "denominator": 284}
    manifest = read(full_run / "manifest.json")
    assert all(c["passed"] is True for c in manifest["checks"])
    assert manifest["population"]["slots"] == 294 and manifest["spend"]["requests"] == 1144


def test_the_reference_must_be_an_unaltered_jev_run_on_the_same_candidates(tmp_path: Path) -> None:
    baseline = load_baseline(BASELINE)
    assert load_reference(JEV_RUN, baseline)["identity"]["folder"] == "jev-20261015"
    with pytest.raises(RunError, match="not a finished run"):
        load_reference(tmp_path / "nowhere", baseline)
    with pytest.raises(RunError, match="not a finished Jev run"):
        load_reference(BASELINE, baseline)
    copy = tmp_path / "jev"
    shutil.copytree(JEV_RUN, copy)
    (copy / "outcomes.csv").write_text(
        (copy / "outcomes.csv").read_text().replace(",1,", ",0,", 1), encoding="utf-8")
    with pytest.raises(RunError, match="no longer evidence"):
        load_reference(copy, baseline)
    other = tmp_path / "other"
    shutil.copytree(JEV_RUN, other)
    manifest = read(other / "manifest.json")
    manifest["baseline"]["scores_sha256"] = "0" * 64
    (other / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RunError, match="different baseline evidence"):
        load_reference(other, baseline)


def test_rebuilding_the_reference_report_does_not_change_which_reference_it_is(
        tmp_path: Path) -> None:
    baseline = load_baseline(BASELINE)
    before = load_reference(JEV_RUN, baseline)["identity"]
    copy = tmp_path / "jev-20261015"
    shutil.copytree(JEV_RUN, copy)
    (copy / "report.md").write_text("# rebuilt with other words\n", encoding="utf-8")
    manifest = read(copy / "manifest.json")
    manifest["files"]["report.md"] = file_sha256(copy / "report.md")
    manifest["created_at"] = "2027-01-01T00:00:00+00:00"
    (copy / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert load_reference(copy, baseline)["identity"] == before
    (copy / "responses.jsonl").write_text(
        (copy / "responses.jsonl").read_text() + "\n", encoding="utf-8")
    manifest["files"]["responses.jsonl"] = file_sha256(copy / "responses.jsonl")
    (copy / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert load_reference(copy, baseline)["identity"] != before, "other evidence is another reference"


def test_a_reference_that_covers_other_slots_is_refused(tmp_path: Path) -> None:
    pilot = load_baseline(BASELINE, only=PILOT)
    with pytest.raises(RunError, match="does not cover exactly this run's slots"):
        load_reference(JEV_RUN, pilot)


def test_the_option_labels_and_ids_agree_for_every_pool_size() -> None:
    for size in range(1, 31):
        mapping = label_to_option(size)
        assert mapping[option_label(size)] == f"c{size:02d}" and mapping["0"] == NONE_KEY
        assert len(mapping) == size + 1


# --- each check can fail, and the totals count every stored response ---

@pytest.fixture(scope="module")
def pilot_folder(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """One finished fake pilot run, whose stored evidence the tests below damage in memory."""
    out = tmp_path_factory.mktemp("psnc-pilot") / "psnc"
    patch = pytest.MonkeyPatch()
    patch.setattr(experiment, "load_gold_links", CACHED_GOLD)
    try:
        run_chat(out, baseline_dir=BASELINE, max_requests=100_000, only=PILOT, workers=1,
                 resamples=100, client_factory=lambda: FakeChat())
    finally:
        patch.undo()
    yield out


def with_text(record: dict[str, Any], text: str, **changes: Any) -> dict[str, Any]:
    """Give a stored record another response text and the hash that goes with it."""
    return {**record, "response_text": text,
            "response_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), **changes}


def judged(out: Path, change: Any = None, **inputs_changes: Any) -> dict[str, dict[str, Any]]:
    """Run the chat checks on a pilot folder's stored evidence after a change made in memory."""
    baseline = load_baseline(BASELINE, only=PILOT)
    tasks, _ = build_tasks(baseline, CHAT_PASSES)
    history = JsonlStore(out / "responses.jsonl").read()
    if change is not None:
        history = change(history)
    recorded = latest(history, lambda r: (r["slot_id"], r["pass"]))
    parsed, problems = read_answers(tasks, recorded, PSNC.parse_answer)
    inputs: dict[str, Any] = {
        "slots": len(baseline.slots),
        "candidates": sum(len(baseline.records[s.slot_id]["pool"]) for s in baseline.slots),
        "baseline_files": len(baseline.manifest["files"]), "baseline_records": len(baseline.records),
        "baseline_slots": None}
    inputs.update(inputs_changes)
    checks = chat_checks(tasks, recorded, parsed, problems, history=history, inputs=inputs,
                         cap_usd=None, max_requests=None)
    return {c["name"]: c for c in checks}


def test_every_check_passes_on_untouched_evidence_and_is_reported_in_order(
        pilot_folder: Path) -> None:
    checks = judged(pilot_folder)
    assert list(checks) == list(REQUIRED_CHAT_CHECKS)
    assert all(c["passed"] is True for c in checks.values())


DAMAGE = {
    "inputs identical to the baseline": (None, {"baseline_slots": 294, "baseline_records": 294}),
    "every request answered": (lambda h: h[1:], {}),
    "stored requests regenerate from the plan": (
        lambda h: [{**h[0], "request_sha256": "0" * 64}, *h[1:]], {}),
    "stored response hashes verify": (
        lambda h: [{**h[0], "response_text": h[0]["response_text"] + " "}, *h[1:]], {}),
    "every response is a valid answer": (
        lambda h: [with_text(h[0], '{"model": "x"}'), *h[1:]], {}),
    "one model snapshot served": (
        lambda h: [with_text(h[0], json.dumps({**json.loads(h[0]["response_text"]),
                                               "model": "another-model"})), *h[1:]], {}),
    "token usage fully accounted": (
        lambda h: [with_text(h[0], '{"usage": {}}', parse_error="unusable"), *h], {}),
    "primary pass complete": (lambda h: [r for r in h if r["pass"] != "choice-a1"], {}),
}


def test_every_required_check_has_a_damage_that_fails_it() -> None:
    assert set(DAMAGE) == set(REQUIRED_CHAT_CHECKS)


@pytest.mark.parametrize("name", sorted(DAMAGE))
def test_each_check_fails_when_the_evidence_it_guards_is_damaged(
        pilot_folder: Path, name: str) -> None:
    change, inputs = DAMAGE[name]
    checks = judged(pilot_folder, change, **inputs)
    assert checks[name]["passed"] is False, checks[name]["detail"]


def test_tokens_are_counted_over_every_stored_response_a_rejected_one_included(
        pilot_folder: Path) -> None:
    usage = "token usage fully accounted"
    history = JsonlStore(pilot_folder / "responses.jsonl").read()
    plain = sum(sum(tokens_of(r["response_text"])) for r in history)
    rejected = with_text(history[0], history[0]["response_text"],
                         parse_error="the model reasoned although reasoning is switched off")
    detail = judged(pilot_folder, lambda h: [rejected, *h])[usage]
    again = sum(tokens_of(history[0]["response_text"]))
    assert f"{plain + again:,} tokens used" in detail["detail"]
    assert "1 rejected responses" in detail["detail"] and detail["passed"] is True


# --- the reference run is found where the plan says, or the run does not start ---

def test_a_reference_that_is_not_beside_the_baseline_is_refused_before_any_request(
        tmp_path: Path) -> None:
    elsewhere = tmp_path / "elsewhere" / "jev-20261015"
    shutil.copytree(JEV_RUN, elsewhere)
    fake = FakeChat()
    with pytest.raises(RunError, match="must sit in the same folder as the baseline"):
        execute(tmp_path / "psnc", fake, reference=elsewhere)
    assert fake.calls == [] and not (tmp_path / "psnc" / "plan.json").exists()


def test_another_spelling_of_the_folder_beside_the_baseline_is_the_same_place(
        tmp_path: Path) -> None:
    # The pilot is not covered by the full Jev run, which is the next thing that is checked: so
    # the location rule let this spelling through.
    spelled = RESULTS / "baseline-20261014" / ".." / "jev-20261015"
    with pytest.raises(RunError, match="does not cover exactly this run's slots"):
        execute(tmp_path / "psnc", FakeChat(), reference=spelled)


def test_a_reference_that_changed_after_the_run_was_planned_is_caught_when_it_is_read_again(
) -> None:
    baseline = load_baseline(BASELINE)
    identity = load_reference(JEV_RUN, baseline)["identity"]
    plan = {"reference": {**identity, "outcomes_sha256": "0" * 64}}
    with pytest.raises(RunError, match="changed since this run was planned"):
        extra_analysis(plan, baseline, {}, {})
    assert extra_analysis({}, baseline, {}, {}) == {}


# --- the notes say what the data show ---

def note_inputs(*, repeat: bool, rejected: dict[str, int] | None = None) -> tuple[Any, ...]:
    parsed = {("s", "p"): {"label_mass": 0.96, "top_mass": 0.998, "labels_reported": 18},
              ("t", "p"): {"label_mass": 0.9995, "top_mass": 0.9999, "labels_reported": 4}}
    summary = {"primary": {"populations": {"unreachable": {"n": 3}}},
               "analysis": {"repeatability": {
                   "same_selection": {"numerator": 9, "denominator": 10},
                   "identical_scores": {"numerator": 8, "denominator": 10}}}}
    return (summary, {}, {"empty_pools": 2}, (1, 20), rejected or {}, 300, repeat, parsed)


def test_the_probability_note_says_what_a_probability_is_and_how_far_it_can_be_off() -> None:
    text = "\n".join(chat_notes(*note_inputs(repeat=False)))
    assert "lower bound" not in text
    assert "given that its first token is a label" in text
    assert "at least 0.9600 in every response" in text and "between 4 and 18 labels" in text
    assert "at 0.0021." in text      # (1 - 0.998) / 0.96, the worst of the two responses


def test_the_service_note_does_not_claim_a_repeat_pass_that_was_not_run_and_reads_as_prose() -> None:
    without = "\n".join(chat_notes(*note_inputs(repeat=False)))
    assert "repeated pass" not in without and ". a later" not in without
    assert ". A later run, or one made while the server is busier" in without
    with_repeat = "\n".join(chat_notes(*note_inputs(repeat=True)))
    assert "sent later in the same run" in with_repeat and "minutes" not in with_repeat
    assert "9/10 = 0.900" in with_repeat and "8/10 = 0.800" in with_repeat
    assert ". A later run" in with_repeat


def test_responses_that_reasoned_are_counted_in_the_notes_not_denied() -> None:
    clean = "\n".join(chat_notes(*note_inputs(repeat=False)))
    assert "carries reasoning text, and none did." in clean
    reasoned = "\n".join(chat_notes(*note_inputs(
        repeat=False, rejected={"the model reasoned although reasoning is switched off": 2})))
    assert "2 did and were asked again." in reasoned and "none did" not in reasoned
    assert "`chat_template_kwargs.enable_thinking=false`" in clean


# --- what the report says about how the run was sent and what is pre-specified ---

def test_a_pilot_without_a_reference_says_so_and_says_how_it_was_sent(tmp_path: Path) -> None:
    out = tmp_path / "psnc"
    execute(out)
    report = (out / "report.md").read_text()
    assert "sent one request at a time" in report
    assert "No reference run was named, so the pre-specified comparison with Jev was not made" in report
    assert "against Jev on the same slots" not in report


def test_a_run_sent_by_several_workers_says_so(tmp_path: Path) -> None:
    out = tmp_path / "many"
    execute(out, FakeChat(), workers=4)
    assert "sent by up to 4 workers" in (out / "report.md").read_text()


def test_a_repeat_pass_that_differs_a_little_prints_how_little_not_zero(tmp_path: Path) -> None:
    class Wobbly(FakeChat):
        """Answers an identical request slightly differently the second time it is asked."""

        def __init__(self) -> None:
            super().__init__()
            self.seen: dict[str, int] = {}

        def post(self, body: dict[str, Any]) -> Exchange:
            key = encode(body).decode("utf-8")
            self.seen[key] = self.seen.get(key, 0) + 1
            self.options = {"favourite": 0.7 if self.seen[key] == 1 else 0.7004}
            return super().post(body)

    out = tmp_path / "psnc"
    execute(out, Wobbly())
    report = (out / "report.md").read_text()
    assert re.search(r"largest difference in a candidate's probability was 0\.00[0-4]\d \(mean", report), (
        report[report.index("Identical request sent twice"):][:400])
    assert "identical for 281/281" not in report.replace(",", "")


def test_the_report_of_a_full_run_marks_the_equivalence_reading_as_exploratory(
        full_run: Path) -> None:
    for name in ("report.md", "comparison.md"):
        text = (full_run / name).read_text()
        assert "Exploratory, because whether such twins count is an open question" in text, name
        assert "also by gold vocabulary, is pre-specified too" in text or name == "comparison.md"
    manifest = read(full_run / "manifest.json")
    assert manifest["reference"]["folder"] == "jev-20261015"


# --- the command line passes on everything it was given ---

def test_psnc_run_forwards_every_option(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}
    monkeypatch.setattr("iadopt_linking.cli.run_chat",
                        lambda out, **options: seen.update(out=out, **options))
    assert main(["psnc-run", "--baseline", "b", "--out", "o", "--max-requests", "7", "--model", MODEL,
                 "--env-file", "e", "--lab-root", "l", "--only", "A, B", "--passes",
                 "choice-a1,choice-a3", "--workers", "2", "--run-id", "r", "--reference", "ref"]) == 0
    assert seen == {"out": "o", "baseline_dir": "b", "max_requests": 7, "model": MODEL,
                    "env_file": "e", "lab_root": "l", "only": ["A", "B"],
                    "passes": ["choice-a1", "choice-a3"], "workers": 2, "run_id": "r",
                    "reference": "ref"}


def test_psnc_run_defaults_to_one_request_at_a_time_and_no_reference(
        monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}
    monkeypatch.setattr("iadopt_linking.cli.run_chat",
                        lambda out, **options: seen.update(out=out, **options))
    assert main(["psnc-run", "--baseline", "b", "--out", "o", "--max-requests", "7"]) == 0
    assert seen["workers"] == 1 and seen["reference"] is None and seen["model"] == MODEL
    assert seen["only"] is None and seen["passes"] is None and seen["env_file"] is None
    assert inspect.signature(run_chat).parameters["workers"].default == 1
