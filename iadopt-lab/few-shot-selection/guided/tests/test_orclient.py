"""Offline tests for orclient.py: the pinned host, the cap, retries and the stop rules. No provider."""

from __future__ import annotations

import asyncio
import json

import engine
import orclient
import pytest

KEY = "sk-or-v1-0123456789abcdef"
ANSWER = {"id": "gen-1", "model": "qwen/qwen3.8-27b", "provider": "Near AI",
          "choices": [{"finish_reason": "stop", "message": {"content": "{}", "reasoning": None},
                       "logprobs": {"content": [{"token": "1", "top_logprobs": [
                           {"token": "1", "logprob": -0.1}, {"token": "2", "logprob": -2.5}]}]}}],
          "usage": {"prompt_tokens": 5000, "completion_tokens": 100, "cost": 0.0003,
                    "prompt_tokens_details": {"cached_tokens": 4000},
                    "completion_tokens_details": {"reasoning_tokens": 0}}}


class Reply:
    """A canned HTTP reply."""

    def __init__(self, status: int, body: object, headers: dict | None = None) -> None:
        """Hold the reply.

        Args: status: HTTP status; body: text, or an object sent as JSON; headers: reply headers.
        Returns: nothing.
        Raises: nothing.
        Side Effects: none.
        """
        self.status_code = status
        self.text = body if isinstance(body, str) else json.dumps(body)
        self.headers = headers or {}


class Client:
    """A stand-in for the HTTP client that replays canned replies and keeps what was sent."""

    def __init__(self, replies: list) -> None:
        """Queue the replies.

        Args: replies: `Reply` objects or exceptions, used in order; the last one repeats.
        Returns: nothing.
        Raises: nothing.
        Side Effects: none.
        """
        self.replies, self.sent = list(replies), []

    async def post(self, url: str, headers: dict, json: dict) -> Reply:  # noqa: A002 - httpx's name
        """Record one request and return the next reply.

        Args: url: the address; headers: request headers; json: the body.
        Returns: the next canned reply.
        Raises: the queued exception, when the next item is one.
        Side Effects: appends to `sent`.
        """
        self.sent.append({"url": url, "headers": dict(headers), "body": json})
        item = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(item, Exception):
            raise item
        return item


async def no_sleep(seconds: float) -> None:
    """Skip a wait.

    Args: seconds: the wait asked for.
    Returns: nothing.
    Raises: nothing.
    Side Effects: none.
    """


def transport_with(tmp_path, replies, limit=1.0):
    """Build a transport over canned replies with its own ledger.

    Args: tmp_path: a temporary folder; replies: for `Client`; limit: the run's spending limit.
    Returns: (transport, client, spend).
    Raises: nothing.
    Side Effects: none until the transport is called.
    """
    spend = orclient.Spend(ledger=tmp_path / "ledger.jsonl", limit=limit)
    client = Client(replies)
    transport, _ = orclient.open_transport(spend, "test", client=client, key=KEY, sleep=no_sleep)
    return transport, client, spend


BODY = {"model": "Qwen3.8-27B", "stream": False, "temperature": 0.5, "top_p": 1.0, "max_tokens": 16000,
        "chat_template_kwargs": {"enable_thinking": False}, "messages": [{"role": "user", "content": "hello"}]}


def test_translate_pins_one_host_and_switches_reasoning_off():
    """The body names this module's model, one host without fallbacks, and no PSNC-only setting."""
    out = orclient.translate(BODY)
    assert out["model"] == orclient.MODEL and out["messages"] == BODY["messages"]
    assert out["provider"] == {"only": ["near-ai"], "allow_fallbacks": False, "require_parameters": True,
                               "quantizations": ["fp8"]}
    assert out["reasoning"] == {"enabled": False} and out["usage"] == {"include": True}
    assert (out["temperature"], out["top_p"], out["max_tokens"]) == (0.5, 1.0, 16000)
    assert "chat_template_kwargs" not in out and "user" not in out


def test_read_keeps_cost_host_and_first_token_alternatives():
    """An answer keeps what the runners and the ledger need."""
    out = orclient.read(200, json.dumps(ANSWER), KEY)
    assert out["ok"] and out["answer"] == "{}" and out["host"] == "Near AI" and out["cost_usd"] == 0.0003
    assert (out["prompt_tokens"], out["completion_tokens"], out["cached_tokens"]) == (5000, 100, 4000)
    assert out["reasoning"] == "" and out["reasoning_tokens"] == 0
    assert [row["token"] for row in out["top"]] == ["1", "2"]


def test_read_scrubs_the_key_and_the_account_id():
    """A stored error never holds the key, a token or the account's user id, nested or not."""
    body = json.dumps({"error": {"message": f"bad key {KEY}", "metadata": {"user_id": "user_2abcDEF"}}})
    out = orclient.read(401, body, KEY)
    assert not out["ok"] and out["status"] == 401
    assert KEY not in out["error"] and "user_2abcDEF" not in out["error"] and "sk-or" not in out["error"]
    nested = json.dumps({"error": {"message": json.dumps({"user_id": "user_9zzYYY", "why": "no credit"})}})
    assert "user_9zzYYY" not in orclient.read(402, nested, KEY)["error"]
    assert "user_9zzYYY" not in orclient.read(200, nested, KEY)["error"]


def test_an_error_inside_a_200_reply_is_a_failure():
    """OpenRouter can answer 200 with an error object; that is no answer."""
    out = orclient.read(200, json.dumps({"error": {"code": 502, "message": "upstream failed"}}), KEY)
    assert not out["ok"] and out["status"] == 502 and "upstream failed" in out["error"]
    assert not orclient.read(200, "<html>", KEY)["ok"]
    assert not orclient.read(200, json.dumps({"choices": []}), KEY)["ok"]
    assert not orclient.read(200, json.dumps({"choices": [None]}), KEY)["ok"]         # no exception either
    assert not orclient.read(200, json.dumps({"choices": ["not an object"]}), KEY)["ok"]
    assert not orclient.read(200, json.dumps([1, 2]), KEY)["ok"]
    odd = orclient.read(200, json.dumps({"choices": [{"message": None}], "usage": None}), KEY)
    assert odd["ok"] and odd["answer"] == "" and odd["cost_usd"] is None


def test_the_limit_of_a_run_never_exceeds_the_cap(tmp_path):
    """Reservations stop at the limit, and a limit above the cap is cut to the cap."""
    spend = orclient.Spend(ledger=tmp_path / "ledger.jsonl", limit=0.05)
    assert spend.reserve(0.03) and not spend.reserve(0.03) and spend.refused == 1
    spend.settle(0.03, {"cost_usd": 0.01})
    assert spend.spent == pytest.approx(0.01) and spend.room() == pytest.approx(0.04)
    assert orclient.Spend(ledger=tmp_path / "ledger.jsonl").spent == pytest.approx(0.01)   # read back
    assert orclient.Spend(ledger=tmp_path / "none.jsonl", limit=10 ** 6).limit == orclient.CAP_USD == 50.0
    assert orclient.Spend(ledger=tmp_path / "none.jsonl", purpose="p", allowance=10 ** 6).limit == 50.0


def test_the_allowance_of_a_purpose_holds_over_every_launch(tmp_path):
    """What one purpose may book is a total over its ledger rows, not a fresh amount per launch."""
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text("".join(json.dumps(row) + "\n" for row in (
        {"purpose": "stage", "cost_usd": 0.03, "booked_usd": 0.03},
        {"purpose": "other", "cost_usd": 0.50},                     # an older row without "booked_usd"
        {"purpose": "stage", "cost_usd": None, "booked_usd": 0.01})))
    first = orclient.Spend(ledger=ledger, purpose="stage", allowance=0.05)
    assert first.spent == pytest.approx(0.54) and first.room() == pytest.approx(0.01)
    assert not first.reserve(0.02) and first.reserve(0.01)
    first.settle(0.01, {"purpose": "stage", "cost_usd": 0.01, "booked_usd": 0.01})
    again = orclient.Spend(ledger=ledger, purpose="stage", allowance=0.05)
    assert again.room() == pytest.approx(0.0) and not again.reserve(0.001)
    assert orclient.Spend(ledger=ledger, purpose="other", allowance=0.75).room() == pytest.approx(0.25)


async def test_a_transient_failure_is_retried_and_every_attempt_is_in_the_ledger(tmp_path):
    """A 429 followed by an answer costs two attempts and gives the answer."""
    transport, client, spend = transport_with(tmp_path, [Reply(429, "slow down", {"retry-after": "1"}),
                                                         Reply(200, ANSWER)])
    out = await transport(BODY)
    assert out["ok"] and out["attempts"] == 2 and len(client.sent) == 2
    rows = [json.loads(line) for line in (tmp_path / "ledger.jsonl").read_text().splitlines()]
    assert [(row["status"], row["cost_usd"], row["booked_usd"], row["estimated"]) for row in rows] == [
        (429, None, 0.0, False), (200, 0.0003, 0.0003, False)]
    assert spend.spent == pytest.approx(0.0003) and spend.reserved == pytest.approx(0.0)


async def test_a_request_carries_the_credential_and_nothing_about_the_account(tmp_path):
    """Only the authorization and the content type are sent as headers."""
    transport, client, _ = transport_with(tmp_path, [Reply(200, ANSWER)])
    await transport(BODY)
    assert client.sent[0]["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert set(client.sent[0]["headers"]) == {"Authorization", "Content-Type"}
    assert KEY not in json.dumps(client.sent[0]["body"])


async def test_a_refused_credential_or_credit_is_not_retried(tmp_path):
    """HTTP 402 means no credit; asking again changes nothing."""
    transport, client, _ = transport_with(tmp_path, [Reply(402, "no credit")])
    out = await transport(BODY)
    assert not out["ok"] and out["status"] == 402 and out["attempts"] == 1 and len(client.sent) == 1


async def test_a_connection_error_is_data_and_is_booked_at_the_worst_case(tmp_path):
    """A transport error never raises, is tried `MAX_ATTEMPTS` times, and counts as if it had been billed."""
    transport, client, spend = transport_with(tmp_path, [TimeoutError()])
    out = await transport(BODY)
    assert not out["ok"] and out["error"] == "TimeoutError" and len(client.sent) == orclient.MAX_ATTEMPTS
    bound = orclient.worst_call_usd(16000)
    rows = [json.loads(line) for line in (tmp_path / "ledger.jsonl").read_text().splitlines()]
    assert [(row["booked_usd"], row["estimated"], row["cost_usd"]) for row in rows] == [(bound, True, None)] * 3
    assert spend.spent == pytest.approx(3 * bound)


async def test_an_answer_without_a_cost_is_booked_from_its_tokens(tmp_path):
    """A reply that reports no cost counts at the listed prices; without tokens too, at the worst case."""
    no_cost = json.loads(json.dumps(ANSWER))
    del no_cost["usage"]["cost"]
    nothing = json.loads(json.dumps(ANSWER))
    nothing["usage"] = None
    transport, _, spend = transport_with(tmp_path, [Reply(200, no_cost), Reply(200, nothing)])
    assert (await transport(BODY))["ok"] and (await transport(BODY))["ok"]
    rows = [json.loads(line) for line in (tmp_path / "ledger.jsonl").read_text().splitlines()]
    listed = (5000 * 0.04 + 100 * 1.35) / 1e6
    assert rows[0]["booked_usd"] == pytest.approx(listed) and rows[0]["estimated"] and rows[0]["cost_usd"] is None
    assert rows[1]["booked_usd"] == orclient.worst_call_usd(16000) and rows[1]["estimated"]
    assert spend.spent == pytest.approx(listed + orclient.worst_call_usd(16000))
    assert orclient.book({"ok": False, "status": 200, "error": "error in body"}, 0.02) == (0.02, True)
    assert orclient.book({"ok": False, "status": 500}, 0.02) == (0.0, False)


async def test_the_cap_holds_when_many_calls_start_at_once(tmp_path):
    """Two hundred calls against a limit of ten cents: only what the worst case allows is sent."""
    spend = orclient.Spend(ledger=tmp_path / "ledger.jsonl", limit=0.10)
    dear = json.loads(json.dumps(ANSWER))
    dear["usage"]["cost"] = 0.02

    class Slow(Client):
        """A client that lets other calls run before it answers."""

        async def post(self, url: str, headers: dict, json: dict) -> Reply:  # noqa: A002 - httpx's name
            """Yield to the other calls, then answer.

            Args: url, headers, json: as for `Client.post`.
            Returns: the canned reply.
            Raises: nothing.
            Side Effects: appends to `sent`.
            """
            await asyncio.sleep(0)
            return await super().post(url, headers, json)

    client = Slow([Reply(200, dear)])
    transport, _ = orclient.open_transport(spend, "test", client=client, key=KEY, sleep=no_sleep)
    results = await asyncio.gather(*(transport(BODY) for _ in range(200)))
    sent = [r for r in results if not r.get("refused")]
    assert len(sent) == len(client.sent) == 4 and all(r["ok"] for r in sent)
    assert spend.spent == pytest.approx(0.08) and spend.spent <= spend.limit and spend.reserved == pytest.approx(0.0)
    assert all(r["error"] == "spend cap reached" for r in results if r.get("refused"))


async def test_a_long_prompt_reserves_more(tmp_path):
    """The reservation grows with the prompt, so a limit a short prompt fits can refuse a long one."""
    transport, client, spend = transport_with(tmp_path, [Reply(200, ANSWER)], limit=0.002)
    short = dict(BODY, max_tokens=400)
    assert (await transport(short))["ok"] and len(client.sent) == 1          # reserves about $0.0010
    long_prompt = dict(short, messages=[{"role": "user", "content": "word " * 20000}])
    refused = await transport(long_prompt)                                    # would reserve about $0.0025
    assert refused.get("refused") and len(client.sent) == 1
    assert orclient.worst_call_usd(400, 50000) == pytest.approx((50000 * 0.04 + 400 * 1.35) / 1e6)


async def test_one_process_at_a_time_may_use_a_ledger(tmp_path):
    """A second transport on the same ledger is refused until the first is closed."""
    spend = orclient.Spend(ledger=tmp_path / "ledger.jsonl", limit=1.0)
    first, client = orclient.open_transport(spend, "test", client=Client([Reply(200, ANSWER)]), key=KEY)
    with pytest.raises(SystemExit):
        orclient.open_transport(orclient.Spend(ledger=tmp_path / "ledger.jsonl"), "test", client=client, key=KEY)
    await orclient.close_transport(first, client)
    second, _ = orclient.open_transport(orclient.Spend(ledger=tmp_path / "ledger.jsonl"), "test", client=client, key=KEY)
    assert (await second(BODY))["ok"]


async def test_the_ledger_is_read_again_once_it_is_held(tmp_path):
    """Rows another run wrote between this run's start and its lock count against the limit."""
    ledger = tmp_path / "ledger.jsonl"
    spend = orclient.Spend(ledger=ledger, limit=0.05)
    ledger.write_text(json.dumps({"purpose": "elsewhere", "booked_usd": 0.04}) + "\n")
    transport, client, = orclient.open_transport(spend, "test", client=Client([Reply(200, ANSWER)]), key=KEY)
    assert spend.spent == pytest.approx(0.04)
    out = await transport(BODY)
    assert out.get("refused") and client.sent == []


async def test_an_unwritable_ledger_stops_further_sending(tmp_path):
    """A paid answer is still returned, and nothing more is sent once the ledger cannot be written."""
    transport, client, spend = transport_with(tmp_path, [Reply(200, ANSWER)])
    (tmp_path / "ledger.jsonl").mkdir()                      # a folder where the file should be
    first = await transport(BODY)
    assert first["ok"] and first["cost_usd"] == 0.0003 and spend.spent == pytest.approx(0.0003)
    assert "cannot be written" in spend.failure and spend.reserved == pytest.approx(0.0)
    second = await transport(BODY)
    assert second.get("refused") and "cannot be written" in second["error"] and len(client.sent) == 1


async def test_nothing_is_sent_when_the_limit_has_no_room(tmp_path):
    """A call whose worst case does not fit the limit is refused before any request."""
    transport, client, spend = transport_with(tmp_path, [Reply(200, ANSWER)], limit=0.001)
    out = await transport(BODY)
    assert not out["ok"] and out["refused"] and out["error"] == "spend cap reached" and out["attempts"] == 0
    assert client.sent == [] and spend.refused == 1 and not (tmp_path / "ledger.jsonl").exists()


def record(**changes):
    """A stored call record as `run_jobs` writes it, with some fields changed.

    Args: changes: fields to override.
    Returns: the record.
    Raises: nothing.
    Side Effects: none.
    """
    base = {"ok": True, "status": 200, "error": None, "host": "Near AI", "reasoning_chars": 0,
            "reasoning_tokens": 0}
    return dict(base, **changes)


def test_the_stop_rules(tmp_path):
    """Another host, a refused credential, early or lasting failures and reasoning each stop a run."""
    def guard():
        """A fresh guard.

        Args: none.
        Returns: the guard.
        Raises: nothing.
        Side Effects: none.
        """
        return orclient.Guard(orclient.Spend(ledger=tmp_path / "ledger.jsonl"))

    fine = guard()
    for _ in range(100):
        fine.record(record())
    assert fine.reason is None
    other = guard()
    other.record(record(host="Wafer"))
    assert "not the pinned host" in other.reason
    refused = guard()
    refused.record(record(ok=False, status=402, error="HTTP 402"))
    assert "refused" in refused.reason
    early = guard()
    for _ in range(orclient.EARLY_FAILURES + 1):
        early.record(record(ok=False, status=500, error="HTTP 500"))
    assert "calls failed" in early.reason
    late = guard()
    for _ in range(orclient.EARLY_WINDOW):
        late.record(record())
    for _ in range(orclient.EARLY_FAILURES + 5):
        late.record(record(ok=False, status=500, error="HTTP 500"))
    assert late.reason is None
    for _ in range(orclient.LATE_FAILURES - orclient.EARLY_FAILURES - 5):
        late.record(record(ok=False, status=None, error="TimeoutError"))
    assert late.reason is None                              # exactly half of the last 50
    late.record(record(ok=False, status=None, error="TimeoutError"))
    assert late.reason == "26 of the last 50 calls failed"
    mixed = guard()
    for _ in range(orclient.EARLY_WINDOW):
        mixed.record(record())
    for position in range(400):                             # later, one failure in three never stops a run
        mixed.record(record() if position % 3 else record(ok=False, status=500, error="HTTP 500"))
    assert mixed.reason is None
    thinking = guard()
    for _ in range(orclient.THINKING_LIMIT + 1):
        thinking.record(record(reasoning_tokens=40))
    assert "reasoning" in thinking.reason
    capped = guard()
    capped.stop("spend cap reached")
    capped.stop("something later")
    assert capped.reason == "spend cap reached"


async def test_run_jobs_stores_costs_resumes_and_keeps_cap_refusals_out(tmp_path):
    """Records carry the cost and the host; answered calls are not repeated; refusals are not stored."""
    jobs = engine.schedule(["a", "b"], ["v1", "v2"], 1, 7)
    seen = []

    async def transport(body):
        """Answer every request, except the third, which the cap refuses.

        Args: body: the request body.
        Returns: a result in the shape of `orclient.read`.
        Raises: nothing.
        Side Effects: appends to `seen`.
        """
        seen.append(body)
        if len(seen) == 3:
            return {"ok": False, "status": None, "error": "spend cap reached", "refused": True,
                    "latency": 0.0, "attempts": 0}
        return dict(orclient.read(200, json.dumps(ANSWER), KEY), latency=0.2, attempts=1)

    path = tmp_path / "calls.jsonl"
    guard = orclient.Guard(orclient.Spend(ledger=tmp_path / "ledger.jsonl"))
    first = await orclient.run_jobs(jobs, lambda job: "prompt " + job[0] + job[2], transport, path, guard, concurrency=1)
    assert first["issued"] == 2 and first["skipped"] == 2 and first["stopped"] == "spend cap reached"
    stored = engine.latest(path)
    assert len(stored) == 2 and all(r["cost_usd"] == 0.0003 and r["host"] == "Near AI" for r in stored.values())
    assert all("chat_template_kwargs" not in body and body["temperature"] == 0.5 for body in seen)
    seen.clear()
    again = orclient.Guard(orclient.Spend(ledger=tmp_path / "ledger.jsonl"))

    async def answering(body):
        """Answer every request.

        Args: body: the request body.
        Returns: a successful result.
        Raises: nothing.
        Side Effects: appends to `seen`.
        """
        seen.append(body)
        return dict(orclient.read(200, json.dumps(ANSWER), KEY), latency=0.2, attempts=1)

    second = await orclient.run_jobs(jobs, lambda job: "prompt", answering, path, again, concurrency=2)
    assert second["issued"] == 2 and second["stopped"] is None and len(engine.latest(path)) == 4


async def test_an_answer_from_another_host_is_never_kept(tmp_path):
    """It stops the run, is stored as a failed call, and is asked again when the run is resumed."""
    jobs = engine.schedule(["a"], ["v1", "v2", "v3"], 1, 7)
    elsewhere = json.loads(json.dumps(ANSWER))
    elsewhere["provider"] = "Wafer"

    async def wandering(body):
        """Answer from a host that is not the pinned one.

        Args: body: the request body.
        Returns: a successful result served by another host.
        Raises: nothing.
        Side Effects: none.
        """
        return dict(orclient.read(200, json.dumps(elsewhere), KEY), latency=0.2, attempts=1)

    path = tmp_path / "calls.jsonl"
    guard = orclient.Guard(orclient.Spend(ledger=tmp_path / "ledger.jsonl"))
    first = await orclient.run_jobs(jobs, lambda job: "prompt", wandering, path, guard, concurrency=1)
    assert first["issued"] == 1 and first["skipped"] == 2 and "not the pinned host" in first["stopped"]
    stored = engine.latest(path)
    assert len(stored) == 1 and all(not r["ok"] and r["error"] == "served by another host: Wafer"
                                    for r in stored.values())

    async def pinned(body):
        """Answer from the pinned host.

        Args: body: the request body.
        Returns: a successful result.
        Raises: nothing.
        Side Effects: none.
        """
        return dict(orclient.read(200, json.dumps(ANSWER), KEY), latency=0.2, attempts=1)

    again = orclient.Guard(orclient.Spend(ledger=tmp_path / "ledger.jsonl"))
    second = await orclient.run_jobs(jobs, lambda job: "prompt", pinned, path, again, concurrency=1)
    assert second["issued"] == 3 and second["stopped"] is None
    kept = engine.latest(path)
    assert len(kept) == 3 and all(r["ok"] and r["host"] == "Near AI" for r in kept.values())


def test_the_host_rule():
    """Only hosts with 8 bits or more, every needed parameter and a reliable last day pass, cheapest first."""
    def host(name, quantization="fp8", uptime=100.0, price=("0.00000004", "0.00000135"), drop=()):
        """One listing row.

        Args: name: the host; quantization: declared precision; uptime: last-day uptime; price:
            prompt and completion price per token; drop: parameters the host does not list.
        Returns: the row.
        Raises: nothing.
        Side Effects: none.
        """
        return {"provider_name": name, "tag": name.lower() + "/" + quantization, "quantization": quantization,
                "uptime_last_1d": uptime, "pricing": {"prompt": price[0], "completion": price[1]},
                "supported_parameters": [p for p in (*orclient.NEEDED_PARAMETERS, "stop") if p not in drop]}

    rows = orclient.eligible([host("Dear", price=("0.00000099", "0.00000149")), host("Cheap"),
                              host("FourBit", quantization="fp4"), host("Unknown", quantization="unknown"),
                              host("Flaky", uptime=99.4), host("NoLogprobs", drop=("logprobs",)),
                              host("NoUptime", uptime=None)])
    assert [row["name"] for row in rows] == ["Cheap", "Dear"]
    assert rows[0]["typical_call_usd"] == pytest.approx((4300 * 0.04 + 90 * 1.35) / 1e6)
