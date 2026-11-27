"""Offline tests for fill_levers_test.py: the note-first prompt and one unit's calls, with a fake transport."""

from __future__ import annotations

import asyncio
import json

import fill_levers_test as fl
import knowledge as kn
import pytest
import rag_tuning as rt

GOOD = json.dumps({"hasStatisticalModifier": "", "hasProperty": "count", "hasObjectOfInterest": "night", "hasMatrix": "hotel",
                   "hasContextObject": "", "hasConstraint": []})


def test_note_first_replaces_both_instructions():
    """Both output instructions are replaced once; a prompt without them is refused."""
    prompt = f"rules\n{fl.PLAIN_HEAD}\nexamples\nTARGET\n{fl.PLAIN_TAIL}\n"
    changed = fl.note_first(prompt)
    assert fl.NOTE_HEAD in changed and fl.NOTE_TAIL in changed
    assert fl.PLAIN_HEAD not in changed and fl.PLAIN_TAIL not in changed
    assert changed.replace(fl.NOTE_HEAD, fl.PLAIN_HEAD).replace(fl.NOTE_TAIL, fl.PLAIN_TAIL) == prompt
    with pytest.raises(ValueError):
        fl.note_first("no instructions here")
    with pytest.raises(ValueError):
        fl.note_first(prompt + fl.PLAIN_TAIL)


@pytest.fixture(scope="module")
def ctx2():
    """Stage 5's context, for the schema check, with a word-overlap table instead of the stored one.

    Args: none.
    Returns: the context.
    Raises: as the loader.
    Side Effects: reads local files.
    """
    def overlap(first: str, second: str) -> float:
        """Shared words over all words.

        Args: first, second: texts.
        Returns: the share.
        Raises: nothing.
        Side Effects: none.
        """
        a, b = set(kn.words(first)), set(kn.words(second))
        return len(a & b) / len(a | b) if a | b else 0.0

    base = kn.load()
    return rt.load(table=rt.similarity_table(base["ids"], lambda v: base["records"][v]["definition"], overlap))


def fake(answers):
    """A transport that answers in order and remembers the requests.

    Args: answers: replies in the order they will be given.
    Returns: (transport, sent requests).
    Raises: nothing.
    Side Effects: none until called.
    """
    sent, queue = [], list(answers)

    async def transport(body):
        sent.append(body)
        return {"ok": True, "status": 200, "answer": queue.pop(0), "latency": 1.0, "served_model": body["model"]}

    return transport, sent


def test_unit_without_a_re_ask(ctx2):
    """A usable first answer: four calls, and `best+retry` is that answer."""
    prompt = f"x\n{fl.PLAIN_HEAD}\ny\n{fl.PLAIN_TAIL}"
    transport, sent = fake([GOOD, GOOD, GOOD, "note\n" + GOOD])
    records = asyncio.run(fl.unit(ctx2, prompt, 1, "v", transport))
    assert [r["arm"] for r in records] == ["best", "best#2", "best#3", "best+form", "best+retry"]
    assert len(sent) == 4 and sent[3]["messages"][0]["content"] == fl.note_first(prompt)
    assert records[-1]["answer"] == GOOD and records[-1]["repaired"] is False
    assert all(s["temperature"] == 0.5 and s["model"] == "Qwen3.8-27B" for s in sent)


def test_unit_with_a_re_ask(ctx2):
    """An unusable first answer goes back once; a usable repair replaces it, and its time adds up."""
    prompt = f"x\n{fl.PLAIN_HEAD}\ny\n{fl.PLAIN_TAIL}"
    transport, sent = fake(["not json", GOOD, GOOD, GOOD, GOOD])
    records = asyncio.run(fl.unit(ctx2, prompt, 1, "v", transport))
    retry = records[-1]
    assert len(sent) == 5 and len(sent[4]["messages"]) == 3 and sent[4]["messages"][1]["content"] == "not json"
    assert retry["repaired"] is True and retry["repair_valid"] is True and retry["answer"] == GOOD
    assert retry["latency_total"] == 2.0 and records[0]["answer"] == "not json"
    transport, sent = fake(["not json", GOOD, GOOD, GOOD, "still not json"])
    retry = asyncio.run(fl.unit(ctx2, prompt, 1, "v", transport))[-1]
    assert retry["repair_valid"] is False and retry["answer"] == "not json"
