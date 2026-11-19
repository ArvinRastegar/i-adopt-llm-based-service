"""Fixtures for the guided-decomposition tests. No provider access; no database."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
for path in (HERE, HERE.parent / "example_selection", HERE.parent, ROOT / "src"):
    sys.path.insert(0, str(path))


@pytest.fixture(scope="session")
def records() -> list[dict]:
    """The real 102-row canonical corpus.

    Args: none.
    Returns: canonical corpus records.
    Raises: nothing.
    Side Effects: reads the corpus files.
    """
    from iadopt_lab.corpus.ingestion import load_canonical_records
    return list(load_canonical_records(ROOT))


@pytest.fixture(scope="session")
def trees() -> dict:
    """The built tree data.

    Args: none.
    Returns: the "trees" mapping of `data/trees.json`.
    Raises: OSError when the data has not been built.
    Side Effects: reads the data file.
    """
    import rulebook
    return rulebook.load("trees")["trees"]


@pytest.fixture(scope="session")
def patterns() -> dict:
    """The built pattern table.

    Args: none.
    Returns: {number as text: pattern}.
    Raises: OSError when the data has not been built.
    Side Effects: reads the data file.
    """
    import rulebook
    return rulebook.load("patterns")["patterns"]


@pytest.fixture(scope="session")
def key() -> dict:
    """The built expert answer key.

    Args: none.
    Returns: {variable_id: entry}.
    Raises: OSError when the data has not been built.
    Side Effects: reads the data file.
    """
    import route_key
    return route_key.load()
