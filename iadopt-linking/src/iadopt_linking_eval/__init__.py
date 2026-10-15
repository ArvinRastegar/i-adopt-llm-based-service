"""Pure, evidence-preserving I-ADOPT link evaluation.

Same contract as ``iadopt_eval`` in iadopt-lab: no experiment, network, database, clock or
filesystem dependency, and the standard library only, so iadopt-lab can import it later without
inheriting this project's dependencies. Everything the scorer needs is supplied by the caller.

Protocol (see DECISIONS.md): a prediction is correct when its (vocabulary, local identifier)
equals the gold's; metrics are numerator/denominator receipts.
"""

from .identity import (
    NON_VOCABULARY,
    REACHABLE,
    UNKNOWN,
    Identity,
    classify,
    is_controlled_vocabulary,
)
from .metrics import SCORER_VERSION, choose, evaluate_slot, summarise

__all__ = [
    "NON_VOCABULARY", "REACHABLE", "SCORER_VERSION", "UNKNOWN", "Identity", "choose", "classify",
    "evaluate_slot", "is_controlled_vocabulary", "summarise",
]
