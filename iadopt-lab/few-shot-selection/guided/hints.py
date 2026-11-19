"""Pattern guidance for one variable: what a walk landed on, and how the prompt says it.

EXPLORATORY side experiment (guided decomposition). See `rulebook.py` for status and isolation.

"Facts" are the patterns a tree walk ends on: one property pattern, the object-of-interest and
matrix patterns, whether there is a context object or a statistical modifier, and the constraint
types. In the ceiling test they come from the expert answer key; later they come from the model's
own walk. `render` turns facts into the text block appended to the target definition.
"""

from __future__ import annotations

from typing import Any

FEATURES = ("form", "names", "types", "texts")
SHAPE_TEXT = {"single": "one term.",
              "ratio": "an asymmetric system with hasNumerator and hasDenominator.",
              "symmetric": "a symmetric system with hasPart.",
              "source_target": "an asymmetric system with hasSource and hasTarget."}
NONE_TEXT = 'none; use "".'
HEADING = "PATTERN GUIDANCE FOR THIS DEFINITION (not part of the definition)"
STRONG_SHAPES = ("ratio", "symmetric", "source_target")


def facts_from_key(entry: dict[str, Any]) -> dict[str, Any]:
    """Read the facts of one variable from its expert answer-key entry.

    Args: entry: an answer-key entry from `route_key.load`.
    Returns: {"property", "object", "matrix", "context", "constraint": pattern numbers,
        "statistical_modifier": bool, "names": {number: tag name}}.
    Raises: KeyError when the entry lacks a field.
    Side Effects: none. Deterministic.
    """
    return {"property": list(entry["property"]), "object": list(entry["object"]),
            "matrix": list(entry["matrix"]), "context": list(entry["context"]),
            "constraint": list(entry["constraint"]),
            "statistical_modifier": bool(entry["statistical_modifier"]),
            "names": dict(entry.get("names", {}))}


def object_shape(facts: dict[str, Any], patterns: dict[str, Any]) -> str | None:
    """The object-of-interest shape the facts imply.

    Args: facts: from `facts_from_key` or a walk; patterns: the pattern table.
    Returns: a shape name, or None when neither an object pattern nor the property pattern says.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    shapes = [patterns.get(str(n), {}).get("shape", "single") for n in facts["object"]]
    for strong in STRONG_SHAPES:
        if strong in shapes:
            return strong
    if shapes:
        return "single"
    for number in facts["property"]:
        shape = patterns.get(str(number), {}).get("object_shape")
        if shape:
            return "single" if shape == "variable" else shape
    return None


def matrix_shape(facts: dict[str, Any], patterns: dict[str, Any]) -> str:
    """The matrix shape the facts imply.

    Args: facts: as above; patterns: the pattern table.
    Returns: "empty", "single" or "source_target".
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    if not facts["matrix"]:
        return "empty"
    shapes = [patterns.get(str(n), {}).get("shape", "single") for n in facts["matrix"]]
    return "source_target" if "source_target" in shapes else "single"


def constraint_names(facts: dict[str, Any], types: list[dict[str, Any]]) -> list[str]:
    """The constraint types of the facts, by their name in the rulebook table.

    Args: facts: as above; types: the constraint-type table.
    Returns: sorted distinct names; a number missing from the table keeps its tag name.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    by_number = {row["number"]: row["name"].lower() for row in types}
    return sorted({by_number.get(n) or facts["names"].get(str(n), str(n)) for n in facts["constraint"]})


def expected_form(facts: dict[str, Any], patterns: dict[str, Any], types: list[dict[str, Any]]) -> dict[str, Any]:
    """The form a decomposition must have to fit the facts.

    Args: facts, patterns, types: as above.
    Returns: {"object": shape or None, "matrix": shape, "context": bool,
        "statistical_modifier": bool, "constraint_types": [names]}.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    return {"object": object_shape(facts, patterns), "matrix": matrix_shape(facts, patterns),
            "context": bool(facts["context"]), "statistical_modifier": facts["statistical_modifier"],
            "constraint_types": constraint_names(facts, types)}


def _pattern_line(part: str, number: int, patterns: dict[str, Any], names: dict[str, str], texts: bool) -> str:
    """One line naming a pattern, with its rulebook text when asked.

    Args: part: the part's name; number: pattern number; patterns: the pattern table; names: tag
        names for numbers the table lacks; texts: add definition and design.
    Returns: the line.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    pattern = patterns.get(str(number), {})
    line = f"- {part} pattern {number}: {pattern.get('name') or names.get(str(number), '')}."
    if texts:
        for field in ("definition", "design"):
            text = (pattern.get(field) or "").strip()
            if text:
                line += " " + text + ("" if text.endswith(".") else ".")
    return line


def render(facts: dict[str, Any], features: frozenset[str] | set[str], patterns: dict[str, Any],
           types: list[dict[str, Any]]) -> str:
    """Write the guidance block for one variable.

    Args: facts: what the walk landed on; features: any of "form", "names", "types", "texts";
        patterns, types: the rulebook data.
    Returns: the block, starting with two newlines so it can follow the definition; "" when no
        feature is asked for.
    Raises: ValueError for an unknown feature.
    Side Effects: none. Deterministic.
    """
    unknown = set(features) - set(FEATURES)
    if unknown:
        raise ValueError(f"unknown guidance features: {sorted(unknown)}")
    if not features:
        return ""
    form = expected_form(facts, patterns, types)
    lines = ["", "", HEADING]
    if "form" in features or "types" in features:
        lines.append("The decomposition must have this form:")
    if "form" in features:
        if form["object"]:
            lines.append(f"- hasObjectOfInterest: {SHAPE_TEXT[form['object']]}")
        lines.append(f"- hasMatrix: {NONE_TEXT if form['matrix'] == 'empty' else SHAPE_TEXT[form['matrix']]}")
        lines.append(f"- hasContextObject: {'one term.' if form['context'] else NONE_TEXT}")
        lines.append(f"- hasStatisticalModifier: {'present.' if form['statistical_modifier'] else NONE_TEXT}")
    if "types" in features:
        names = form["constraint_types"]
        if names:
            lines.append("- hasConstraint: one or more constraints of each of these types and of no "
                         f"other type: {'; '.join(names)}. Each label starts with its type and a colon.")
        else:
            lines.append("- hasConstraint: none; use [].")
    if "names" in features:
        lines.append("It follows these I-ADOPT patterns:")
        texts = "texts" in features
        for part, key in (("Property", "property"), ("Object of interest", "object"),
                          ("Matrix", "matrix"), ("Context object", "context")):
            for number in facts[key]:
                lines.append(_pattern_line(part, number, patterns, facts["names"], texts))
    if "texts" in features and "types" in features and form["constraint_types"]:
        by_name = {row["name"].lower(): row for row in types}
        lines.append("Constraint types:")
        for name in form["constraint_types"]:
            definition = (by_name.get(name, {}).get("definition") or "").strip()
            lines.append(f"- {name}: {definition}" if definition else f"- {name}")
    return "\n".join(lines)


def conformance(prediction: dict[str, Any] | None, form: dict[str, Any]) -> dict[str, bool]:
    """Whether a decomposition has the form the facts demand, part by part.

    Args: prediction: a valid six-field decomposition, or None for an unusable answer; form: from
        `expected_form`.
    Returns: {"object", "matrix", "context", "statistical_modifier", "constraint_types", "all"}.
    Raises: nothing.
    Side Effects: none. Deterministic.
    """
    import route_key

    if prediction is None:
        return {name: False for name in ("object", "matrix", "context", "statistical_modifier",
                                         "constraint_types", "all")}
    types = sorted({route_key.constraint_type(c["label"]) for c in prediction["hasConstraint"]})
    out = {"object": form["object"] is None or route_key.shape(prediction["hasObjectOfInterest"]) == form["object"],
           "matrix": route_key.shape(prediction["hasMatrix"]) == form["matrix"],
           "context": bool(prediction["hasContextObject"]) == form["context"],
           "statistical_modifier": bool(prediction["hasStatisticalModifier"]) == form["statistical_modifier"],
           "constraint_types": types == form["constraint_types"]}
    out["all"] = all(out.values())
    return out
