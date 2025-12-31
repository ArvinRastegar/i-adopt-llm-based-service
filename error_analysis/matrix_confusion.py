import json
import pandas as pd
import json
import csv
from collections import Counter

# --- CONFIG ---
file_path = "input.xlsx"
sheet_name = "LLM outputs"   # replace with the correct tab name

# --- LOAD DATA ---
df = pd.read_excel(file_path, sheet_name=sheet_name)

# --- FILTER ONLY strict_minimal ---
df_filtered = df[df["prompt_version"] == "strict_min"].copy()

# --- PARSE JSON COLUMNS ---
def parse_json(x):
    if pd.isna(x):
        return None
    try:
        return json.loads(x)
    except Exception:
        return None  # or raise if you prefer

df["ground_truth_obj"] = df["ground_truth_json"].apply(parse_json)
df["predicted_obj"] = df["predicted_json"].apply(parse_json)

def normalize(s: str) -> str:
    return s.strip()


def summarize_hasProperty_results(results):
    counter = Counter()

    for r in results:
        if not isinstance(r, dict):
            continue
        for category in r.get("found_in", []):
            counter[category] += 1

    return dict(counter)

def value_exists_anywhere(target: str, data) -> bool:
    """Recursive search (used for hasProperty and hasObjectOfInterest buckets)."""
    if isinstance(data, dict):
        return any(value_exists_anywhere(target, v) for v in data.values())
    if isinstance(data, list):
        return any(value_exists_anywhere(target, item) for item in data)
    return isinstance(data, str) and normalize(data) == target

def locate_gt_hasProperty(gt_json, pred_json):
    gt_prop = gt_json.get("hasProperty")
    if not isinstance(gt_prop, str) or not gt_prop.strip():
        return {"found": False, "reason": "GT has no valid hasProperty string"}

    target = normalize(gt_prop)
    found_in = []
    details = {}

    # 1) hasProperty (anywhere inside pred["hasProperty"])
    if "hasProperty" in pred_json and value_exists_anywhere(target, pred_json["hasProperty"]):
        found_in.append("hasProperty")
        details["hasProperty"] = "matched somewhere inside predicted hasProperty"

    # 2) hasObjectOfInterest (anywhere inside pred["hasObjectOfInterest"])
    if "hasObjectOfInterest" in pred_json and value_exists_anywhere(target, pred_json["hasObjectOfInterest"]):
        found_in.append("hasObjectOfInterest")
        details["hasObjectOfInterest"] = "matched somewhere inside predicted hasObjectOfInterest"

    # 3) hasMatrix (ONLY pred["hasMatrix"]["hasSource"])
    matrix = pred_json.get("hasMatrix")
    if isinstance(matrix, dict):
        src = matrix.get("hasSource")
        if isinstance(src, str) and normalize(src) == target:
            found_in.append("hasMatrix")
            details["hasMatrix"] = "matched pred.hasMatrix.hasSource"

    # 4) hasConstraint (ONLY each constraint's "label")
    constraints = pred_json.get("hasConstraint")
    if isinstance(constraints, list):
        hits = []
        for i, c in enumerate(constraints):
            if isinstance(c, dict):
                lbl = c.get("label")
                if isinstance(lbl, str) and normalize(lbl) == target:
                    hits.append(i)
        if hits:
            found_in.append("hasConstraint")
            details["hasConstraint"] = [f"matched pred.hasConstraint[{i}].label" for i in hits]

    return {
        "hasProperty": target,
        "found": bool(found_in),
        "found_in": found_in,   # only among: hasProperty, hasMatrix, hasObjectOfInterest, hasConstraint
        "details": details      # optional extra info
    }

def locate_gt_hasConstraint(gt_json, pred_json):
    # Safety checks
    if not isinstance(gt_json, dict):
        return {"found": False, "reason": "GT JSON missing or not an object"}
    if not isinstance(pred_json, dict):
        return {"found": False, "reason": "Predicted JSON missing or not an object"}

    gt_constraints = gt_json.get("hasConstraint")
    if not isinstance(gt_constraints, list) or len(gt_constraints) == 0:
        return {"found": False, "reason": "GT has no hasConstraint list"}

    results = []
    all_found = True

    # Pre-extract predicted constraint labels (only allowed location)
    pred_constraint_labels = []
    pred_constraints = pred_json.get("hasConstraint")
    if isinstance(pred_constraints, list):
        for i, c in enumerate(pred_constraints):
            if isinstance(c, dict) and isinstance(c.get("label"), str):
                pred_constraint_labels.append((i, normalize(c["label"])))

    # Pre-extract pred hasMatrix.hasSource (only allowed location)
    pred_matrix_source = None
    matrix = pred_json.get("hasMatrix")
    if isinstance(matrix, dict) and isinstance(matrix.get("hasSource"), str):
        pred_matrix_source = normalize(matrix["hasSource"])

    for j, gc in enumerate(gt_constraints):
        if not isinstance(gc, dict):
            results.append({
                "gt_index": j,
                "found": False,
                "reason": "GT constraint is not an object"
            })
            all_found = False
            continue

        gt_label = gc.get("label")
        if not isinstance(gt_label, str) or not gt_label.strip():
            results.append({
                "gt_index": j,
                "found": False,
                "reason": "GT constraint has no valid label"
            })
            all_found = False
            continue

        target = normalize(gt_label)
        found_in = []
        details = {}

        # 1) pred.hasConstraint[*].label
        hits = [i for (i, lbl) in pred_constraint_labels if lbl == target]
        if hits:
            found_in.append("hasConstraint")
            details["hasConstraint"] = [f"matched pred.hasConstraint[{i}].label" for i in hits]

        # 2) pred.hasMatrix.hasSource
        if pred_matrix_source is not None and pred_matrix_source == target:
            found_in.append("hasMatrix")
            details["hasMatrix"] = "matched pred.hasMatrix.hasSource"

        # 3) pred.hasProperty (anywhere inside)
        if "hasProperty" in pred_json and value_exists_anywhere(target, pred_json["hasProperty"]):
            found_in.append("hasProperty")
            details["hasProperty"] = "matched somewhere inside predicted hasProperty"

        # 4) pred.hasObjectOfInterest (anywhere inside)
        if "hasObjectOfInterest" in pred_json and value_exists_anywhere(target, pred_json["hasObjectOfInterest"]):
            found_in.append("hasObjectOfInterest")
            details["hasObjectOfInterest"] = "matched somewhere inside predicted hasObjectOfInterest"

        found = bool(found_in)
        if not found:
            all_found = False

        results.append({
            "gt_index": j,
            "gt_constraint": {"label": target, "on": gc.get("on")},
            "found": found,
            "found_in": found_in,
            "details": details
        })

    return {
        "found": all_found,              # True only if every GT constraint label was found somewhere allowed
        "n_gt_constraints": len(gt_constraints),
        "n_found": sum(1 for r in results if r.get("found")),
        "results": results
    }

list_results = []
for idx, row in df.iterrows():
    gt_obj = row["ground_truth_obj"]
    pred_obj = row["predicted_obj"]

    results = locate_gt_hasProperty(gt_obj,pred_obj)

    list_results.append(results)

print(results)
found_in_counter = Counter()

for row_result in list_results:
    for r in row_result.get("results", []):
        for loc in r.get("found_in", []):
            found_in_counter[loc] += 1

found_in_summary = dict(found_in_counter)
print(found_in_summary)




