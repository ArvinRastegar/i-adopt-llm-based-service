import pandas as pd
import json
import csv

# --- CONFIG ---
file_path = "input.xlsx"
sheet_name = "LLM outputs"   # replace with the correct tab name
output_csv = "gt_validation_definition.csv"

hallucinations_metric = {}
hallucinations_metric["strict_min"] = 0
hallucinations_metric["strict_min"] = 0
hallucinations_metric["strict_min"] = 0


# --- LOAD DATA ---
df = pd.read_excel(file_path, sheet_name=sheet_name)

# --- FILTER ONLY strict_minimal ---
#df_filtered = df[df["prompt_version"] == "strict_min"].copy()

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
#df["ground_truth_obj"] = df["ground_truth_json"].apply(parse_json)
#df["predicted_obj"] = df["predicted_json"].apply(parse_json)

# --- HELPER: WHAT COUNTS AS "EMPTY"? ---
def is_empty(value):
    # You can adjust this definition if needed
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    if isinstance(value, (list, dict, set, tuple)) and len(value) == 0:
        return True
    return False

# --- STRUCTURE COMPARISON ---
def same_structure(gt, pred):
    #if not isinstance(gt, dict) or not isinstance(pred, dict):
    #    return False

    gt_keys = set(gt.keys())
    pred_keys = set(pred.keys())

    # Condition 1: same keys
    #if gt_keys != pred_keys:
    #    return False

    hallucinations = {}
    hallucinations["hasStatisticalModifier"] = 0
    hallucinations["hasProperty"] = 0
    hallucinations["hasObjectOfInterest"] = 0
    hallucinations["hasMatrix"] = 0
    hallucinations["hasContextObject"] = 0
    hallucinations["hasConstraints"] = 0

    keys = ["hasStatisticalModifier","hasProperty","hasObjectOfInterest","hasMatrix","hasContextObject","hasConstraints"]

    for key in keys:
        if key not in gt and key in pred:
            pred_val = pred[key]
            pred_empty = is_empty(pred_val)
            if not pred_empty:
                hallucinations[key] = 1
        elif key in gt and key in pred:
            gt_val = gt[key]
            pred_val = pred[key]

            gt_empty = is_empty(gt_val)
            pred_empty = is_empty(pred_val)

            if gt_empty and not pred_empty:
                hallucinations[key] = 1

    print(f"{hallucinations['hasStatisticalModifier']},{hallucinations['hasProperty']},{hallucinations['hasObjectOfInterest']},{hallucinations['hasMatrix']},{hallucinations['hasContextObject']},{hallucinations['hasConstraints']}")
    return [hallucinations["hasStatisticalModifier"],hallucinations["hasProperty"],hallucinations["hasObjectOfInterest"],hallucinations["hasMatrix"],hallucinations["hasContextObject"],hallucinations["hasConstraints"]]    

    #print(metric_value)



    return True
def validate_iadopt(obj):
    """
    Validates that required semantic elements appear inside the 'comment' value.
    Returns a dictionary with boolean validation results.
    """

    # Normalize comment for comparison
    comment = obj.get("definition", "")
    comment_lower = comment.lower()

    results = {
        "property_in_comment": False,
        "matrix_sources_in_comment": False,
        "object_of_interest_in_comment": False,
        "constraints_in_comment": False
    }

    # --- 1) hasProperty must appear in comment ---
    prop = obj.get("hasProperty")
    if isinstance(prop, str) and prop.lower() in comment_lower:
        results["property_in_comment"] = True

    # --- 2) hasSource and hasTarget (inside hasMatrix) must appear in comment ---
    matrix = obj.get("hasMatrix", {})
    sources_ok = True

    if isinstance(matrix, dict):
        src = matrix.get("hasSource")
        tgt = matrix.get("hasTarget")

        if isinstance(src, str) and src.lower() not in comment_lower:
            sources_ok = False
        if isinstance(tgt, str) and tgt.lower() not in comment_lower:
            sources_ok = False

    results["matrix_sources_in_comment"] = sources_ok

    # --- 3) hasObjectOfInterest must appear in comment ---
    obj_interest = obj.get("hasObjectOfInterest")
    if isinstance(obj_interest, str) and obj_interest.lower() in comment_lower:
        results["object_of_interest_in_comment"] = True

    # --- 4) Each constraint label must appear in comment ---
    constraints = obj.get("hasConstraint", [])
    constraints_ok = True

    if isinstance(constraints, list):
        for c in constraints:
            if isinstance(c, dict):
                label = c.get("label")
                if isinstance(label, str) and label.lower() not in comment_lower:
                    constraints_ok = False

    results["constraints_in_comment"] = constraints_ok

    return results
# --- LOOP INSTEAD OF LAMBDA ---
structure_results = []
all_rows = []

average_jaccard = 0


for idx, row in df.iterrows():
    gt_obj = row["ground_truth_obj"]

    results = validate_iadopt(gt_obj)

    row = {}

    if results["property_in_comment"]:
        row["property_in_comment"] = 1
    else:
        row["property_in_comment"] = 0
    
    if results["matrix_sources_in_comment"]:
        row["matrix_sources_in_comment"] = 1
    else:
        row["matrix_sources_in_comment"] = 0

    if results["object_of_interest_in_comment"]:
        row["object_of_interest_in_comment"] = 1
    else:
        row["object_of_interest_in_comment"] = 0


    if results["constraints_in_comment"]:
        row["constraints_in_comment"] = 1
    else:
        row["constraints_in_comment"] = 0


    all_rows.append(row)

    print("************************************************************************")
    print(gt_obj["comment"])
    print(results)


header = ["property_in_comment","matrix_sources_in_comment","object_of_interest_in_comment","constraints_in_comment"]

with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for row in all_rows:
        writer.writerow([row[h] for h in header])

# Optional: see which rows failed
#print(df_filtered[["ground_truth_json", "predicted_json", "structure_match"]])

