import pandas as pd
import json
import csv

# --- CONFIG ---
file_path = "input.xlsx"
sheet_name = "LLM outputs"   # replace with the correct tab name
output_csv = "hallucination_results.csv"

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

# --- LOOP INSTEAD OF LAMBDA ---
structure_results = []
all_rows = []

average_jaccard = 0

print("hasStatisticalModifier,hasProperty,hasObjectOfInterest,hasMatrix,hasContextObject,hasConstraints")

for idx, row in df.iterrows():
    gt_obj = row["ground_truth_obj"]
    pred_obj = row["predicted_obj"]

    row_values = same_structure(gt_obj, pred_obj)

    all_rows.append(row_values) 

header = ["hasStatisticalModifier","hasProperty","hasObjectOfInterest","hasMatrix","hasContextObject","hasConstraints"]    

with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(all_rows)


# Optional: see which rows failed
#print(df_filtered[["ground_truth_json", "predicted_json", "structure_match"]])

