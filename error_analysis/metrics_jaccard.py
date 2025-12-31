import pandas as pd
import json

# --- CONFIG ---
file_path = "input.xlsx"
sheet_name = "LLM outputs"   # replace with the correct tab name

# --- LOAD DATA ---
df = pd.read_excel(file_path, sheet_name=sheet_name)

# --- FILTER ONLY strict_minimal ---
df_filtered = df[df["prompt_version"] == "constraint_first"].copy()

# --- PARSE JSON COLUMNS ---
def parse_json(x):
    if pd.isna(x):
        return None
    try:
        return json.loads(x)
    except Exception:
        return None  # or raise if you prefer

df_filtered["ground_truth_obj"] = df_filtered["ground_truth_json"].apply(parse_json)
df_filtered["predicted_obj"] = df_filtered["predicted_json"].apply(parse_json)
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

    num_intersection = 0
    for key in gt_keys:
        gt_val = gt[key]
        if key in pred:
            pred_val = pred[key]

            gt_empty = is_empty(gt_val)
            pred_empty = is_empty(pred_val)

            # Condition 2: empty in ground truth → empty in predicted
            if not gt_empty and not pred_empty:
                num_intersection = num_intersection + 1

            # Condition 3: non-empty in GT → non-empty in predicted
            if gt_empty and pred_empty:
                num_intersection = num_intersection + 1
    
    metric_value =  num_intersection / len(gt_keys)
    formatted = f"{metric_value:.2f}".replace(".", ",")
    print(formatted)
    return metric_value
    #print(metric_value)



    return True

# --- LOOP INSTEAD OF LAMBDA ---
structure_results = []

average_jaccard = 0

for idx, row in df_filtered.iterrows():
    gt_obj = row["ground_truth_obj"]
    pred_obj = row["predicted_obj"]

    match = same_structure(gt_obj, pred_obj)

    average_jaccard = average_jaccard + match

print(average_jaccard/100)

# Optional: see which rows failed
#print(df_filtered[["ground_truth_json", "predicted_json", "structure_match"]])

