import pandas as pd
import json

# --- CONFIG ---
file_path = "input.xlsx"
sheet_name = "LLM outputs"   # replace with the correct tab name

# --- LOAD DATA ---
df = pd.read_excel(file_path, sheet_name=sheet_name)

# --- FILTER ONLY strict_minimal ---
df_filtered = df[df["prompt_version"] == "strict_minimal"].copy()

# --- PARSE JSON COLUMNS ---
def parse_json(x):
    try:
        return json.loads(x)
    except:
        return None  # or raise an error

df_filtered["ground_truth_obj"] = df_filtered["ground_truth_json"].apply(parse_json)
df_filtered["predicted_obj"] = df_filtered["predicted_json"].apply(parse_json)

# --- COMPARE JSON OBJECTS ---
def compare_json(a, b):
    return a == b

df_filtered["match"] = df_filtered.apply(
    lambda row: compare_json(row["ground_truth_obj"], row["predicted_obj"]),
    axis=1
)

# --- RESULT ---
print(df_filtered[["ground_truth_json", "predicted_json", "match"]])
