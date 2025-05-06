import os
import pandas as pd
import json

# Path to your CSV file
csv_path = "/Users/rastegar-a/Documents/GitHub/i-adopt-llm-based-service/benchmarking_example/data/Challenge variable descriptions version 2 - agreed solutions 2.csv"

# Output folder
output_folder = "/Users/rastegar-a/Documents/GitHub/i-adopt-llm-based-service/benchmarking_example/data"
os.makedirs(output_folder, exist_ok=True)

# Read the CSV
df = pd.read_csv(csv_path)

# Forward-fill the first column to group related rows
df.iloc[:, 0] = df.iloc[:, 0].fillna(method="ffill")

# Get column names
columns = df.columns.tolist()

# Group by first column (variable ID or index)
grouped = df.groupby(columns[0])

for var_id, group in grouped:
    # Choose the variable name from the first non-null "Variable Name" entry
    var_name = str(group.iloc[0].get("Variable Name", f"variable_{var_id}")).strip()
    var_name_clean = var_name.replace(" ", "_").replace("/", "_") or f"variable_{var_id}"

    json_filename = f"{var_name_clean}.json"
    json_path = os.path.join(output_folder, json_filename)

    # Create a dictionary where each key is a column name
    # and each value is:
    # - a single string if only one unique non-null value
    # - a list if multiple unique non-null values
    variable_dict = {}
    for col in columns[1:]:  # Skip the index/grouping column
        values = group[col].dropna().unique().tolist()
        if not values:
            continue
        variable_dict[col] = values[0] if len(values) == 1 else values

    # Save JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(variable_dict, f, indent=2, ensure_ascii=False)

print(f"Created {len(grouped)} JSON files in {output_folder}")
