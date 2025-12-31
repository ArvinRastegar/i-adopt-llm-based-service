import json
from difflib import SequenceMatcher
from pathlib import Path
import pandas as pd

# ---- Config ----
EXCEL_PATH = Path("input.xlsx")
SHEET_NAME = "LLM outputs"

GT_COL = "ground_truth_json"
PRED_COL = "predicted_json"

FIELD_NAME = "hasProperty"   # <-- target field
SIM_THRESHOLD = 0.8

# Optional: set to None to disable prompt_version filtering
PROMPT_COL = "prompt_version"
PROMPT_FILTER_VALUE = "strict_minimal"

OUT_XLSX = Path("hasProperty_similarity_matches.xlsx")


# ---- Helpers ----
def safe_load_json(cell):
    if pd.isna(cell) or str(cell).strip() == "":
        return None
    try:
        return json.loads(cell)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid JSON cell: {e}\nCell content (truncated): {str(cell)[:200]}"
        )

def normalize_text(s: str) -> str:
    return " ".join(str(s).strip().split()).lower()

def text_similarity(a: str, b: str) -> float:
    a_n, b_n = normalize_text(a), normalize_text(b)
    if not a_n or not b_n:
        return 0.0
    return SequenceMatcher(None, a_n, b_n).ratio()

def iter_string_values(obj, path=""):
    """Yield (path, string_value) for every string found anywhere inside obj."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_path = f"{path}.{k}" if path else str(k)
            yield from iter_string_values(v, new_path)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            new_path = f"{path}[{i}]"
            yield from iter_string_values(v, new_path)
    else:
        if isinstance(obj, str):
            yield path, obj

def gt_field_as_text(gt_obj, field_name: str):
    """Return the GT field as text, or None if missing."""
    if not isinstance(gt_obj, dict):
        return None
    val = gt_obj.get(field_name)
    if val is None:
        return None
    if isinstance(val, str):
        return val
    # stringify lists/dicts/numbers deterministically
    return json.dumps(val, ensure_ascii=False)

def best_match_in_pred(gt_text: str, pred_obj):
    """Return (best_path, best_value, best_score)."""
    best_path, best_value, best_score = None, None, 0.0
    if not gt_text or pred_obj is None:
        return best_path, best_value, best_score

    for path, val in iter_string_values(pred_obj):
        score = text_similarity(gt_text, val)
        if score > best_score:
            best_path, best_value, best_score = path, val, score

    return best_path, best_value, best_score


def main():
    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)

    # Optional filter by prompt_version
    if PROMPT_FILTER_VALUE is not None:
        if PROMPT_COL not in df.columns:
            raise KeyError(f"'{PROMPT_COL}' column not found, cannot filter.")
        df = df[df[PROMPT_COL] == PROMPT_FILTER_VALUE].copy()

    rows_out = []

    for idx, row in df.iterrows():
        gt_obj = safe_load_json(row.get(GT_COL))
        pred_obj = safe_load_json(row.get(PRED_COL))

        gt_text = gt_field_as_text(gt_obj, FIELD_NAME)
        best_path, best_val, best_score = best_match_in_pred(gt_text, pred_obj)

        # Keep only matches strictly greater than threshold
        if best_score > SIM_THRESHOLD:
            out_row = {
                "row_index": int(idx),
                FIELD_NAME: gt_text,
                "pred_best_match_path": best_path,
                "pred_best_match_value": best_val,
                "pred_best_match_similarity": best_score,
            }
            if PROMPT_FILTER_VALUE is not None:
                out_row[PROMPT_COL] = row.get(PROMPT_COL)
            rows_out.append(out_row)

    out_df = pd.DataFrame(rows_out)

    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        out_df.to_excel(writer, index=False, sheet_name="matches")

    print(f"Wrote {len(out_df)} rows to: {OUT_XLSX}")

if __name__ == "__main__":
    main()
