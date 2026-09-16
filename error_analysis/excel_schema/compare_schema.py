#!/usr/bin/env python3
"""Programmatic schema comparison: generated workbook vs. the template.

Exits non-zero if any structural difference is found.
    python compare_schema.py [OLD.xlsx] [NEW.xlsx]
"""
import pandas as pd, openpyxl, json, sys, pathlib

HERE = pathlib.Path(__file__).resolve().parent
OLD = sys.argv[1] if len(sys.argv) > 1 else str(HERE / "onlyPhaseOne20251208_142819.xlsx")
NEW = sys.argv[2] if len(sys.argv) > 2 else str(HERE / "onlyPhaseOne_glm52_from_database.xlsx")
PROV = pathlib.Path(NEW).with_suffix(".provenance.json")
fails = []
def chk(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond: fails.append(msg)

wo = openpyxl.load_workbook(OLD); wn = openpyxl.load_workbook(NEW)
print("1-2. SHEETS")
chk(wo.sheetnames == wn.sheetnames, f"sheet names & order identical: {wn.sheetnames}")

print("\n3-4,8-11. COLUMNS per sheet")
for s in wo.sheetnames:
    ho = [c.value for c in wo[s][1]]; hn = [c.value for c in wn[s][1]]
    chk(ho == hn, f"[{s}] {len(ho)} columns, exact names+order match")
    if ho != hn:
        print("        old:", ho); print("        new:", hn)

print("\n11. NO ACCIDENTAL INDEX COLUMN")
for s in wo.sheetnames:
    first = wn[s].cell(1,1).value
    chk(first not in (None, "", "Unnamed: 0", "index"), f"[{s}] first header is {first!r}")

print("\n5. DATA TYPES / CELL REPRESENTATION")
do = pd.read_excel(OLD, sheet_name="LLM outputs"); dn = pd.read_excel(NEW, sheet_name="LLM outputs")
diffs = [f"{c}:{do[c].dtype}->{dn[c].dtype}" for c in do.columns if str(do[c].dtype)!=str(dn[c].dtype)]
chk(not diffs, "[LLM outputs] all column dtypes match" + ("" if not diffs else f" -- {diffs}"))
# Summary metric cells: compare the written cell types, not pandas' value-inferred
# column dtype (a column of whole numbers infers int64 in either workbook).
def celltypes(p):
    ws = openpyxl.load_workbook(p)["Summary"]
    return {type(ws.cell(r,c).value).__name__ for r in range(2,ws.max_row+1) for c in range(5,47)}
chk(celltypes(OLD)==celltypes(NEW), f"[Summary] metric cell types identical: {sorted(celltypes(NEW))}")
so_=pd.read_excel(OLD,sheet_name="Summary"); sn_=pd.read_excel(NEW,sheet_name="Summary")
chk(all(str(so_[c].dtype)==str(sn_[c].dtype) for c in list(so_.columns)[:4]),
    "[Summary] key columns (Model/Temperature/PromptVersion/Shot) dtypes match")
chk(all(pd.api.types.is_numeric_dtype(sn_[c]) for c in list(sn_.columns)[4:]),
    "[Summary] all 42 metric columns numeric")

print("\n6. SUMMARY SCHEMA")
so = pd.read_excel(OLD, sheet_name="Summary"); sn = pd.read_excel(NEW, sheet_name="Summary")
chk(list(so.columns)==list(sn.columns), f"Summary has identical {len(sn.columns)} columns")
chk(len(sn)==3, f"Summary has 3 rows (one per prompt version), got {len(sn)}")
chk(sn["F_exact"].tolist()==sorted(sn["F_exact"].tolist(), reverse=True), "Summary sorted by F_exact descending")

print("\n7. FORMULAS")
def formulas(wb): return [(s,c.coordinate,c.value) for s in wb.sheetnames for r in wb[s].iter_rows() for c in r if isinstance(c.value,str) and c.value.startswith("=")]
chk(len(formulas(wo))==len(formulas(wn))==0, "neither workbook contains formulas (all values precomputed)")

print("\n12-14. CONTENT")
dn = pd.read_excel(NEW, sheet_name="LLM outputs")
chk(set(dn["model"].unique())=={"z-ai/glm-5.2"}, f"single model only: {list(dn['model'].unique())}")
pv = dn["prompt_version"].value_counts().to_dict()
chk(len(pv)==3, f"exactly 3 prompt versions: {pv}")
# Each prompt_version block must carry exactly one temperature/shot pairing:
# the fixed-grid export uses one temperature throughout, the best-configuration
# export uses each variant's own best temperature.
grp = dn.groupby("prompt_version")[["temperature","shot"]].nunique()
chk((grp["temperature"]==1).all() and (grp["shot"]==1).all(),
    "each prompt_version block has exactly one temperature and one shot count")
combos = sorted({(r.prompt_version, r.temperature, r.shot) for r in dn.itertuples()})
print("          grid: " + "; ".join(f"{a} @ temp {b}/{c}-shot" for a,b,c in combos))
chk(not dn.duplicated(subset=["variable","model","temperature","prompt_version","shot"]).any(),
    "no duplicate variable+model+temperature+prompt_version+shot combinations")

print("\n15. GT/PRED PAIRING (against DB provenance)")
prov = json.loads(PROV.read_text())
chk(len(prov["rows"])==len(dn), f"provenance rows == workbook rows ({len(dn)})")
pairs_wb = list(zip(dn["variable"], dn["prompt_version"]))
pairs_pv = [(r["variable"], r["prompt_version"]) for r in prov["rows"]]
chk(pairs_wb==pairs_pv, "row order matches provenance (variable+prompt_version, in order)")
chk(len({r["prediction_id"] for r in prov["rows"]})==len(prov["rows"]), "every row has a distinct prediction_id")

print("\n16. JSON PARSEABILITY")
for col in ["ground_truth_json","predicted_json"]:
    bad = 0
    for v in dn[col]:
        try: json.loads(v)
        except Exception: bad += 1
    chk(bad==0, f"[LLM outputs.{col}] all {len(dn)} cells parse as JSON")
for s in [x for x in wn.sheetnames if x.endswith("concepts")]:
    d = pd.read_excel(NEW, sheet_name=s); bad=0
    for col in ["ground_truth","predicted"]:
        for v in d[col]:
            try: json.loads(v)
            except Exception: bad+=1
    chk(bad==0, f"[{s}] all concept cells parse as JSON")

print("\nROW ALIGNMENT ACROSS SHEETS")
base = dn[["variable","prompt_version"]]
for s in [x for x in wn.sheetnames if x.endswith("concepts")]:
    d = pd.read_excel(NEW, sheet_name=s)
    chk(d[["variable","prompt_version"]].equals(base), f"[{s}] row order identical to 'LLM outputs'")

print("\nKEYS REQUIRED BY CONSUMING SCRIPTS")
need = ["comment","definition","hasProperty","hasObjectOfInterest","hasMatrix","hasConstraint",
        "hasStatisticalModifier","hasContextObject","label"]
for col in ["ground_truth_json","predicted_json"]:
    missing = {k for v in dn[col] for k in need if k not in json.loads(v)}
    chk(not missing, f"[{col}] every row carries all keys scripts touch" + ("" if not missing else f" -- missing {missing}"))

print("\n" + "="*62)
print(("ALL CHECKS PASSED" if not fails else f"{len(fails)} FAILURES") )
for f in fails: print("  -", f)
sys.exit(1 if fails else 0)
