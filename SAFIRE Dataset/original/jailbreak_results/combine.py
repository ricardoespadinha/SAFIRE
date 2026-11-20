#!/usr/bin/env python3
"""
Combine the per-victim response files into one file per variation set.

Input : jailbreak_results_<victim>_<set>.csv   (6 victims x 3 sets = 18 files)
        columns: category, prompt, reference_response_refuse, response_0, response_-1 ... response_5
Output: jailbreak_results_combined_<set>.csv   (3 files, 6 x 578 rows each)
        same columns plus a leading `victim` column.

Note on jailbreak_results_gemma3_original.csv: from row ~268 onward every row has two extra trailing
empty fields (",,"), so pandas.read_csv rejects it. This script reads with the csv module instead,
drops extra trailing fields ONLY if they are empty, and raises if anything non-empty would be lost.

Usage:
  python combine_jailbreak_results.py --data-dir /path/to/csvs --out-dir combined
"""
import argparse
import csv
import io
from pathlib import Path

import pandas as pd

VICTIMS = ["gemma3", "gemma4", "llama3", "llama4", "mistral", "nemotron"]
SETS = ["original"]  # file-name suffixes (q3_6 == "q3.6" in the judged files)

csv.field_size_limit(10**9)  # some responses are very long


def read_robust(path: Path) -> pd.DataFrame:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    header, body = rows[0], rows[1:]
    width = len(header)
    fixed, n_trimmed = [], 0
    for i, row in enumerate(body, start=2):
        if len(row) > width:
            extra = row[width:]
            if any(c.strip() for c in extra):
                raise ValueError(f"{path.name} line/row {i}: {len(extra)} extra NON-empty fields; refusing to drop")
            row, n_trimmed = row[:width], n_trimmed + 1
        elif len(row) < width:
            raise ValueError(f"{path.name} row {i}: only {len(row)} of {width} fields")
        fixed.append(row)
    if n_trimmed:
        print(f"  [{path.name}] trimmed empty trailing fields on {n_trimmed} rows")
    return pd.DataFrame(fixed, columns=header)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=".", type=Path)
    ap.add_argument("--out-dir", default=".", type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for s in SETS:
        parts, ref_cols = [], None
        for v in VICTIMS:
            path = args.data_dir / f"jailbreak_results_{v}_{s}.csv"
            df = read_robust(path)
            if ref_cols is None:
                ref_cols = list(df.columns)
            elif list(df.columns) != ref_cols:
                raise ValueError(f"{path.name}: columns differ from the first file")
            if df["prompt"].duplicated().any():
                raise ValueError(f"{path.name}: duplicate base prompts")
            df.insert(0, "victim", v)
            parts.append(df)
        # every victim must contain the same base prompts (same set)
        ref = set(parts[0]["prompt"])
        for p in parts[1:]:
            if set(p["prompt"]) != ref:
                raise ValueError(f"base prompts differ across victims for set {s}")
        combined = pd.concat(parts, ignore_index=True)
        out = args.out_dir / f"jailbreak_results_combined_{s}.csv"
        combined.to_csv(out, index=False)
        print(f"{s}: {len(combined):,} rows ({len(VICTIMS)} victims x {len(combined)//len(VICTIMS)}) -> {out}")


if __name__ == "__main__":
    main()