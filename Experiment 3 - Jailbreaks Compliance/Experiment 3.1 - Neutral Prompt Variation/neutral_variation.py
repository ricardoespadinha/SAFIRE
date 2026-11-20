#!/usr/bin/env python3
"""
For each combined results CSV, look only at rows where response_variant == 'response_0'.
Each row's source_file encodes:
    judged_results_<model>_judge_<judge>_<variation>.csv
where <variation> is one of: g4, original, q3.6

For every base model, we collect the set of prompts seen under each of the
3 variation sets, then compute what fraction of that model's distinct prompts
are present in ALL 3 variation sets. We report this per-model and as an
overall (all-models-pooled) percentage for each file.
"""

import re
import pandas as pd
from pathlib import Path

FILES = [
    "complied_results_safire_combined_judge_gpt_with_original.csv",
    "complied_results_safire_combined_judge_q3.6_with_original.csv",
    "complied_results_safire_combined_judge_g4_with_original.csv",
]

# Directory where per-file result CSVs (and the combined summary) are written
OUTPUT_DIR = Path("")

SOURCE_FILE_RE = re.compile(r"^judged_results_([^_]+)_judge_.+_(g4|original|q3\.6)\.csv$")
VARIATIONS = ["g4", "original", "q3.6"]


def parse_source_file(name: str):
    """Return (model, variation) parsed from a source_file value, or (None, None)."""
    m = SOURCE_FILE_RE.match(name)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def analyze_file(path: Path):
    df = pd.read_csv(path)

    # Only response_0 rows
    df = df[df["response_variant"] == "response_0"].copy()

    parsed = df["source_file"].apply(parse_source_file)
    df["model"] = parsed.apply(lambda t: t[0])
    df["variation"] = parsed.apply(lambda t: t[1])

    unparsed = df[df["model"].isna()]
    if not unparsed.empty:
        unmatched_names = sorted(unparsed["source_file"].unique())
        print(f"  WARNING: {len(unmatched_names)} source_file value(s) did not match the "
              f"expected naming pattern and were excluded: {unmatched_names}")
        df = df.dropna(subset=["model"])

    results = {}
    overall_in_all = 0
    overall_total = 0

    for model, group in df.groupby("model"):
        # prompts seen per variation for this model
        sets_by_variation = {
            v: set(group.loc[group["variation"] == v, "prompt"])
            for v in VARIATIONS
        }

        all_prompts = set().union(*sets_by_variation.values())
        in_all_three = set.intersection(*sets_by_variation.values()) if all(sets_by_variation.values()) else set()
        # (if any variation set is empty, intersection is empty too; guard just for clarity)
        if any(len(s) == 0 for s in sets_by_variation.values()):
            in_all_three = set()
        else:
            in_all_three = set.intersection(*sets_by_variation.values())

        total = len(all_prompts)
        in_all = len(in_all_three)
        pct = (in_all / total * 100) if total else 0.0

        results[model] = {
            "total_distinct_prompts": total,
            "in_all_3_variations": in_all,
            "percentage": pct,
            "counts_per_variation": {v: len(s) for v, s in sets_by_variation.items()},
        }

        overall_in_all += in_all
        overall_total += total

    overall_pct = (overall_in_all / overall_total * 100) if overall_total else 0.0

    return results, overall_in_all, overall_total, overall_pct


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    combined_rows = []  # rows across all input files, for one combined summary CSV

    for fname in FILES:
        path = Path(fname)
        if not path.exists():
            # also try the uploads dir if run from elsewhere
            alt = Path("/mnt/user-data/uploads") / fname
            if alt.exists():
                path = alt
            else:
                print(f"File not found: {fname}")
                continue

        print("=" * 90)
        print(f"File: {fname}")
        print("=" * 90)

        results, overall_in_all, overall_total, overall_pct = analyze_file(path)

        rows = []
        for model, info in sorted(results.items()):
            counts = info["counts_per_variation"]
            print(
                f"  Model: {model:<12} "
                f"g4={counts['g4']:<5} original={counts['original']:<5} q3.6={counts['q3.6']:<5}  "
                f"-> in all 3: {info['in_all_3_variations']}/{info['total_distinct_prompts']} "
                f"({info['percentage']:.2f}%)"
            )
            row = {
                "source_file": fname,
                "model": model,
                "count_g4": counts["g4"],
                "count_original": counts["original"],
                "count_q3.6": counts["q3.6"],
                "distinct_prompts_total": info["total_distinct_prompts"],
                "prompts_in_all_3": info["in_all_3_variations"],
                "percentage_in_all_3": round(info["percentage"], 2),
            }
            rows.append(row)
            combined_rows.append(row)

        # Overall row for this file
        overall_row = {
            "source_file": fname,
            "model": "OVERALL",
            "count_g4": "",
            "count_original": "",
            "count_q3.6": "",
            "distinct_prompts_total": overall_total,
            "prompts_in_all_3": overall_in_all,
            "percentage_in_all_3": round(overall_pct, 2),
        }
        rows.append(overall_row)
        combined_rows.append(overall_row)

        print("-" * 90)
        print(f"  OVERALL (all models pooled): {overall_in_all}/{overall_total} prompts "
              f"in all 3 variation sets ({overall_pct:.2f}%)")
        print()

        # Write one CSV per input file
        out_name = f"{path.stem}_variation_coverage.csv"
        out_path = OUTPUT_DIR / out_name
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"  -> wrote {out_path}")
        print()

    # Combined summary CSV across all input files
    if combined_rows:
        combined_path = OUTPUT_DIR / "all_files_variation_coverage_summary.csv"
        pd.DataFrame(combined_rows).to_csv(combined_path, index=False)
        print(f"Combined summary written to {combined_path}")


if __name__ == "__main__":
    main()