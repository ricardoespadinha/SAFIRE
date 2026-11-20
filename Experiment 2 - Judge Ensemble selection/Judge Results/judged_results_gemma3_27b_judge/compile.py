"""
extract_complied.py

Scans all CSV files with 'judged_results' in their name in a given folder,
identifies rows where any judge_response_* column contains 'COMPLIED',
and writes the prompt + corresponding response(s) to an output CSV.

Usage:
    python extract_complied.py --input_dir /path/to/folder --output /path/to/output.csv

Defaults:
    --input_dir  : current directory (.)
    --output     : complied_results.csv
"""

import os
import glob
import argparse
import pandas as pd


def extract_complied(input_dir: str, output_path: str) -> None:
    # Find all CSVs with 'judged_results' in the filename
    pattern = os.path.join(input_dir, "*judged_results*.csv")
    files = glob.glob(pattern)

    if not files:
        print(f"No files matching '*judged_results*.csv' found in: {input_dir}")
        return

    print(f"Found {len(files)} file(s): {[os.path.basename(f) for f in files]}")

    records = []

    for filepath in files:
        filename = os.path.basename(filepath)
        print(f"\nProcessing: {filename}")

        df = pd.read_csv(filepath, low_memory=False)

        # Identify all judge_response_* columns and their paired response_* columns
        judge_cols = [c for c in df.columns if c.startswith("judge_response_")]

        for judge_col in judge_cols:
            # Derive the matching response column: judge_response_X -> response_X
            suffix = judge_col[len("judge_response_"):]  # e.g. "0", "-1", "1", ...
            response_col = f"response_{suffix}"

            if response_col not in df.columns:
                print(f"  Warning: '{response_col}' not found for '{judge_col}', skipping.")
                continue

            # Find rows where this judge column contains 'COMPLIED' (case-insensitive)
            mask = df[judge_col].astype(str).str.contains("COMPLIED", case=False, na=False)
            matched = df[mask][["prompt", response_col, judge_col]].copy()

            if matched.empty:
                continue

            matched = matched.rename(columns={
                response_col: "response",
                judge_col: "judge_response",
            })
            matched["source_file"] = filename
            matched["response_variant"] = f"response_{suffix}"

            records.append(matched)
            print(f"  '{judge_col}': {len(matched)} COMPLIED row(s) found.")

    if not records:
        print("\nNo COMPLIED responses found across all files.")
        return

    result = pd.concat(records, ignore_index=True)

    # Reorder columns for readability
    result = result[["source_file", "response_variant", "prompt", "response", "judge_response"]]

    result.to_csv(output_path, index=False)
    print(f"\nDone. {len(result)} total row(s) written to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract COMPLIED responses from judged_results CSVs.")
    parser.add_argument(
        "--input_dir",
        default=".",
        help="Folder containing the judged_results CSV files (default: current directory)",
    )
    parser.add_argument(
        "--output",
        default="complied_results.csv",
        help="Path for the output CSV (default: complied_results.csv)",
    )
    args = parser.parse_args()

    extract_complied(input_dir=args.input_dir, output_path=args.output)