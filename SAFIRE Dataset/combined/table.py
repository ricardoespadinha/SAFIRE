import pandas as pd
from pathlib import Path

# Folder containing the three CSV files
folder = Path(__file__).parent

# Input files
files = [
    folder / "jailbreak_table_g4.csv",
    folder / "jailbreak_table_original.csv",
    folder / "jailbreak_table_q3.6.csv",
]

# Read the CSV files
tables = [pd.read_csv(file) for file in files]

# Check that all files have the same columns
columns = tables[0].columns

for i, table in enumerate(tables[1:], start=2):
    if not table.columns.equals(columns):
        raise ValueError(
            f"File {files[i-1].name} has different columns "
            f"from the first file."
        )

# Check that all files contain the same LLMs
for i, table in enumerate(tables):
    if set(table["LLM"]) != set(tables[0]["LLM"]):
        raise ValueError(
            f"File {files[i].name} contains different LLM names."
        )

# Merge the three tables using LLM as the key
result = tables[0].copy()

for table in tables[1:]:
    result = result.merge(
        table,
        on="LLM",
        suffixes=("", "_other")
    )

    # Sum each house column
    for column in columns:
        if column != "LLM":
            result[column] += result.pop(f"{column}_other")

# Save the summed table
output_file = folder / "jailbreak_table_sum.csv"
result.to_csv(output_file, index=False)

print("Summed table:")
print(result.to_string(index=False))

print(f"\nSaved to: {output_file}")