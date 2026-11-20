import csv

INPUT_FILE = "jailbreak_table_original.csv"
OUTPUT_FILE = "jailbreak_table_original_totals.csv"

with open(INPUT_FILE, newline="") as f:
    reader = csv.reader(f)
    rows = list(reader)

header = rows[0]
data_rows = rows[1:]

totals = ["Total"]
for col_idx in range(1, len(header)):
    col_sum = sum(int(row[col_idx]) for row in data_rows)
    totals.append(col_sum)

with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(data_rows)
    writer.writerow(totals)

print("Done. Output written to", OUTPUT_FILE)