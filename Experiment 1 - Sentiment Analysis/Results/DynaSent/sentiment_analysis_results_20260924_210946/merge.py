import pandas as pd

# Load the CSV files
table_sst = pd.read_csv("table_dynasent.csv")
overall_accuracy_summary = pd.read_csv("overall_accuracy_summary.csv")

# Merge accuracy into table_sst based on the Model column
table_sst = table_sst.merge(
    overall_accuracy_summary[["Model", "Accuracy"]],
    on="Model",
    how="left"
)

# Save the updated table
table_sst.to_csv("table_dynasent_with_accuracy.csv", index=False)

print(table_sst)