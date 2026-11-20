import pandas as pd

# Load the CSV
df = pd.read_csv("my_ratings.csv")

# Count prompts for each rating within each voting group
table = pd.crosstab(
    df["n_votes"],
    df["rating"]
)

# Make sure all voting groups 0, 1, 2, 3 are present
table = table.reindex(index=[0, 1, 2, 3], fill_value=0)

# Make sure all rating categories are present
ratings = [
    "Non-Harmful Compliance",
    "Refusal",
    "Partial Harm",
    "Full Harm"
]
table = table.reindex(columns=ratings, fill_value=0)

# Add total number of prompts for each voting group
table["Total"] = table.sum(axis=1)

# Rename the index
table.index.name = "Voting Group"

print(table)

# Save the table to a CSV file
table.to_csv("rating_counts_by_voting_group.csv")