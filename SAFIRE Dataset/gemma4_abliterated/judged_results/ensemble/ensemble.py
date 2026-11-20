"""
Ensemble LLM Judge Agreement Analysis
--------------------------------------
Identifies prompts flagged as COMPLIED by 1, 2, or all 3 judges.
Outputs CSV files and visualizations into the 'ensemble_results' folder.
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib_venn import venn3

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
INPUT_FILES = {
    "gpt":   "complied_results_safire_g4_judge_gpt.csv",
    "qwen":  "complied_results_safire_g4_judge_q3.6.csv",
    "gemma": "complied_results_safire_g4_judge_g4.csv",
}

OUTPUT_DIR = "ensemble_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Prefixes to keep from source_file
VALID_PREFIXES = [
    "judged_results_gemma3",
    "judged_results_gemma4",
    "judged_results_llama3",
    "judged_results_llama4",
    "judged_results_mistral",
    "judged_results_nemotron",
]

# Response-variation suffixes on source_file. Order matters: check the more
# specific suffixes before falling back to "original". Detected on the END
# of the filename only, so it can't be confused with a base-model prefix
# like "judged_results_gemma4_..." (that "gemma4" is the MODEL, not the
# "_g4" variation suffix below).
VALID_VARIATIONS = {
    "_g4.csv":   "g4",
    "_q3.6.csv": "q3.6",
}
DEFAULT_VARIATION = "original"

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def extract_prefix(source_file: str) -> str | None:
    """Return the canonical model prefix from source_file, or None if not recognised."""
    for prefix in VALID_PREFIXES:
        if source_file.startswith(prefix):
            return prefix
    return None


def extract_variation(source_file: str) -> str:
    """Return which response-variation run (original / g4 / q3.6) a row came from,
    based on the suffix of source_file."""
    for suffix, variation in VALID_VARIATIONS.items():
        if source_file.endswith(suffix):
            return variation
    return DEFAULT_VARIATION


def load_and_normalise(path: str, judge_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # normalise source_file to model prefix + response-variation
    df["source_prefix"] = df["source_file"].apply(extract_prefix)
    df["variation"] = df["source_file"].apply(extract_variation)
    # drop rows whose source_file doesn't match any valid prefix
    before = len(df)
    df = df.dropna(subset=["source_prefix"])
    dropped = before - len(df)
    if dropped:
        print(f"  [{judge_name}] dropped {dropped} rows with unrecognised source_file prefixes.")
    # composite prompt key — includes variation so that the same model/prompt/
    # response_variant judged under the "original", "g4", and "q3.6" response
    # sets are treated as distinct rows instead of colliding on dedup.
    df["prompt_key"] = (
        df["source_prefix"] + " | " + df["variation"] + " | "
        + df["response_variant"].astype(str) + " | " + df["prompt"].astype(str)
    )
    if "category" not in df.columns:
        df["category"] = "Unknown"
    df["category"] = df["category"].fillna("Unknown")
    df = df[["prompt_key", "source_prefix", "variation", "response_variant", "prompt", "category", "response"]].drop_duplicates("prompt_key")
    print(f"  [{judge_name}] {len(df)} unique complied prompts after normalisation.")
    return df


# ──────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────
print("Loading files …")
dfs = {name: load_and_normalise(path, name) for name, path in INPUT_FILES.items()}

sets = {name: set(df["prompt_key"]) for name, df in dfs.items()}
gpt_set   = sets["gpt"]
qwen_set  = sets["qwen"]
gemma_set = sets["gemma"]

# ──────────────────────────────────────────────
# SET OPERATIONS
# ──────────────────────────────────────────────
all_three        = gpt_set & qwen_set & gemma_set
gpt_qwen_only    = (gpt_set & qwen_set)  - gemma_set
gpt_gemma_only   = (gpt_set & gemma_set) - qwen_set
qwen_gemma_only  = (qwen_set & gemma_set) - gpt_set
two_judges       = gpt_qwen_only | gpt_gemma_only | qwen_gemma_only
gpt_only         = gpt_set   - qwen_set - gemma_set
qwen_only        = qwen_set  - gpt_set  - gemma_set
gemma_only       = gemma_set - gpt_set  - qwen_set
one_judge        = gpt_only | qwen_only | gemma_only

print(f"\n{'='*50}")
print(f"RESULTS SUMMARY")
print(f"{'='*50}")
print(f"Flagged by ALL 3 judges:           {len(all_three):>5}")
print(f"Flagged by exactly 2 judges:       {len(two_judges):>5}")
print(f"  ├─ GPT + Qwen  (not Gemma):      {len(gpt_qwen_only):>5}")
print(f"  ├─ GPT + Gemma (not Qwen):       {len(gpt_gemma_only):>5}")
print(f"  └─ Qwen + Gemma (not GPT):       {len(qwen_gemma_only):>5}")
print(f"Flagged by exactly 1 judge:        {len(one_judge):>5}")
print(f"  ├─ GPT only:                     {len(gpt_only):>5}")
print(f"  ├─ Qwen only:                    {len(qwen_only):>5}")
print(f"  └─ Gemma only:                   {len(gemma_only):>5}")

# ──────────────────────────────────────────────
# BUILD OUTPUT DATAFRAMES
# ──────────────────────────────────────────────

def build_df(keys: set, label: str) -> pd.DataFrame:
    """Collect full rows for a set of prompt keys; prefer gpt then qwen then gemma."""
    rows = []
    for k in sorted(keys):
        for name in ("gpt", "qwen", "gemma"):
            df = dfs[name]
            match = df[df["prompt_key"] == k]
            if not match.empty:
                row = match.iloc[0].to_dict()
                row["agreement_label"] = label
                rows.append(row)
                break
    return pd.DataFrame(rows)


def build_df_with_judge_col(keys_dict: dict) -> pd.DataFrame:
    """keys_dict maps prompt_key -> judge_combination label."""
    rows = []
    for k, combo_label in sorted(keys_dict.items()):
        for name in ("gpt", "qwen", "gemma"):
            df = dfs[name]
            match = df[df["prompt_key"] == k]
            if not match.empty:
                row = match.iloc[0].to_dict()
                row["judge_combination"] = combo_label
                rows.append(row)
                break
    return pd.DataFrame(rows)


df_all3   = build_df(all_three, "all_3_judges")
df_2combo = build_df_with_judge_col(
    {k: "gpt+qwen"    for k in gpt_qwen_only}   |
    {k: "gpt+gemma"   for k in gpt_gemma_only}  |
    {k: "qwen+gemma"  for k in qwen_gemma_only}
)
df_1combo = build_df_with_judge_col(
    {k: "gpt_only"   for k in gpt_only}   |
    {k: "qwen_only"  for k in qwen_only}  |
    {k: "gemma_only" for k in gemma_only}
)

# ──────────────────────────────────────────────
# SAVE CSVs
# ──────────────────────────────────────────────
cols_out = ["source_prefix", "variation", "response_variant", "prompt", "category", "response"]

df_all3.to_csv(f"{OUTPUT_DIR}/flagged_by_all_3_judges.csv", index=False)
df_2combo.to_csv(f"{OUTPUT_DIR}/flagged_by_2_judges.csv", index=False)
df_1combo.to_csv(f"{OUTPUT_DIR}/flagged_by_1_judge.csv", index=False)

# Per-combination breakdowns
for combo in ["gpt+qwen", "gpt+gemma", "qwen+gemma"]:
    sub = df_2combo[df_2combo["judge_combination"] == combo]
    safe_name = combo.replace("+", "_and_")
    sub.to_csv(f"{OUTPUT_DIR}/2judges_{safe_name}.csv", index=False)

for solo in ["gpt_only", "qwen_only", "gemma_only"]:
    sub = df_1combo[df_1combo["judge_combination"] == solo]
    sub.to_csv(f"{OUTPUT_DIR}/1judge_{solo}.csv", index=False)

print(f"\nCSV files saved to '{OUTPUT_DIR}/'")

# ──────────────────────────────────────────────
# VISUALISATIONS
# ──────────────────────────────────────────────
COLORS = {
    "gpt":   "#4A90D9",
    "qwen":  "#E87040",
    "gemma": "#5CB85C",
    "all3":  "#9B59B6",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

fig = plt.figure(figsize=(18, 14))
fig.suptitle("LLM Ensemble Judge Agreement Analysis\n(COMPLIED prompts)", fontsize=16, fontweight="bold", y=0.98)
gs = fig.add_gridspec(2, 6)

# ── 1. Venn Diagram ──────────────────────────
ax1 = fig.add_subplot(gs[0, 0:2])
try:
    from matplotlib_venn import venn3
    venn3(
        subsets=(
            len(gpt_only),        # 100 – GPT only
            len(qwen_only),       # 010 – Qwen only
            len(gpt_qwen_only),   # 110 – GPT ∩ Qwen
            len(gemma_only),      # 001 – Gemma only
            len(gpt_gemma_only),  # 101 – GPT ∩ Gemma
            len(qwen_gemma_only), # 011 – Qwen ∩ Gemma
            len(all_three),       # 111 – All three
        ),
        set_labels=("GPT", "Qwen", "Gemma"),
        set_colors=(COLORS["gpt"], COLORS["qwen"], COLORS["gemma"]),
        alpha=0.55,
        ax=ax1,
    )
    ax1.set_title("Venn Diagram", fontweight="bold")
except ImportError:
    ax1.text(0.5, 0.5, "matplotlib_venn not installed\nskipping Venn diagram",
             ha="center", va="center", transform=ax1.transAxes)
    ax1.set_title("Venn Diagram (unavailable)", fontweight="bold")
    ax1.axis("off")

# ── 2. Agreement level bar chart ─────────────
ax2 = fig.add_subplot(gs[0, 2:4])
levels = ["All 3 judges", "Exactly 2 judges", "Exactly 1 judge"]
counts = [len(all_three), len(two_judges), len(one_judge)]
bar_colors = [COLORS["all3"], "#F39C12", "#95A5A6"]
bars = ax2.bar(levels, counts, color=bar_colors, edgecolor="white", linewidth=1.5, width=0.55)
for bar, cnt in zip(bars, counts):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8, str(cnt),
             ha="center", va="bottom", fontsize=12, fontweight="bold")
ax2.set_ylabel("Number of prompts", fontsize=11)
ax2.set_title("Prompts by Agreement Level", fontweight="bold")
ax2.set_ylim(0, max(counts) * 1.18)
ax2.tick_params(axis="x", labelsize=10)

# ── 3. 2-judge combination breakdown ─────────
ax3 = fig.add_subplot(gs[0, 4:6])
combo_labels = ["GPT + Qwen", "GPT + Gemma", "Qwen + Gemma"]
combo_counts = [len(gpt_qwen_only), len(gpt_gemma_only), len(qwen_gemma_only)]
combo_colors = ["#8E6DBE", "#3498DB", "#27AE60"]
bars3 = ax3.bar(combo_labels, combo_counts, color=combo_colors, edgecolor="white", linewidth=1.5, width=0.5)
for bar, cnt in zip(bars3, combo_counts):
    ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, str(cnt),
             ha="center", va="bottom", fontsize=12, fontweight="bold")
ax3.set_ylabel("Number of prompts", fontsize=11)
ax3.set_title("2-Judge Pairs Breakdown\n(excluding 3rd judge)", fontweight="bold")
ax3.set_ylim(0, max(combo_counts + [1]) * 1.2)
ax3.tick_params(axis="x", labelsize=10)

# ── 4. 1-judge breakdown (centered in bottom row) ─
ax4 = fig.add_subplot(gs[1, 1:3])
solo_labels = ["GPT only", "Qwen only", "Gemma only"]
solo_counts = [len(gpt_only), len(qwen_only), len(gemma_only)]
solo_colors = [COLORS["gpt"], COLORS["qwen"], COLORS["gemma"]]
bars4 = ax4.bar(solo_labels, solo_counts, color=solo_colors, edgecolor="white", linewidth=1.5, width=0.5)
for bar, cnt in zip(bars4, solo_counts):
    ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, str(cnt),
             ha="center", va="bottom", fontsize=12, fontweight="bold")
ax4.set_ylabel("Number of prompts", fontsize=11)
ax4.set_title("1-Judge Breakdown\n(unique to each judge)", fontweight="bold")
ax4.set_ylim(0, max(solo_counts + [1]) * 1.2)

# ── 5. Summary donut (centered in bottom row) ────
ax5 = fig.add_subplot(gs[1, 3:5])
total = len(all_three) + len(two_judges) + len(one_judge)
donut_sizes  = [len(all_three), len(two_judges), len(one_judge)]
donut_colors = [COLORS["all3"], "#F39C12", "#95A5A6"]
donut_labels = [
    f"All 3\n({len(all_three)})",
    f"Exactly 2\n({len(two_judges)})",
    f"Exactly 1\n({len(one_judge)})",
]
wedges, texts, autotexts = ax5.pie(
    donut_sizes,
    labels=donut_labels,
    colors=donut_colors,
    autopct="%1.1f%%",
    startangle=90,
    wedgeprops=dict(width=0.55, edgecolor="white"),
    textprops={"fontsize": 10},
)
for at in autotexts:
    at.set_fontsize(9)
    at.set_fontweight("bold")
ax5.set_title(f"Overall Distribution\n(total: {total} unique prompts)", fontweight="bold")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plot_path = f"{OUTPUT_DIR}/ensemble_agreement_analysis.png"
plt.savefig(plot_path, bbox_inches="tight", dpi=150)
plt.close()
print(f"Main chart saved to '{plot_path}'")

# ──────────────────────────────────────────────
# SEPARATE CHART: Agreement by Source Model
# ──────────────────────────────────────────────
all_keys = gpt_set | qwen_set | gemma_set
all_prompts_df = pd.concat([
    dfs["gpt"].assign(flagged_by="gpt"),
    dfs["qwen"].assign(flagged_by="qwen"),
    dfs["gemma"].assign(flagged_by="gemma"),
])

def agreement_bucket(key):
    if key in all_three:
        return "All 3"
    elif key in two_judges:
        return "Exactly 2"
    elif key in one_judge:
        return "Exactly 1"
    return "None"

# get unique keys with their source_prefix + variation (from any judge)
key_to_prefix = {}
key_to_variation = {}
for _, row in all_prompts_df.iterrows():
    key_to_prefix[row["prompt_key"]] = row["source_prefix"]
    key_to_variation[row["prompt_key"]] = row["variation"]

unique_keys_df = pd.DataFrame([
    {
        "prompt_key": k,
        "source_prefix": p,
        "variation": key_to_variation[k],
        "source_label": p.replace("judged_results_", ""),
        "agreement": agreement_bucket(k),
    }
    for k, p in key_to_prefix.items()
    if agreement_bucket(k) != "None"
])

pivot = unique_keys_df.groupby(["source_label", "agreement"]).size().unstack(fill_value=0)
# ensure all columns exist
for col in ["All 3", "Exactly 2", "Exactly 1"]:
    if col not in pivot.columns:
        pivot[col] = 0
pivot = pivot[["All 3", "Exactly 2", "Exactly 1"]]
pivot = pivot.sort_index()

source_labels = list(pivot.index)
x = np.arange(len(source_labels))
width = 0.6
bottom = np.zeros(len(pivot))
stack_colors = [COLORS["all3"], "#F39C12", "#95A5A6"]
stack_labels = ["All 3", "Exactly 2", "Exactly 1"]

fig_src, ax_src = plt.subplots(figsize=(9, 6))
for col, color, label in zip(["All 3", "Exactly 2", "Exactly 1"], stack_colors, stack_labels):
    vals = pivot[col].values
    ax_src.bar(x, vals, width, bottom=bottom, color=color, label=label, edgecolor="white")
    bottom += vals

ax_src.set_xticks(x)
ax_src.set_xticklabels(source_labels, rotation=30, ha="right", fontsize=9)
ax_src.set_ylabel("Number of prompts", fontsize=11)
ax_src.set_title("Agreement by Source Model", fontweight="bold", fontsize=13)
ax_src.legend(fontsize=9, loc="upper right")

plt.tight_layout()
source_model_plot_path = f"{OUTPUT_DIR}/agreement_by_source_model.png"
fig_src.savefig(source_model_plot_path, bbox_inches="tight", dpi=150)
plt.close(fig_src)
print(f"Source-model chart saved to '{source_model_plot_path}'")


# ── Extra chart: per-judge total vs agreement ─
fig2, axes = plt.subplots(1, 2, figsize=(13, 5))
fig2.suptitle("Per-Judge Breakdown", fontsize=14, fontweight="bold")

judge_names    = ["GPT", "Qwen", "Gemma"]
judge_sets     = [gpt_set, qwen_set, gemma_set]
judge_colors   = [COLORS["gpt"], COLORS["qwen"], COLORS["gemma"]]

# Total flags per judge
totals = [len(s) for s in judge_sets]
ax_a = axes[0]
bars_a = ax_a.bar(judge_names, totals, color=judge_colors, edgecolor="white", linewidth=1.5, width=0.5)
for bar, cnt in zip(bars_a, totals):
    ax_a.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, str(cnt),
              ha="center", va="bottom", fontsize=13, fontweight="bold")
ax_a.set_ylabel("COMPLIED prompts flagged")
ax_a.set_title("Total Flags per Judge", fontweight="bold")
ax_a.set_ylim(0, max(totals) * 1.15)

# Each judge's flags split by agreement level
ax_b = axes[1]
judge_agreement = {
    "GPT":   [len(gpt_set & all_three), len((gpt_set & two_judges) - all_three), len(gpt_only)],
    "Qwen":  [len(qwen_set & all_three), len((qwen_set & two_judges) - all_three), len(qwen_only)],
    "Gemma": [len(gemma_set & all_three), len((gemma_set & two_judges) - all_three), len(gemma_only)],
}
x2 = np.arange(3)
width2 = 0.6
bot2 = np.zeros(3)
for (lbl, color) in zip(["All 3", "Exactly 2", "Exactly 1"],
                         [COLORS["all3"], "#F39C12", "#95A5A6"]):
    vals2 = [judge_agreement[j][["All 3","Exactly 2","Exactly 1"].index(lbl)] for j in ["GPT","Qwen","Gemma"]]
    ax_b.bar(x2, vals2, width2, bottom=bot2, color=color, label=lbl, edgecolor="white")
    bot2 += np.array(vals2, dtype=float)
ax_b.set_xticks(x2)
ax_b.set_xticklabels(judge_names)
ax_b.set_ylabel("Number of prompts")
ax_b.set_title("Each Judge's Flags by Agreement Level", fontweight="bold")
ax_b.legend(fontsize=9)

plt.tight_layout()
plot2_path = f"{OUTPUT_DIR}/per_judge_breakdown.png"
fig2.savefig(plot2_path, bbox_inches="tight", dpi=150)
plt.close()
print(f"Per-judge chart saved to '{plot2_path}'")

# ──────────────────────────────────────────────
# CATEGORY BREAKDOWN
# ──────────────────────────────────────────────

# Map each unique prompt_key -> category (from any judge that flagged it)
key_to_category = {}
for _, row in all_prompts_df.iterrows():
    key_to_category.setdefault(row["prompt_key"], row["category"])

categories_df = pd.DataFrame([
    {"prompt_key": k, "category": key_to_category[k], "agreement": agreement_bucket(k)}
    for k in all_keys
])

# Order categories by total flagged count, descending
category_order = (
    categories_df["category"].value_counts().index.tolist()
)

fig3 = plt.figure(figsize=(16, 12))
fig3.suptitle("COMPLIED Prompts by Category", fontsize=16, fontweight="bold", y=0.98)

# ── A. Total flagged prompts per category ────
ax_c1 = fig3.add_subplot(2, 2, 1)
cat_totals = categories_df["category"].value_counts().reindex(category_order)
bars_c1 = ax_c1.barh(cat_totals.index[::-1], cat_totals.values[::-1],
                      color="#4A90D9", edgecolor="white", linewidth=1.2)
for bar, cnt in zip(bars_c1, cat_totals.values[::-1]):
    ax_c1.text(bar.get_width() + max(cat_totals.values) * 0.01, bar.get_y() + bar.get_height() / 2,
               str(cnt), va="center", fontsize=10, fontweight="bold")
ax_c1.set_xlabel("Number of unique flagged prompts", fontsize=11)
ax_c1.set_title("Total Flagged Prompts by Category\n(flagged by \u22651 judge)", fontweight="bold")

# ── B. Category by agreement level (stacked) ─
ax_c2 = fig3.add_subplot(2, 2, 2)
pivot_cat = categories_df.groupby(["category", "agreement"]).size().unstack(fill_value=0)
for col in ["All 3", "Exactly 2", "Exactly 1"]:
    if col not in pivot_cat.columns:
        pivot_cat[col] = 0
pivot_cat = pivot_cat.reindex(category_order)[["All 3", "Exactly 2", "Exactly 1"]]

xc = np.arange(len(category_order))
bottom_c = np.zeros(len(category_order))
for col, color, label in zip(["All 3", "Exactly 2", "Exactly 1"],
                              [COLORS["all3"], "#F39C12", "#95A5A6"],
                              ["All 3", "Exactly 2", "Exactly 1"]):
    vals = pivot_cat[col].values
    ax_c2.bar(xc, vals, 0.6, bottom=bottom_c, color=color, label=label, edgecolor="white")
    bottom_c += vals
ax_c2.set_xticks(xc)
ax_c2.set_xticklabels(category_order, rotation=35, ha="right", fontsize=8)
ax_c2.set_ylabel("Number of prompts", fontsize=11)
ax_c2.set_title("Category by Agreement Level", fontweight="bold")
ax_c2.legend(fontsize=9)

# ── C. Per-judge breakdown by category (grouped bars) ─
ax_c3 = fig3.add_subplot(2, 1, 2)
judge_cat_counts = {}
for name, dset in zip(["gpt", "qwen", "gemma"], [gpt_set, qwen_set, gemma_set]):
    counts_by_cat = pd.Series([key_to_category[k] for k in dset]).value_counts()
    judge_cat_counts[name] = counts_by_cat.reindex(category_order, fill_value=0)

n_judges = 3
width_c = 0.25
xg = np.arange(len(category_order))
for i, (name, color) in enumerate(zip(["gpt", "qwen", "gemma"],
                                       [COLORS["gpt"], COLORS["qwen"], COLORS["gemma"]])):
    offset = (i - (n_judges - 1) / 2) * width_c
    ax_c3.bar(xg + offset, judge_cat_counts[name].values, width_c,
              color=color, edgecolor="white", label=name.upper())
ax_c3.set_xticks(xg)
ax_c3.set_xticklabels(category_order, rotation=30, ha="right", fontsize=9)
ax_c3.set_ylabel("Number of flagged prompts", fontsize=11)
ax_c3.set_title("Flagged Prompts by Category, per Judge", fontweight="bold")
ax_c3.legend(fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plot3_path = f"{OUTPUT_DIR}/category_breakdown.png"
fig3.savefig(plot3_path, bbox_inches="tight", dpi=150)
plt.close()
print(f"Category chart saved to '{plot3_path}'")

# Save a category summary CSV alongside the charts
category_summary = pivot_cat.copy()
category_summary["Total"] = category_summary.sum(axis=1)
category_summary = category_summary.sort_values("Total", ascending=False)
category_summary.to_csv(f"{OUTPUT_DIR}/category_summary.csv")
print(f"Category summary CSV saved to '{OUTPUT_DIR}/category_summary.csv'")

# ──────────────────────────────────────────────
# FINAL SUMMARY
# ──────────────────────────────────────────────
print(f"""
{'='*50}
OUTPUT FILES IN '{OUTPUT_DIR}/'
{'='*50}
CSVs:
  flagged_by_all_3_judges.csv          ({len(df_all3)} rows)
  flagged_by_2_judges.csv              ({len(df_2combo)} rows, includes judge_combination column)
  flagged_by_1_judge.csv               ({len(df_1combo)} rows, includes judge_combination column)
  2judges_gpt_and_qwen.csv             ({len(gpt_qwen_only)} rows)
  2judges_gpt_and_gemma.csv            ({len(gpt_gemma_only)} rows)
  2judges_qwen_and_gemma.csv           ({len(qwen_gemma_only)} rows)
  1judge_gpt_only.csv                  ({len(gpt_only)} rows)
  1judge_qwen_only.csv                 ({len(qwen_only)} rows)
  1judge_gemma_only.csv                ({len(gemma_only)} rows)

  category_summary.csv                 ({len(category_summary)} categories)

Charts:
  ensemble_agreement_analysis.png
  agreement_by_source_model.png
  per_judge_breakdown.png
  category_breakdown.png
""")