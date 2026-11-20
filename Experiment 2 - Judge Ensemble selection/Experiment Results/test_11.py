"""
Judge Consensus Analysis
========================
Reads 11 compiled-results CSVs (one per judge), normalises the source_file
column to the shared base name, then finds which (source_file, response_variant,
prompt) triples are agreed upon by exactly N judges.

Output folder: output/
  - consensus_N_of_11.csv  for N in 11..1   (includes response column)
  - charts/               PNG visualisations
"""

import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────

UPLOAD_DIR = Path(".")
OUT_DIR    = Path("output")
CHART_DIR  = OUT_DIR / "charts"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)

# Map file → judge label
FILES = {
    "complied_results_gemma3.csv":            "gemma3",
    "complied_results_gemma3_abliterated.csv": "gemma3_abliterated",
    "complied_results_gemma4.csv":            "gemma4",
    "complied_results_gpt_oss_120b.csv":      "gpt_oss_120b",
    "complied_results_gpt_oss_20b.csv":      "gpt_oss_20b",
    "complied_results_llama3.csv":            "llama3",
    "complied_results_llama4.csv":            "llama4",
    "complied_results_glm4.csv":            "glm4",
    "complied_results_gpt_oss_safeguard_120b.csv":      "gpt_oss_safeguard_120b",
    "complied_results_gpt_oss_safeguard_20b.csv":      "gpt_oss_safeguard_20b",
    "complied_results_qwen3.6.csv":      "qwen3.6",
}

# ── 1. Normalise source_file ───────────────────────────────────────────────

KNOWN_BASES = [
    "judged_results_gemma3",
    "judged_results_gemma4",
    "judged_results_llama3",
    "judged_results_llama4",
    "judged_results_mistral",
    "judged_results_nemotron",
]

def normalise_source(name: str) -> str:
    name = str(name)
    for base in KNOWN_BASES:
        if name.startswith(base):
            return base
    name = re.sub(r"\.csv$", "", name)
    name = re.sub(r"_judge.*$", "", name)
    return name

# ── 2. Load all CSVs ───────────────────────────────────────────────────────

KEY = ["source_file", "response_variant", "prompt"]

print("Loading files …")
frames = {}
# Also keep one copy of the responses for later joining
response_store = []

for fname, judge in FILES.items():
    path = UPLOAD_DIR / fname
    df = pd.read_csv(path)
    df["source_file"] = df["source_file"].apply(normalise_source)

    # Save responses before deduplicating keys (take first response per triple)
    response_store.append(
        df[KEY + ["response"]].drop_duplicates(subset=KEY)
    )

    df = df[KEY].drop_duplicates()
    df["_judge"] = judge
    frames[judge] = df
    print(f"  {judge}: {len(df):,} unique triples")

# Build a single response lookup (first non-null response wins per triple)
responses_df = (
    pd.concat(response_store, ignore_index=True)
    .drop_duplicates(subset=KEY)
)

all_df = pd.concat(frames.values(), ignore_index=True)
JUDGES = list(frames.keys())
N_JUDGES = len(JUDGES)

# ── 3. Count how many judges agree on each triple ─────────────────────────

print("\nCounting judge votes per triple …")

vote_counts = (
    all_df.groupby(KEY)["_judge"]
    .agg(list)
    .reset_index()
    .rename(columns={"_judge": "judges_who_agree"})
)
vote_counts["n_judges"] = vote_counts["judges_who_agree"].apply(len)

# Join the response column right after prompt
vote_counts = vote_counts.merge(responses_df, on=KEY, how="left")

# Reorder columns: source_file, response_variant, prompt, response, judges_who_agree, n_judges
vote_counts = vote_counts[KEY + ["response", "judges_who_agree", "n_judges"]]

# ── 4. Split into consensus buckets (no repetition) ───────────────────────

print("Splitting into consensus buckets …")
consensus = {}
for n in range(N_JUDGES, 0, -1):
    bucket = vote_counts[vote_counts["n_judges"] == n].copy()
    bucket["judges_who_agree"] = bucket["judges_who_agree"].apply(
        lambda lst: ", ".join(sorted(lst))
    )
    consensus[n] = bucket
    print(f"  consensus {n}/11 : {len(bucket):,} triples")

# ── 5. Save consensus CSVs ────────────────────────────────────────────────

print("\nSaving consensus CSVs …")
for n, df in consensus.items():
    out = OUT_DIR / f"consensus_{n}_of_{N_JUDGES}.csv"
    df.to_csv(out, index=False)
    print(f"  → {out.name}")

# ── 6. Judge-level stats ──────────────────────────────────────────────────

judge_stats = {}
for judge in JUDGES:
    jdf = frames[judge]
    merged = jdf.merge(vote_counts[KEY + ["n_judges"]], on=KEY, how="left")
    counts = merged["n_judges"].value_counts().sort_index()
    judge_stats[judge] = counts

stats_df = pd.DataFrame(judge_stats).fillna(0).astype(int)
stats_df.index.name = "consensus_level"
stats_df.to_csv(OUT_DIR / "judge_stats.csv")
print(f"\nJudge stats saved → judge_stats.csv")

sets = {j: set(map(tuple, frames[j][KEY].values)) for j in JUDGES}

# Pairwise agreement matrix
print("Computing pairwise agreement …")
pair_matrix = pd.DataFrame(0, index=JUDGES, columns=JUDGES, dtype=int)
for j in JUDGES:
    pair_matrix.loc[j, j] = len(sets[j])
for j1, j2 in combinations(JUDGES, 2):
    overlap = len(sets[j1] & sets[j2])
    pair_matrix.loc[j1, j2] = overlap
    pair_matrix.loc[j2, j1] = overlap
pair_matrix.to_csv(OUT_DIR / "pairwise_agreement_matrix.csv")

# Jaccard similarity
jaccard = pd.DataFrame(0.0, index=JUDGES, columns=JUDGES)
for j1, j2 in combinations(JUDGES, 2):
    u = len(sets[j1] | sets[j2])
    jac = len(sets[j1] & sets[j2]) / u if u else 0.0
    jaccard.loc[j1, j2] = jac
    jaccard.loc[j2, j1] = jac
for j in JUDGES:
    jaccard.loc[j, j] = 1.0
jaccard.to_csv(OUT_DIR / "jaccard_similarity_matrix.csv")

# ── 7. Charts ─────────────────────────────────────────────────────────────

PALETTE = sns.color_palette("tab10", N_JUDGES)
JUDGE_COLORS = dict(zip(JUDGES, PALETTE))

# 7a. Stacked bar per judge
fig, ax = plt.subplots(figsize=(13, 6))
bottom = np.zeros(N_JUDGES)
x = np.arange(N_JUDGES)
level_colors = plt.cm.RdYlGn(np.linspace(0.15, 0.9, N_JUDGES))
for lvl in range(1, N_JUDGES + 1):
    vals = [stats_df.loc[lvl, j] if lvl in stats_df.index else 0 for j in JUDGES]
    ax.bar(x, vals, bottom=bottom, color=level_colors[lvl - 1],
           label=f"Consensus {lvl}/11", edgecolor="white", linewidth=0.5)
    bottom += np.array(vals)
ax.set_xticks(x)
ax.set_xticklabels(JUDGES, rotation=35, ha="right", fontsize=10)
ax.set_ylabel("Number of triples", fontsize=11)
ax.set_title("Triples flagged per judge — broken down by consensus level",
             fontsize=13, fontweight="bold")
ax.legend(loc="upper right", fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig(CHART_DIR / "01_stacked_bar_judge_consensus.png", dpi=150)
plt.close()

# 7b. Total triples per judge
fig, ax = plt.subplots(figsize=(10, 5))
totals_series = pd.Series({j: len(sets[j]) for j in JUDGES}).sort_values(ascending=False)
colors = [JUDGE_COLORS[j] for j in totals_series.index]
bars = ax.bar(totals_series.index, totals_series.values, color=colors,
              edgecolor="white", linewidth=0.8)
for bar, val in zip(bars, totals_series.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 15,
            str(val), ha="center", va="bottom", fontsize=9)
ax.set_xlabel("Judge", fontsize=11)
ax.set_ylabel("Total unique triples flagged", fontsize=11)
ax.set_title("Total compliance triples flagged per judge", fontsize=13, fontweight="bold")
ax.set_xticklabels(totals_series.index, rotation=35, ha="right")
plt.tight_layout()
plt.savefig(CHART_DIR / "02_total_triples_per_judge.png", dpi=150)
plt.close()

# 7c. Pairwise overlap heatmap
fig, ax = plt.subplots(figsize=(9, 7))
mask = np.eye(N_JUDGES, dtype=bool)
sns.heatmap(pair_matrix, annot=True, fmt="d", cmap="Blues",
            linewidths=0.5, ax=ax, mask=mask,
            xticklabels=JUDGES, yticklabels=JUDGES)
for i, j in enumerate(JUDGES):
    ax.text(i + 0.5, i + 0.5, str(pair_matrix.loc[j, j]),
            ha="center", va="center", fontsize=9, color="gray", fontstyle="italic")
ax.set_title("Pairwise overlap (# shared triples)", fontsize=13, fontweight="bold")
plt.xticks(rotation=40, ha="right")
plt.tight_layout()
plt.savefig(CHART_DIR / "03_pairwise_overlap_heatmap.png", dpi=150)
plt.close()

# 7d. Jaccard similarity heatmap
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(jaccard.astype(float), annot=True, fmt=".2f", cmap="YlGnBu",
            linewidths=0.5, ax=ax, vmin=0, vmax=1,
            xticklabels=JUDGES, yticklabels=JUDGES)
ax.set_title("Jaccard similarity between judges", fontsize=13, fontweight="bold")
plt.xticks(rotation=40, ha="right")
plt.tight_layout()
plt.savefig(CHART_DIR / "04_jaccard_similarity_heatmap.png", dpi=150)
plt.close()

# 7e. Consensus level distribution
level_counts = vote_counts["n_judges"].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(9, 5))
colors_bar = plt.cm.RdYlGn(np.linspace(0.15, 0.9, N_JUDGES))
bars = ax.bar(level_counts.index, level_counts.values,
              color=[colors_bar[i - 1] for i in level_counts.index],
              edgecolor="white", linewidth=0.8)
for bar, val in zip(bars, level_counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
            str(val), ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_xlabel("Number of judges agreeing", fontsize=11)
ax.set_ylabel("Number of unique triples", fontsize=11)
ax.set_title("How many triples reach each consensus level?", fontsize=13, fontweight="bold")
ax.set_xticks(range(1, N_JUDGES + 1))
ax.set_xticklabels([f"{n}/11" for n in range(1, N_JUDGES + 1)])
plt.tight_layout()
plt.savefig(CHART_DIR / "05_consensus_distribution.png", dpi=150)
plt.close()

# 7f. Unanimous by source model
if not consensus[N_JUDGES].empty:
    src_counts = consensus[N_JUDGES]["source_file"].value_counts()
    fig, ax = plt.subplots(figsize=(9, 5))
    src_counts.plot.bar(ax=ax, color=sns.color_palette("Set2", len(src_counts)),
                        edgecolor="white")
    for i, val in enumerate(src_counts.values):
        ax.text(i, val + 0.3, str(val), ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Source file (model)", fontsize=11)
    ax.set_ylabel("Triples with unanimous consensus", fontsize=11)
    ax.set_title("Unanimous (11/11) consensus triples by source model",
                 fontsize=13, fontweight="bold")
    plt.xticks(rotation=40, ha="right")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "06_unanimous_by_source_model.png", dpi=150)
    plt.close()

# 7g. Judge quality ≥4/11
fig, ax = plt.subplots(figsize=(10, 5))
high_frac_4 = {}
for j in JUDGES:
    jdf = frames[j].merge(vote_counts[KEY + ["n_judges"]], on=KEY, how="left")
    total = len(jdf)
    high_frac_4[j] = ((jdf["n_judges"] >= 4).sum() / total * 100) if total else 0
hf_series = pd.Series(high_frac_4).sort_values(ascending=False)
bars = ax.bar(hf_series.index, hf_series.values,
              color=[JUDGE_COLORS[j] for j in hf_series.index],
              edgecolor="white", linewidth=0.8)
for bar, val in zip(bars, hf_series.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
ax.set_ylabel("% of judge's triples with ≥4 consensus", fontsize=11)
ax.set_title("Judge quality: fraction of flagged triples with high consensus (≥4/11)",
             fontsize=12, fontweight="bold")
ax.set_xticklabels(hf_series.index, rotation=35, ha="right")
plt.tight_layout()
plt.savefig(CHART_DIR / "07_judge_quality_high_consensus_4.png", dpi=150)
plt.close()

# 7h. Judge quality ≥5/11
fig, ax = plt.subplots(figsize=(10, 5))
high_frac_5 = {}
for j in JUDGES:
    jdf = frames[j].merge(vote_counts[KEY + ["n_judges"]], on=KEY, how="left")
    total = len(jdf)
    high_frac_5[j] = ((jdf["n_judges"] >= 5).sum() / total * 100) if total else 0
hf_series = pd.Series(high_frac_5).sort_values(ascending=False)
bars = ax.bar(hf_series.index, hf_series.values,
              color=[JUDGE_COLORS[j] for j in hf_series.index],
              edgecolor="white", linewidth=0.8)
for bar, val in zip(bars, hf_series.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
ax.set_ylabel("% of judge's triples with ≥5 consensus", fontsize=11)
ax.set_title("Judge quality: fraction of flagged triples with high consensus (≥5/11)",
             fontsize=12, fontweight="bold")
ax.set_xticklabels(hf_series.index, rotation=35, ha="right")
plt.tight_layout()
plt.savefig(CHART_DIR / "08_judge_quality_high_consensus_5.png", dpi=150)
plt.close()

# 7i. Judge quality ≥6/11
fig, ax = plt.subplots(figsize=(10, 5))
high_frac_5 = {}
for j in JUDGES:
    jdf = frames[j].merge(vote_counts[KEY + ["n_judges"]], on=KEY, how="left")
    total = len(jdf)
    high_frac_5[j] = ((jdf["n_judges"] >= 6).sum() / total * 100) if total else 0
hf_series = pd.Series(high_frac_5).sort_values(ascending=False)
bars = ax.bar(hf_series.index, hf_series.values,
              color=[JUDGE_COLORS[j] for j in hf_series.index],
              edgecolor="white", linewidth=0.8)
for bar, val in zip(bars, hf_series.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
ax.set_ylabel("% of judge's triples with ≥6 consensus", fontsize=11)
ax.set_title("Judge quality: fraction of flagged triples with high consensus (≥6/11)",
             fontsize=12, fontweight="bold")
ax.set_xticklabels(hf_series.index, rotation=35, ha="right")
plt.tight_layout()
plt.savefig(CHART_DIR / "09_judge_quality_high_consensus_6.png", dpi=150)
plt.close()

# 7j. Judge quality ≥7/11
fig, ax = plt.subplots(figsize=(10, 5))
high_frac_5 = {}
for j in JUDGES:
    jdf = frames[j].merge(vote_counts[KEY + ["n_judges"]], on=KEY, how="left")
    total = len(jdf)
    high_frac_5[j] = ((jdf["n_judges"] >= 7).sum() / total * 100) if total else 0
hf_series = pd.Series(high_frac_5).sort_values(ascending=False)
bars = ax.bar(hf_series.index, hf_series.values,
              color=[JUDGE_COLORS[j] for j in hf_series.index],
              edgecolor="white", linewidth=0.8)
for bar, val in zip(bars, hf_series.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
ax.set_ylabel("% of judge's triples with ≥7 consensus", fontsize=11)
ax.set_title("Judge quality: fraction of flagged triples with high consensus (≥7/11)",
             fontsize=12, fontweight="bold")
ax.set_xticklabels(hf_series.index, rotation=35, ha="right")
plt.tight_layout()
plt.savefig(CHART_DIR / "10_judge_quality_high_consensus_7.png", dpi=150)
plt.close()

# 7j. Judge quality ≥8/11
fig, ax = plt.subplots(figsize=(10, 5))
high_frac_5 = {}
for j in JUDGES:
    jdf = frames[j].merge(vote_counts[KEY + ["n_judges"]], on=KEY, how="left")
    total = len(jdf)
    high_frac_5[j] = ((jdf["n_judges"] >= 8).sum() / total * 100) if total else 0
hf_series = pd.Series(high_frac_5).sort_values(ascending=False)
bars = ax.bar(hf_series.index, hf_series.values,
              color=[JUDGE_COLORS[j] for j in hf_series.index],
              edgecolor="white", linewidth=0.8)
for bar, val in zip(bars, hf_series.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
ax.set_ylabel("% of judge's triples with ≥8 consensus", fontsize=11)
ax.set_title("Judge quality: fraction of flagged triples with high consensus (≥8/11)",
             fontsize=12, fontweight="bold")
ax.set_xticklabels(hf_series.index, rotation=35, ha="right")
plt.tight_layout()
plt.savefig(CHART_DIR / "11_judge_quality_high_consensus_8.png", dpi=150)
plt.close()

# 7k. Judge quality ≥9/11
fig, ax = plt.subplots(figsize=(10, 5))
high_frac_5 = {}
for j in JUDGES:
    jdf = frames[j].merge(vote_counts[KEY + ["n_judges"]], on=KEY, how="left")
    total = len(jdf)
    high_frac_5[j] = ((jdf["n_judges"] >= 9).sum() / total * 100) if total else 0
hf_series = pd.Series(high_frac_5).sort_values(ascending=False)
bars = ax.bar(hf_series.index, hf_series.values,
              color=[JUDGE_COLORS[j] for j in hf_series.index],
              edgecolor="white", linewidth=0.8)
for bar, val in zip(bars, hf_series.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
ax.set_ylabel("% of judge's triples with ≥9 consensus", fontsize=11)
ax.set_title("Judge quality: fraction of flagged triples with high consensus (≥9/11)",
             fontsize=12, fontweight="bold")
ax.set_xticklabels(hf_series.index, rotation=35, ha="right")
plt.tight_layout()
plt.savefig(CHART_DIR / "12_judge_quality_high_consensus_9.png", dpi=150)
plt.close()

# ── 8. Summary text ───────────────────────────────────────────────────────

summary_lines = [
    "JUDGE CONSENSUS ANALYSIS – SUMMARY",
    "=" * 50,
    f"Total unique triples across all judges : {len(vote_counts):,}",
    "",
    "Consensus distribution:",
]
for n in range(N_JUDGES, 0, -1):
    summary_lines.append(f"  {n}/11 judges agree : {len(consensus[n]):,} triples")

summary_lines += ["", "Total triples flagged per judge:"]
for j in JUDGES:
    summary_lines.append(f"  {j:30s}: {len(sets[j]):,}")

summary_lines += ["", "% of each judge's triples reaching ≥5/11 consensus (quality proxy):"]
for j, pct in sorted(high_frac_5.items(), key=lambda x: -x[1]):
    summary_lines.append(f"  {j:30s}: {pct:.1f}%")

(OUT_DIR / "summary.txt").write_text("\n".join(summary_lines))
print("\nAll done! Output in:", OUT_DIR)
print("\n".join(summary_lines))