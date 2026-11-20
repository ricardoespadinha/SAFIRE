import pandas as pd
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import os
import warnings

# ============================================================
# Script used to create test and train splits for DynaSent and
# SST. Also used to analyze the difference between winning and
# neutral scores in misclassified neutral prompts, which we 
# used for RoBERTa Dynamic.
# ============================================================

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

SST_FILE = "sst3-dev.csv"
DYNASENT_FILE = "dynasent-v1.1-round02-dynabench-test.jsonl"

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"

# Each dataset is split independently: 50% train, 50% test.
TEST_FRACTION = 0.50
SPLIT_SEED = 4936

# If True, each dataset's train half and test half are each split so
# that the label distribution (negative/neutral/positive) matches the
# full dataset's own distribution -- no undersampling. For SST3 (444
# pos / 229 neu / 428 neg) this preserves that same proportion in both
# halves; for DynaSent (already 240/240/240) this naturally gives an
# even third of each label in both halves. If False, splitting is
# purely random per dataset.
USE_STRATIFIED_SPLIT = True

# Batch size makes inference substantially faster than one sentence at a time.
BATCH_SIZE = 32

# ============================================================
# OUTPUT / LOGGING
# ============================================================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = Path(f"roberta_analysis_seed{SPLIT_SEED}_{timestamp}")
output_dir.mkdir(parents=True, exist_ok=True)

log_file_path = output_dir / "analysis_log.txt"
log_file = open(log_file_path, "w", encoding="utf-8")


def print_log(message="", console=True):
    if console:
        print(message)
    log_file.write(str(message) + "\n")
    log_file.flush()


print_log("=" * 80)
print_log("RoBERTa SCORE ANALYSIS - 50/50 DATASET SPLIT (FIXED SEED)")
print_log("=" * 80)
print_log(f"Output directory: {output_dir}")
print_log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print_log(f"Split seed (fixed): {SPLIT_SEED}")
print_log(f"Test fraction: {TEST_FRACTION:.2f}")
print_log(f"Stratified (per-label proportion preserved) split: {USE_STRATIFIED_SPLIT}")

# ============================================================
# LOAD DATASETS
# ============================================================

def load_datasets():
    frames = []

    print_log("\n" + "=" * 80)
    print_log("LOADING DATASETS")
    print_log("=" * 80)

    try:
        df_sst = pd.read_csv(SST_FILE)
        df_sst["dataset"] = "SST3"
        df_sst["true_label"] = df_sst["label"].astype(str).str.lower()
        if "sentence" not in df_sst.columns:
            raise ValueError("SST3 file must contain a 'sentence' column.")
        print_log(f"SST3: {len(df_sst)} samples")
        frames.append(df_sst)
    except Exception as e:
        print_log(f"ERROR loading SST3: {e}")

    try:
        data = []
        with open(DYNASENT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))

        df_dynasent = pd.DataFrame(data)
        df_dynasent["dataset"] = "DynaSent"
        df_dynasent["true_label"] = (
            df_dynasent["gold_label"].astype(str).str.lower()
        )
        if "sentence" not in df_dynasent.columns:
            raise ValueError("DynaSent file must contain a 'sentence' column.")
        print_log(f"DynaSent: {len(df_dynasent)} samples")
        frames.append(df_dynasent)
    except Exception as e:
        print_log(f"ERROR loading DynaSent: {e}")

    if not frames:
        raise RuntimeError(
            "No datasets loaded. Put SST3-dev.csv and "
            "dynasent-v1.1-round02-dynabench-test.jsonl in the working directory."
        )

    return pd.concat(frames, ignore_index=True)


df_all = load_datasets()

# ============================================================
# 50/50 SPLIT -- DONE SEPARATELY FOR SST3 AND DYNASENT
# ============================================================

def split_each_dataset(df, seed):
    train_parts = []
    test_parts = []

    for dataset_name, group in df.groupby("dataset", sort=False):
        group = group.sample(frac=1.0, random_state=seed).reset_index(drop=True)

        n_test = int(round(len(group) * TEST_FRACTION))
        n_test = max(1, min(len(group) - 1, n_test))

        test_part = group.iloc[:n_test].copy()
        train_part = group.iloc[n_test:].copy()

        train_part["split"] = "train"
        test_part["split"] = "test"

        train_parts.append(train_part)
        test_parts.append(test_part)

        print_log(
            f"\n{dataset_name} split:"
            f"\n  total = {len(group)}"
            f"\n  train = {len(train_part)}"
            f"\n  test  = {len(test_part)}"
        )

    train = pd.concat(train_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)
    return train, test


def split_each_dataset_stratified(df, seed, labels=("negative", "neutral", "positive")):
    """
    Like split_each_dataset, but splits each (dataset, label) group by
    TEST_FRACTION separately, then recombines. This guarantees that the
    label distribution of the train half and the test half both match
    the label distribution of the full dataset (no undersampling):
      - SST3 (444 pos / 229 neu / 428 neg) -> train and test each keep
        that same ~44%/23%/43% split.
      - DynaSent (240/240/240, already balanced) -> train and test each
        come out an even third negative / neutral / positive.
    """
    train_parts = []
    test_parts = []

    for dataset_name, group in df.groupby("dataset", sort=False):
        ds_train_parts = []
        ds_test_parts = []
        counts = {}

        for lbl in labels:
            lbl_df = group[group["true_label"] == lbl]
            counts[lbl] = len(lbl_df)
            if len(lbl_df) == 0:
                continue

            lbl_df = lbl_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

            n_test = int(round(len(lbl_df) * TEST_FRACTION))
            n_test = max(1, min(len(lbl_df) - 1, n_test)) if len(lbl_df) > 1 else len(lbl_df)

            test_lbl = lbl_df.iloc[:n_test].copy()
            train_lbl = lbl_df.iloc[n_test:].copy()

            ds_test_parts.append(test_lbl)
            ds_train_parts.append(train_lbl)

        ds_train = pd.concat(ds_train_parts, ignore_index=True)
        ds_test = pd.concat(ds_test_parts, ignore_index=True)

        ds_train["split"] = "train"
        ds_test["split"] = "test"

        # Reshuffle so the rows aren't grouped by label in the output.
        ds_train = ds_train.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        ds_test = ds_test.sample(frac=1.0, random_state=seed).reset_index(drop=True)

        train_parts.append(ds_train)
        test_parts.append(ds_test)

        train_label_counts = ds_train["true_label"].value_counts().to_dict()
        test_label_counts = ds_test["true_label"].value_counts().to_dict()

        print_log(
            f"\n{dataset_name} STRATIFIED split (label proportions preserved; "
            f"full-dataset counts: {counts}):"
            f"\n  train = {len(ds_train)}  ({train_label_counts})"
            f"\n  test  = {len(ds_test)}  ({test_label_counts})"
        )

    if not train_parts:
        raise RuntimeError(
            "Balanced split produced no data -- check that every dataset "
            "has at least one example of each label."
        )

    train = pd.concat(train_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)
    return train, test


def perform_split(df, seed):
    """Single entry point used everywhere a split is needed, so the
    USE_STRATIFIED_SPLIT toggle only has to be checked in one place."""
    if USE_STRATIFIED_SPLIT:
        return split_each_dataset_stratified(df, seed)
    return split_each_dataset(df, seed)


# ============================================================
# MODEL
# ============================================================

print_log("\n" + "=" * 80)
print_log("LOADING RoBERTa")
print_log("=" * 80)
print_log(f"Model: {MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print_log(f"Device: {device}")


@torch.no_grad()
def get_roberta_scores_batch(texts):
    encoded = tokenizer(
        [str(x) for x in texts],
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    logits = model(**encoded).logits.detach().cpu().numpy()
    probs = softmax(logits, axis=1)

    return probs


def predict_dataframe(df):
    df = df.copy()
    all_probs = []

    texts = df["sentence"].astype(str).tolist()

    for start in range(0, len(texts), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(texts))
        probs = get_roberta_scores_batch(texts[start:end])
        all_probs.append(probs)

        if end % (BATCH_SIZE * 10) == 0 or end == len(texts):
            print_log(f"  Predicted {end}/{len(texts)}")

    probs = np.vstack(all_probs)

    df["roberta_neg"] = probs[:, 0]
    df["roberta_neu"] = probs[:, 1]
    df["roberta_pos"] = probs[:, 2]

    labels = np.array(["negative", "neutral", "positive"])
    df["roberta_prediction"] = labels[np.argmax(probs, axis=1)]
    df["correct"] = df["roberta_prediction"] == df["true_label"]

    return df


# ============================================================
# ANALYSIS FUNCTIONS
# ============================================================

def build_misclassification_table(df_test):
    neutral = df_test[df_test["true_label"] == "neutral"].copy()
    wrong = neutral[~neutral["correct"]].copy()

    records = []

    for _, row in wrong.iterrows():
        if row["roberta_prediction"] == "positive":
            winner_score = row["roberta_pos"]
        elif row["roberta_prediction"] == "negative":
            winner_score = row["roberta_neg"]
        else:
            continue

        records.append({
            "dataset": row["dataset"],
            "true_label": "neutral",
            "predicted_as": row["roberta_prediction"],
            "pos_score": row["roberta_pos"],
            "neu_score": row["roberta_neu"],
            "neg_score": row["roberta_neg"],
            "winner_score": winner_score,
            "neutral_score": row["roberta_neu"],
            "score_diff": winner_score - row["roberta_neu"],
            "sentence": row["sentence"],
        })

    return neutral, pd.DataFrame(records)


# ============================================================
# PREDICT ONCE
# ============================================================

print_log("\n" + "=" * 80)
print_log("RUNNING RoBERTa PREDICTIONS")
print_log("=" * 80)

df_pred = predict_dataframe(df_all)

# ============================================================
# FIXED SPLIT
# ============================================================

selected_seed = SPLIT_SEED
print_log("\n" + "=" * 80)
print_log(f"SPLITTING WITH FIXED SEED {selected_seed}")
print_log("=" * 80)

df_train, df_test = perform_split(df_pred, selected_seed)
df_neutral, df_misclass = build_misclassification_table(df_test)

# ============================================================
# FINAL TEST-SET RESULTS
# ============================================================

print_log("\n" + "=" * 80)
print_log("TEST-SET RESULTS")
print_log("=" * 80)

test_accuracy = (
    df_test["correct"].mean() * 100
    if len(df_test)
    else 0
)

neutral_correct = df_neutral[df_neutral["correct"]]
neutral_accuracy = (
    neutral_correct.shape[0] / len(df_neutral) * 100
    if len(df_neutral)
    else 0
)

print_log(f"Test samples: {len(df_test)}")
print_log(f"Test accuracy: {test_accuracy:.2f}%")
print_log(f"Test neutral samples: {len(df_neutral)}")
print_log(f"Neutral accuracy: {neutral_accuracy:.2f}%")
print_log(f"Neutral misclassifications: {len(df_misclass)}")

if len(df_misclass) > 0:
    x = df_misclass["score_diff"]

    print_log("\nOverall score difference statistics (winner - neutral):")
    print_log(f"  Mean: {x.mean():.4f}")
    print_log(f"  Median: {x.median():.4f}")
    print_log(f"  Std Dev: {x.std():.4f}")
    print_log(f"  Min: {x.min():.4f}")
    print_log(f"  Max: {x.max():.4f}")

    print_log("\nPercentile analysis of score differences:")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        print_log(f"  {p}th percentile: {x.quantile(p / 100):.4f}")

else:
    print_log("\nNo neutral misclassifications occurred in the selected test set.")

# ============================================================
# CORRECTLY CLASSIFIED NEUTRALS
# ============================================================

if len(neutral_correct) > 0:
    neutral_correct = neutral_correct.copy()
    neutral_correct["max_non_neutral"] = neutral_correct[
        ["roberta_pos", "roberta_neg"]
    ].max(axis=1)
    neutral_correct["neu_margin"] = (
        neutral_correct["roberta_neu"]
        - neutral_correct["max_non_neutral"]
    )

    print_log("\n" + "=" * 80)
    print_log("CORRECTLY CLASSIFIED NEUTRAL ANALYSIS")
    print_log("=" * 80)
    print_log(f"Mean neutral margin: {neutral_correct['neu_margin'].mean():.4f}")
    print_log(
        f"Median neutral margin: {neutral_correct['neu_margin'].median():.4f}"
    )

# ============================================================
# SAVE DATA
# ============================================================

df_train.to_csv(output_dir / "train_split.csv", index=False)
df_test.to_csv(output_dir / "test_split.csv", index=False)
df_pred.to_csv(output_dir / "all_predictions.csv", index=False)
df_neutral.to_csv(output_dir / "test_neutral_samples.csv", index=False)

if len(df_misclass) > 0:
    df_misclass.to_csv(
        output_dir / "test_neutral_misclassifications.csv",
        index=False,
    )

if len(neutral_correct) > 0:
    neutral_correct.to_csv(
        output_dir / "test_neutral_correct.csv",
        index=False,
    )

# ============================================================
# SUMMARY CSV
# ============================================================

summary = {
    "split_seed": selected_seed,
    "test_fraction": TEST_FRACTION,
    "train_samples": len(df_train),
    "test_samples": len(df_test),
    "test_accuracy": test_accuracy,
    "test_neutral_samples": len(df_neutral),
    "test_neutral_accuracy": neutral_accuracy,
    "test_neutral_misclassifications": len(df_misclass),
}

if len(df_misclass) > 0:
    x = df_misclass["score_diff"]
    summary.update({
        "score_diff_mean": x.mean(),
        "score_diff_median": x.median(),
        "score_diff_p10": x.quantile(0.10),
        "score_diff_p25": x.quantile(0.25),
        "score_diff_p50": x.quantile(0.50),
        "score_diff_p75": x.quantile(0.75),
        "score_diff_p90": x.quantile(0.90),
        "score_diff_p95": x.quantile(0.95),
        "score_diff_p99": x.quantile(0.99),
    })

pd.DataFrame([summary]).to_csv(
    output_dir / "summary_statistics.csv",
    index=False,
)

# ============================================================
# PLOTS
# ============================================================

try:
    sns.set_style("whitegrid")

    if len(df_misclass) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            "RoBERTa Test-Set Analysis: Misclassified Neutrals",
            fontsize=16,
            fontweight="bold",
        )

        axes[0, 0].hist(
            df_misclass["score_diff"],
            bins=30,
            edgecolor="black",
            alpha=0.7,
        )
        axes[0, 0].axvline(
            df_misclass["score_diff"].median(),
            color="red",
            linestyle="--",
            label=f"Median: {df_misclass['score_diff'].median():.4f}",
        )
        axes[0, 0].set_xlabel("Score Difference (Winner - Neutral)")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].set_title("Score Difference Distribution")
        axes[0, 0].legend()

        axes[0, 1].hist(
            df_misclass["neutral_score"],
            bins=30,
            edgecolor="black",
            alpha=0.7,
        )
        axes[0, 1].set_xlabel("Neutral Score")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].set_title("Neutral Scores")

        axes[1, 0].hist(
            df_misclass["winner_score"],
            bins=30,
            edgecolor="black",
            alpha=0.7,
        )
        axes[1, 0].set_xlabel("Winner Score")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].set_title("Winning Scores")

        axes[1, 1].scatter(
            df_misclass["neutral_score"],
            df_misclass["winner_score"],
            alpha=0.5,
        )
        axes[1, 1].plot([0, 1], [0, 1], "r--")
        axes[1, 1].set_xlabel("Neutral Score")
        axes[1, 1].set_ylabel("Winner Score")
        axes[1, 1].set_title("Neutral vs Winner Scores")
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(
            output_dir / "test_misclassification_analysis.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

except Exception as e:
    print_log(f"Visualization error: {e}")

# ============================================================
# FINAL
# ============================================================

print_log("\n" + "=" * 80)
print_log("ANALYSIS COMPLETE")
print_log("=" * 80)
print_log(f"Results saved to: {output_dir}/")
print_log("Files:")
for p in sorted(output_dir.iterdir()):
    print_log(f"  - {p.name}")

log_file.close()
print(f"\nResults saved to: {output_dir}/")