#!/usr/bin/env python3
"""
Stratified sample of SAFIRE responses for hand verification, stratified by the
number of judges (0-3) that labelled the response COMPLIED.

Unit of analysis (one "prompt" in the sense of your question):
    (victim model, variation set, intensity level)  applied to one base prompt
i.e. one attack prompt sent to one victim, with its one response.
Full population = 6 victims x 3 variation sets x 578 base prompts x 11 intensities.

How the files fit together
  * complied_results_* files contain ONLY responses a judge called COMPLIED.
      - `prompt`            = the base prompt (== intensity_0 text in the safire files)
      - `response_variant`  = response_<k>  -> the intensity_<k> column
      - `source_file`       = judged_results_<victim>_judge_<judge>_<variation set>.csv
  A unit's vote count = number of the 3 judge files it appears in; units in none = 0 votes.
  * safire_<set>.csv give the full grid (the population) and the actual attack text.

Within each stratum, n is allocated proportionally across victim x category cells
(largest-remainder rounding), then sampled uniformly at random within each cell.

Usage
  python sample_by_judge_votes.py --data-dir /path/to/csvs --out-dir out
  # response text comes from the combined files made by combine_jailbreak_results.py
  python sample_by_judge_votes.py --data-dir /path/to/csvs --responses-dir combined --out-dir out

Outputs (in --out-dir)
  stratum_population_sizes.csv   population size, sample size per stratum
  stratum_allocation.csv         per stratum x victim x category: population + sample counts
  hand_verification_sample.csv   the sampled prompts / responses / judge votes
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

JUDGE_FILES = {  # judge label -> complied-results file
    "gemma4": "complied_results_safire_combined_judge_g4.csv",
    "gpt_oss_safeguard": "complied_results_safire_combined_judge_gpt.csv",
    "qwen3.6": "complied_results_safire_combined_judge_q3.6.csv",
}
SET_FILES = {  # variation set -> full prompt file
    "original": "safire_original.csv",
    "g4": "safire_g4.csv",
    "q3.6": "safire_q3.6.csv",
}
KEY = ["victim", "variation_set", "intensity", "base_prompt"]


# ----------------------------------------------------------------------------- load
def load_population(data_dir: Path, victims: list[str]) -> pd.DataFrame:
    """Full grid: every (victim, set, base prompt, intensity) with its attack text."""
    frames = []
    for vset, fname in SET_FILES.items():
        df = pd.read_csv(data_dir / fname)
        int_cols = [c for c in df.columns if c.startswith("intensity_")]
        df["base_prompt"] = df["intensity_0"]  # copy, so intensity_0 is also melted as its own level
        long = df.melt(id_vars=["category", "base_prompt"], value_vars=int_cols,
                       var_name="intensity", value_name="attack_prompt")
        long["intensity"] = long["intensity"].str.replace("intensity_", "", regex=False).astype(int)
        long["variation_set"] = vset
        frames.append(long)
    grid = pd.concat(frames, ignore_index=True)
    pop = pd.concat([grid.assign(victim=v) for v in victims], ignore_index=True)
    return pop


def load_judge(data_dir: Path, judge: str, fname: str) -> pd.DataFrame:
    df = pd.read_csv(data_dir / fname)
    ext = df["source_file"].str.extract(r"judged_results_(.+?)_judge_.*_(original|g4|q3\.6)\.csv$")
    df["victim"], df["variation_set"] = ext[0], ext[1]
    if df["victim"].isna().any():
        raise ValueError(f"Could not parse victim/set from source_file in {fname}")
    df["intensity"] = df["response_variant"].str.replace("response_", "", regex=False).astype(int)
    df = df.rename(columns={"prompt": "base_prompt"})
    if df.duplicated(KEY).any():
        raise ValueError(f"Duplicate units in {fname}")
    df[f"vote_{judge}"] = 1
    return df


def load_responses(resp_dir: Path) -> pd.DataFrame:
    """Long table of every victim response: victim, variation_set, intensity, prompt(base), response."""
    frames = []
    for vset, suffix in [("original", "original"), ("g4", "g4"), ("q3.6", "q3.6")]:
        df = pd.read_csv(resp_dir / f"jailbreak_results_combined_{suffix}.csv")
        rc = [c for c in df.columns if c.startswith("response_")]
        long = df.melt(id_vars=["victim", "prompt"], value_vars=rc, var_name="intensity", value_name="response")
        long["intensity"] = long["intensity"].str.replace("response_", "", regex=False).astype(int)
        long["variation_set"] = vset
        frames.append(long)
    return pd.concat(frames, ignore_index=True)


# ----------------------------------------------------------------------------- sampling
def allocate(cell_sizes: pd.Series, n: int) -> pd.Series:
    """Proportional allocation with largest-remainder rounding, capped at cell size."""
    n = min(n, int(cell_sizes.sum()))
    exact = n * cell_sizes / cell_sizes.sum()
    alloc = np.floor(exact).astype(int)
    remainder = n - int(alloc.sum())
    order = (exact - alloc).sort_values(ascending=False).index
    for idx in order[:remainder]:
        alloc[idx] += 1
    over = alloc > cell_sizes  # only possible for tiny cells; push surplus to cells with room
    while over.any():
        surplus = int((alloc[over] - cell_sizes[over]).sum())
        alloc[over] = cell_sizes[over]
        room = (cell_sizes - alloc)
        for idx in (exact - alloc).where(room > 0).dropna().sort_values(ascending=False).index[:surplus]:
            alloc[idx] += 1
        over = alloc > cell_sizes
    return alloc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=".", type=Path)
    ap.add_argument("--out-dir", default="out", type=Path)
    ap.add_argument("--responses-dir", default=".", type=Path,
                    help="folder with jailbreak_results_combined_{original,g4,q3.6}.csv "
                         "(output of combine_jailbreak_results.py); supplies the response text for every unit")
    ap.add_argument("--n", type=int, default=50, help="sample size per stratum")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    judges = {j: load_judge(args.data_dir, j, f) for j, f in JUDGE_FILES.items()}
    victims = sorted(set().union(*[set(d["victim"]) for d in judges.values()]))
    pop = load_population(args.data_dir, victims)
    print(f"Population: {len(pop):,} units  ({len(victims)} victims: {', '.join(victims)})")

    # attach votes + response text / source files
    for j, d in judges.items():
        cols = KEY + [f"vote_{j}", "source_file", "response"]
        sub = d[cols].rename(columns={"source_file": f"source_file_{j}", "response": f"response_{j}"})
        pop = pop.merge(sub, on=KEY, how="left", validate="one_to_one")
        n_unmatched = len(d) - pop[f"vote_{j}"].notna().sum()
        if n_unmatched:
            raise ValueError(f"{n_unmatched} complied rows from judge {j} not found in the population grid")
        pop[f"vote_{j}"] = pop[f"vote_{j}"].fillna(0).astype(int)

    vote_cols = [f"vote_{j}" for j in judges]
    pop["n_votes"] = pop[vote_cols].sum(axis=1)

    # canonical response text for EVERY unit, from the combined response files
    resp = load_responses(args.responses_dir).rename(columns={"prompt": "base_prompt"})
    pop = pop.merge(resp, on=KEY, how="left", validate="one_to_one")
    if pop["response"].isna().any():
        raise ValueError(f"{int(pop['response'].isna().sum())} units have no response in --responses-dir")

    # flag judge votes that were cast on a DIFFERENT response text than the canonical one
    for j in judges:
        rj = pop[f"response_{j}"]
        pop[f"judged_other_response_{j}"] = rj.notna() & (rj != pop["response"])
        pop = pop.drop(columns=f"response_{j}")
    mismatch_cols = [f"judged_other_response_{j}" for j in judges]
    pop["any_judge_saw_different_response"] = pop[mismatch_cols].any(axis=1)
    for j in judges:
        print(f"  judge {j}: {int(pop[f'judged_other_response_{j}'].sum())} COMPLIED votes were on a "
              f"response that differs from the combined-file response")

    # ------------------------------------------------------------------ strata
    sizes = pop.groupby("n_votes").size().reindex([0, 1, 2, 3], fill_value=0)
    rng_seed = args.seed
    samples, alloc_rows = [], []
    for votes in [0, 1, 2, 3]:
        stratum = pop[pop.n_votes == votes]
        cells = stratum.groupby(["victim", "category"]).size()
        alloc = allocate(cells, args.n)
        for (v, c), k in alloc.items():
            alloc_rows.append({"n_votes": votes, "victim": v, "category": c,
                               "population": int(cells[(v, c)]), "sample": int(k)})
            if k:
                cell = stratum[(stratum.victim == v) & (stratum.category == c)]
                samples.append(cell.sample(int(k), random_state=rng_seed))
    sample = pd.concat(samples, ignore_index=True)

    # ------------------------------------------------------------------ outputs
    pop_sizes = pd.DataFrame({"n_votes": sizes.index, "population_size": sizes.values})
    pop_sizes["sample_size"] = pop_sizes["n_votes"].map(sample.groupby("n_votes").size()).fillna(0).astype(int)
    pop_sizes["share_of_total"] = (pop_sizes["population_size"] / pop_sizes["population_size"].sum()).round(4)
    pop_sizes.to_csv(args.out_dir / "stratum_population_sizes.csv", index=False)
    pd.DataFrame(alloc_rows).to_csv(args.out_dir / "stratum_allocation.csv", index=False)

    out_cols = (["n_votes", "victim", "category", "variation_set", "intensity",
                 "base_prompt", "attack_prompt", "response"] + vote_cols
                + [f"source_file_{j}" for j in judges] + mismatch_cols + ["any_judge_saw_different_response"])
    sample = sample[out_cols].sort_values(["n_votes", "victim", "category", "variation_set", "intensity"])
    sample.to_csv(args.out_dir / "hand_verification_sample.csv", index=False)

    print("\nPopulation size per stratum (number of judge votes for COMPLIED):")
    print(pop_sizes.to_string(index=False))
    n_diff = int(sample["any_judge_saw_different_response"].sum())
    if n_diff:
        print(f"\nNOTE: {n_diff} sampled rows had a judge vote cast on a different response than the one shown "
              f"(see judged_other_response_* columns).")
    print(f"\nWrote outputs to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()