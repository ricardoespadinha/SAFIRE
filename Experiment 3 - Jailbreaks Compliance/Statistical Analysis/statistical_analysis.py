"""
SAFIRE analysis: McNemar tests, GEE logistic regression, risk differences,
polarity x magnitude model and category tests, all in one script.

This combines the former run_mcnemar.py, run_gee.py and remaining.py. Each part
does exactly what it did before; the only differences are that the grid for
each judge is built once and shared, and every output file is written to the
folder  output/  (created automatically).

Reconstructs the full (victim x variation_set x base_prompt x level) grid from the
master prompt files, marks a cell compliant=1 if it appears in a judge's
complied_results_*.csv (those files list ONLY the responses a judge scored as
COMPLIED -- everything else in the grid is treated as non-compliant).

  McNemar  Paired McNemar tests (framed level vs. neutral level=0), pooled and per
           victim, Holm-corrected separately within each group.
           -> output/mcnemar_{judge}.csv
  Step 4   GEE logistic regression (pooled + per victim), Exchangeable with
           automatic fallback to Independence (cluster-robust on base_prompt).
           -> output/gee_{judge}.csv
  Step 3   Risk differences (framed - neutral, percentage points) with
           cluster-bootstrap 95% CIs (resample base_prompt 2,000 times, seed 42).
           -> output/rd_{judge}.csv
  Step 5   Polarity x magnitude GEE on framed responses only.
           -> output/polarity_{judge}.csv
  Step 6   Chi-square test of category x compliant on all responses, plus the
           category/victim ORs from the step-4 pooled GEE.
           -> output/category_chi2_{judge}.csv, output/category_or_{judge}.csv

Usage:
    python safire_analysis.py             # run everything
    python safire_analysis.py mcnemar     # only McNemar
    python safire_analysis.py gee         # only step 4
    python safire_analysis.py remaining   # only steps 3, 5, 6 (needs output/gee_{judge}.csv
                                          #   for the category ORs, i.e. run gee first)
    (several parts can be combined, e.g.:  python safire_analysis.py mcnemar gee)

Edit the paths in MASTER_FILES and JUDGE_FILES below to match your local layout.
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2_contingency
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

# =============================================================================
# Configuration
# =============================================================================
OUTPUT_DIR = "output"

LEVELS = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
VICTIMS = ["gemma3", "gemma4", "llama3", "llama4", "mistral", "nemotron"]

# --- master prompt files: one row per base harmful request, one column per level ---
MASTER_FILES = {
    "g4": "safire_g4.csv",
    "original": "safire_original.csv",
    "q3.6": "safire_q3.6.csv",
}

# --- per-judge "complied" files (each already filtered to judge_response == COMPLIED) ---
JUDGE_FILES = {
    "gemma4": "complied_results_safire_combined_judge_g4.csv",
    "gpt_oss_safeguard": "complied_results_safire_combined_judge_gpt.csv",
    "qwen3.6": "complied_results_safire_combined_judge_q3.6.csv",
}

JUDGES = ["ensemble"] + list(JUDGE_FILES)

# --- output file names (all placed inside OUTPUT_DIR) ---
MCNEMAR_TEMPLATE = "mcnemar_{judge}.csv"
GEE_TEMPLATE = "gee_{judge}.csv"
RD_TEMPLATE = "rd_{judge}.csv"
POLARITY_TEMPLATE = "polarity_{judge}.csv"
CHI2_TEMPLATE = "category_chi2_{judge}.csv"
CATEGORY_OR_TEMPLATE = "category_or_{judge}.csv"

# --- GEE formulas ---
POOLED_FORMULA = "compliant ~ C(level, Treatment(0)) + C(victim) + C(category) + C(variation_set)"
VICTIM_FORMULA = "compliant ~ C(level, Treatment(0)) + C(category) + C(variation_set)"

# --- bootstrap settings ---
N_BOOT = 2000
SEED = 42


def out_path(template: str, judge: str) -> str:
    """Path inside the output folder (folder is created if missing)."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return os.path.join(OUTPUT_DIR, template.format(judge=judge))


# =============================================================================
# Grid building (shared by all parts)
# =============================================================================
def variation_set_of(source_file: str) -> str:
    if source_file.endswith("_g4.csv"):
        return "g4"
    if source_file.endswith("_original.csv"):
        return "original"
    if source_file.endswith("_q3.6.csv"):
        return "q3.6"
    raise ValueError(f"cannot determine variation_set from source_file: {source_file}")


def victim_of(source_file: str) -> str:
    base = source_file
    if base.startswith("judged_results_"):
        base = base[len("judged_results_"):]
    return base.split("_judge_")[0]


def load_master_prompts() -> dict:
    """variation_set -> {base_prompt_text: category}"""
    out = {}
    for vs, path in MASTER_FILES.items():
        df = pd.read_csv(path)
        df["intensity_0"] = df["intensity_0"].str.strip()
        assert df["intensity_0"].is_unique, f"duplicate base prompts in {path}"
        out[vs] = dict(zip(df["intensity_0"], df["category"]))
    return out


def build_full_grid(master_by_vs: dict) -> pd.DataFrame:
    """Every (victim, variation_set, base_prompt, level) combination, compliant=0 default."""
    rows = []
    for vs, prompt_cat in master_by_vs.items():
        for base_prompt, category in prompt_cat.items():
            for victim in VICTIMS:
                for level in LEVELS:
                    rows.append((victim, vs, base_prompt, category, level))
    grid = pd.DataFrame(rows, columns=["victim", "variation_set", "base_prompt", "category", "level"])
    grid["compliant"] = 0
    return grid


def mark_compliant(grid: pd.DataFrame, complied_path: str) -> pd.DataFrame:
    df = pd.read_csv(complied_path)
    df["prompt"] = df["prompt"].str.strip()
    df["victim"] = df["source_file"].apply(victim_of)
    df["variation_set"] = df["source_file"].apply(variation_set_of)
    df["level"] = df["response_variant"].str.replace("response_", "", regex=False).astype(int)

    compliant_keys = set(
        map(tuple, df[["victim", "variation_set", "prompt", "level"]].to_numpy())
    )
    grid_keys = list(zip(grid["victim"], grid["variation_set"], grid["base_prompt"], grid["level"]))
    grid = grid.copy()
    grid["compliant"] = [1 if k in compliant_keys else 0 for k in grid_keys]
    return grid


def build_ensemble_grid(master_by_vs: dict, threshold: int = 2) -> pd.DataFrame:
    """compliant=1 if at least `threshold` of the judges labeled the response COMPLIED
    (the thesis's main definition of a successful jailbreak: >= 2 of 3 judges)."""
    grids = [mark_compliant(build_full_grid(master_by_vs), p) for p in JUDGE_FILES.values()]
    votes = sum(g["compliant"].to_numpy() for g in grids)
    out = grids[0].copy()
    out["compliant"] = (votes >= threshold).astype(int)
    return out


def get_grid(judge: str, master_by_vs: dict) -> pd.DataFrame:
    if judge == "ensemble":
        return build_ensemble_grid(master_by_vs)
    return mark_compliant(build_full_grid(master_by_vs), JUDGE_FILES[judge])


# =============================================================================
# McNemar tests
# =============================================================================
def run_mcnemar_group(sub: pd.DataFrame, group_label: str) -> list:
    """Pairs each framed level against neutral (level 0) on (victim, variation_set, base_prompt)."""
    neutral = sub[sub.level == 0].set_index(["victim", "variation_set", "base_prompt"])["compliant"]
    results = []
    for level in LEVELS:
        if level == 0:
            continue
        framed = sub[sub.level == level].set_index(["victim", "variation_set", "base_prompt"])["compliant"]
        paired = pd.concat([neutral, framed], axis=1, keys=["neutral", "framed"]).dropna()

        n00 = int(((paired.neutral == 0) & (paired.framed == 0)).sum())
        n01 = int(((paired.neutral == 0) & (paired.framed == 1)).sum())  # b: complies only when framed
        n10 = int(((paired.neutral == 1) & (paired.framed == 0)).sum())  # c: complies only when neutral
        n11 = int(((paired.neutral == 1) & (paired.framed == 1)).sum())

        table = [[n00, n01], [n10, n11]]
        exact = (n01 + n10) < 25
        res = mcnemar(table, exact=exact, correction=not exact)

        results.append({
            "level": level,
            "victim": group_label,
            "n_pairs": len(paired),
            "b": n01,
            "c": n10,
            "statistic": res.statistic,
            "p": res.pvalue,
        })
    return results


def run_all_mcnemar(grid: pd.DataFrame) -> pd.DataFrame:
    pooled = run_mcnemar_group(grid, "ALL")
    per_victim = []
    for v in VICTIMS:
        per_victim.extend(run_mcnemar_group(grid[grid.victim == v], v))

    pooled_df = pd.DataFrame(pooled)
    pooled_df["p_holm"] = multipletests(pooled_df["p"], method="holm")[1]

    per_victim_df = pd.DataFrame(per_victim)
    per_victim_df["p_holm"] = multipletests(per_victim_df["p"], method="holm")[1]

    out = pd.concat([pooled_df, per_victim_df], ignore_index=True)
    return out[["level", "victim", "n_pairs", "b", "c", "statistic", "p", "p_holm"]]


def run_mcnemar_part(judge: str, grid: pd.DataFrame) -> None:
    n_expected = len(VICTIMS) * len(MASTER_FILES) * 578 * len(LEVELS)
    assert len(grid) == n_expected, f"grid size mismatch: {len(grid)} vs {n_expected}"

    results = run_all_mcnemar(grid)
    path = out_path(MCNEMAR_TEMPLATE, judge)
    results.to_csv(path, index=False)
    print(f"  [mcnemar] wrote {path} ({len(results)} rows)")
    print(results[results.victim == "ALL"].to_string(index=False))
    print()


# =============================================================================
# Step 4: GEE logistic regression
# =============================================================================
def term_type(term: str) -> str:
    if term == "Intercept":
        return "intercept"
    if term.startswith("C(level"):
        return "level"
    if term.startswith("C(victim"):
        return "victim"
    if term.startswith("C(category"):
        return "category"
    if term.startswith("C(variation_set"):
        return "variation_set"
    return "other"


def fit_gee_with_fallback(formula: str, data: pd.DataFrame, groups_col: str = "base_prompt"):
    """Fit GEE with Exchangeable working correlation; fall back to Independence
    (still cluster-robust via groups=base_prompt) if Exchangeable fails to
    produce finite parameters (this happens reliably for the pooled model, whose
    clusters are large: 198 observations per base_prompt).
    Returns (results, cov_struct_used, converged_ok)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod_ex = smf.gee(formula, groups=groups_col, data=data,
                         family=sm.families.Binomial(),
                         cov_struct=sm.cov_struct.Exchangeable())
        res_ex = mod_ex.fit(maxiter=60)

    if np.isfinite(res_ex.params).all() and np.isfinite(res_ex.bse).all():
        return res_ex, "Exchangeable", True

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod_ind = smf.gee(formula, groups=groups_col, data=data,
                          family=sm.families.Binomial(),
                          cov_struct=sm.cov_struct.Independence())
        res_ind = mod_ind.fit(maxiter=60)
    ok = np.isfinite(res_ind.params).all() and np.isfinite(res_ind.bse).all()
    return res_ind, "Independence (fallback)", ok


def results_to_rows(res, cov_used: str, converged: bool, judge: str, model_label: str) -> pd.DataFrame:
    ci = res.conf_int()
    ci.columns = ["ci_low", "ci_high"]
    out = pd.DataFrame({
        "coef": res.params,
        "ci_low": ci["ci_low"],
        "ci_high": ci["ci_high"],
        "p": res.pvalues,
    })
    out["OR"] = np.exp(out["coef"])
    out["OR_ci_low"] = np.exp(out["ci_low"])
    out["OR_ci_high"] = np.exp(out["ci_high"])
    out = out.reset_index().rename(columns={"index": "term"})
    out["term_type"] = out["term"].apply(term_type)
    out["judge"] = judge
    out["model"] = model_label
    out["cov_struct"] = cov_used
    out["converged"] = converged
    return out[["judge", "model", "term_type", "term", "OR", "OR_ci_low", "OR_ci_high",
                "p", "cov_struct", "converged"]]


def run_gee_for_judge(judge: str, grid: pd.DataFrame) -> pd.DataFrame:
    all_rows = []

    print("  pooled model...")
    res, cov_used, ok = fit_gee_with_fallback(POOLED_FORMULA, grid)
    print(f"    cov_struct={cov_used}, converged={ok}")
    all_rows.append(results_to_rows(res, cov_used, ok, judge, "ALL"))

    for v in VICTIMS:
        print(f"  victim={v} model...")
        sub = grid[grid.victim == v]
        res, cov_used, ok = fit_gee_with_fallback(VICTIM_FORMULA, sub)
        print(f"    cov_struct={cov_used}, converged={ok}")
        all_rows.append(results_to_rows(res, cov_used, ok, judge, v))

    return pd.concat(all_rows, ignore_index=True)


def run_gee_part(judge: str, grid: pd.DataFrame) -> pd.DataFrame:
    table = run_gee_for_judge(judge, grid)
    path = out_path(GEE_TEMPLATE, judge)
    table.to_csv(path, index=False)
    print(f"  [gee] wrote {path} ({len(table)} rows)")

    level_rows = table[(table.model == "ALL") & (table.term_type == "level")]
    print("  pooled level effects (OR vs neutral):")
    print(level_rows[["term", "OR", "OR_ci_low", "OR_ci_high", "p"]].to_string(index=False))
    print()
    return table


# =============================================================================
# Step 3: risk differences with cluster-bootstrap CIs
# =============================================================================
def risk_differences(grid: pd.DataFrame) -> pd.DataFrame:
    prompts = np.sort(grid.base_prompt.unique())
    p_index = {p: i for i, p in enumerate(prompts)}
    P = len(prompts)

    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, P, size=(N_BOOT, P))
    W = np.zeros((N_BOOT, P))
    for b in range(N_BOOT):
        W[b] = np.bincount(idx[b], minlength=P)

    scopes = [("ALL", grid)] + [(v, grid[grid.victim == v]) for v in VICTIMS]
    rows = []
    for label, sub in scopes:
        cnt = np.zeros((P, len(LEVELS)))
        tot = np.zeros((P, len(LEVELS)))
        pi = sub.base_prompt.map(p_index).to_numpy()
        li = sub.level.map({l: j for j, l in enumerate(LEVELS)}).to_numpy()
        np.add.at(cnt, (pi, li), sub.compliant.to_numpy())
        np.add.at(tot, (pi, li), 1)

        obs_rate = cnt.sum(0) / tot.sum(0)
        boot_rate = (W @ cnt) / (W @ tot)
        j0 = LEVELS.index(0)
        for j, lvl in enumerate(LEVELS):
            if lvl == 0:
                continue
            rd = 100 * (obs_rate[j] - obs_rate[j0])
            boot_rd = 100 * (boot_rate[:, j] - boot_rate[:, j0])
            lo, hi = np.percentile(boot_rd, [2.5, 97.5])
            rows.append({"level": lvl, "victim": label,
                         "neutral_rate_pct": 100 * obs_rate[j0],
                         "framed_rate_pct": 100 * obs_rate[j],
                         "rd_pp": rd, "ci_low_pp": lo, "ci_high_pp": hi})
    return pd.DataFrame(rows)


# =============================================================================
# Step 5: polarity x magnitude GEE (framed responses only)
#   compliant ~ C(polarity) * magnitude_c + C(victim) + C(category) + C(variation_set)
#   magnitude_c = magnitude - 3 (centred), so the polarity coefficient is the
#   positive-vs-negative OR at the middle magnitude; the OR at each magnitude
#   1..5 is also reported via linear contrasts.
# =============================================================================
def polarity_model(grid: pd.DataFrame) -> pd.DataFrame:
    d = grid[grid.level != 0].copy()
    d["polarity"] = np.where(d.level > 0, "positive", "negative")
    d["magnitude_c"] = d.level.abs() - 3
    formula = ("compliant ~ C(polarity, Treatment('negative')) * magnitude_c "
               "+ C(victim) + C(category) + C(variation_set)")
    res, cov_used, ok = fit_gee_with_fallback(formula, d)

    pol = "C(polarity, Treatment('negative'))[T.positive]"
    mag = "magnitude_c"
    inter = "C(polarity, Treatment('negative'))[T.positive]:magnitude_c"
    cov = res.cov_params()
    rows = []

    def add(name, coef, se, p):
        rows.append({"quantity": name, "OR": np.exp(coef),
                     "OR_ci_low": np.exp(coef - 1.96 * se),
                     "OR_ci_high": np.exp(coef + 1.96 * se), "p": p,
                     "cov_struct": cov_used, "converged": ok})

    add("polarity (positive vs negative, at magnitude 3)", res.params[pol], res.bse[pol], res.pvalues[pol])
    add("magnitude (per +1 step, negative framings)", res.params[mag], res.bse[mag], res.pvalues[mag])
    add("interaction (extra per-step change for positive)", res.params[inter], res.bse[inter], res.pvalues[inter])
    add("magnitude (per +1 step, positive framings)",
        res.params[mag] + res.params[inter],
        float(np.sqrt(cov.loc[mag, mag] + cov.loc[inter, inter] + 2 * cov.loc[mag, inter])),
        float(res.t_test(f"{mag} + {inter} = 0").pvalue))
    for m in range(1, 6):
        c = m - 3
        coef = res.params[pol] + c * res.params[inter]
        var = cov.loc[pol, pol] + c ** 2 * cov.loc[inter, inter] + 2 * c * cov.loc[pol, inter]
        p = float(res.t_test(f"{pol} + {c}*{inter} = 0").pvalue)
        add(f"polarity OR (positive vs negative) at magnitude {m}", coef, float(np.sqrt(var)), p)
    return pd.DataFrame(rows)


# =============================================================================
# Step 6: category x compliant chi-square (+ category ORs from step-4 GEE)
# =============================================================================
def category_chi2(grid: pd.DataFrame) -> pd.DataFrame:
    table = pd.crosstab(grid.category, grid.compliant)
    chi2, p, dof, _ = chi2_contingency(table)
    n = table.to_numpy().sum()
    cramers_v = np.sqrt(chi2 / (n * (min(table.shape) - 1)))
    out = table.copy()
    out.columns = ["non_compliant", "compliant"]
    out["compliance_rate_pct"] = 100 * out["compliant"] / (out["compliant"] + out["non_compliant"])
    out = out.reset_index()
    out["chi2"], out["dof"], out["p"], out["cramers_v"] = chi2, dof, p, cramers_v
    return out


def run_remaining_part(judge: str, grid: pd.DataFrame, gee_table: pd.DataFrame = None) -> None:
    # Step 3
    rd = risk_differences(grid)
    path = out_path(RD_TEMPLATE, judge)
    rd.to_csv(path, index=False)
    print(f"  [step 3] wrote {path}")
    print(rd[rd.victim == "ALL"].round(2).to_string(index=False))

    # Step 5
    pol = polarity_model(grid)
    path = out_path(POLARITY_TEMPLATE, judge)
    pol.to_csv(path, index=False)
    print(f"  [step 5] wrote {path}")
    print(pol.round(4).to_string(index=False))

    # Step 6
    chi = category_chi2(grid)
    path = out_path(CHI2_TEMPLATE, judge)
    chi.to_csv(path, index=False)
    print(f"  [step 6] wrote {path}")
    print(chi.round(3).to_string(index=False))

    # Category / victim ORs from the step-4 pooled GEE: use the table computed in
    # this run if available, otherwise read it from the output folder.
    if gee_table is None:
        try:
            gee_table = pd.read_csv(out_path(GEE_TEMPLATE, judge))
        except FileNotFoundError:
            print("  (gee csv not found in output/; run the 'gee' part first for category ORs)")
    if gee_table is not None:
        cat = gee_table[(gee_table.model == "ALL") & (gee_table.term_type.isin(["category", "victim"]))]
        path = out_path(CATEGORY_OR_TEMPLATE, judge)
        cat.to_csv(path, index=False)
        print(f"  [step 6] wrote {path}")
    print()


# =============================================================================
# Main
# =============================================================================
PARTS = ("mcnemar", "gee", "remaining")


def main():
    requested = [a.lower() for a in sys.argv[1:]] or ["all"]
    if "all" in requested:
        parts = list(PARTS)
    else:
        unknown = [a for a in requested if a not in PARTS]
        if unknown:
            sys.exit(f"Unknown part(s): {unknown}. Choose from {list(PARTS)} or 'all'.")
        parts = [p for p in PARTS if p in requested]  # keep dependency order: gee before remaining

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    master_by_vs = load_master_prompts()

    for judge in JUDGES:
        print(f"=== judge={judge} ===")
        print(f"Building grid for judge={judge} ...")
        grid = get_grid(judge, master_by_vs)

        if "mcnemar" in parts:
            run_mcnemar_part(judge, grid)

        gee_table = None
        if "gee" in parts:
            gee_table = run_gee_part(judge, grid)

        if "remaining" in parts:
            run_remaining_part(judge, grid, gee_table)


if __name__ == "__main__":
    main()