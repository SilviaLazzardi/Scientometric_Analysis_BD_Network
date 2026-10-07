"""
Reusable functions for the BD-Net scientometric analysis.

This module contains only the analysis used in the main manuscript:
1. Data loading and preprocessing
2. Descriptive publication counts / Table 1
3. Interrupted time-series analysis / Figure 1
4. Matched post-2009 publication-growth analysis / Figure 2
5. Annualized fractional author productivity / Figure 3
6. BD-Net and yearly European co-authorship network analyses / Figure 4 and Table 2

Supplementary and superseded analyses have intentionally been removed.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pycountry
import statsmodels.api as sm
import statsmodels.formula.api as smf
from networkx.algorithms.community import greedy_modularity_communities
from scipy.stats import mannwhitneyu, wilcoxon
from statsmodels.stats.multitest import multipletests




def _cliffs_delta(x, y):
    """Return Cliff's delta (x relative to y) and conventional magnitude label."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    delta = (np.sum(x[:, None] > y[None, :]) - np.sum(x[:, None] < y[None, :])) / (len(x) * len(y))
    ad = abs(delta)
    if ad < 0.147:
        magnitude = "negligible"
    elif ad < 0.33:
        magnitude = "small"
    elif ad < 0.474:
        magnitude = "medium"
    else:
        magnitude = "large"
    return float(delta), magnitude

# =============================================================================
# 1. DATA LOADING AND PREPROCESSING
# =============================================================================


def load_author_registry(path):
    """Load and validate the BD-Net Scopus author-ID registry."""
    required = {"Cognome", "Scopus Author ID", "First publication year"}
    registry = pd.read_excel(path, dtype={"Scopus Author ID": "string"})

    missing = required.difference(registry.columns)
    if missing:
        raise ValueError(f"Author registry is missing columns: {sorted(missing)}")

    registry = registry.copy()
    registry["Scopus Author ID"] = registry["Scopus Author ID"].str.strip()

    if registry["Scopus Author ID"].isna().any() or registry["Cognome"].isna().any():
        raise ValueError("Author registry contains blank IDs or surnames")
    if not registry["Scopus Author ID"].str.fullmatch(r"\d+").all():
        raise ValueError("Scopus Author ID values must contain digits only")
    if registry["Scopus Author ID"].duplicated().any():
        raise ValueError("Author registry contains duplicate Scopus Author IDs")

    publication_year = pd.to_numeric(
        registry["First publication year"], errors="coerce"
    )
    if publication_year.isna().any():
        raise ValueError("Author registry contains invalid first-publication years")
    if not publication_year.eq(publication_year.astype(int)).all():
        raise ValueError("First-publication years must be whole numbers")

    registry["First publication year"] = publication_year.astype(int)
    return registry


def split_scopus_ids(value):
    """Convert a semicolon-delimited Scopus ID field to a list of strings."""
    if pd.isna(value):
        return []
    return [token.strip() for token in str(value).split(";") if token.strip()]


def _is_country(name):
    try:
        pycountry.countries.lookup(name)
        return True
    except LookupError:
        return False


def _extract_affiliation_countries(value):
    """Extract terminal country names from the Scopus Affiliations field."""
    if pd.isna(value):
        return []

    countries = []
    for affiliation in str(value).split(";"):
        country = affiliation.split(",")[-1].strip()
        if not country:
            continue
        if _is_country(country) or country == "Turkey":
            countries.append(country)
    return countries


def return_merged_file(path):
    """Read all CSV exports in a directory and deduplicate records by Title."""
    path = Path(path)
    files = sorted(path.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {path}")

    frames = [
        pd.read_csv(file, dtype={"Author(s) ID": "string"})
        for file in files
    ]
    merged = pd.concat(frames, ignore_index=True)
    return merged.drop_duplicates(subset=["Title"]).reset_index(drop=True)


def preprocess_merged_data(merged_unique, author_ids, european_countries):
    """
    Add the fields used by the analysis:
      Country      list of affiliation countries
      Author_IDs   list of Scopus author IDs
      BD-Authors   registered BD-Net IDs on the paper, or 0 if absent
      Binary_EU    1 if >=1 affiliation is in the predefined European set
      n_authors    number of Scopus author IDs on the paper
    """
    df = merged_unique.copy()

    if "Author(s) ID" not in df.columns:
        raise ValueError("Scopus export must contain an 'Author(s) ID' column")
    if "Affiliations" not in df.columns:
        raise ValueError("Scopus export must contain an 'Affiliations' column")

    df["Country"] = df["Affiliations"].apply(_extract_affiliation_countries)
    df["Author_IDs"] = df["Author(s) ID"].apply(split_scopus_ids)

    bd_set = {str(a).strip() for a in author_ids}
    df["BD-Authors"] = df["Author_IDs"].apply(
        lambda ids: [a for a in ids if a in bd_set] or 0
    )

    europe = set(european_countries)
    df["Binary_EU"] = df["Country"].apply(
        lambda countries: int(bool(set(countries) & europe))
    )

    df = df[df["Authors"].notna()].copy()
    df["n_authors"] = df["Author_IDs"].map(len)
    return df


def preprocess_and_save(path, path_save, filename, author_ids, european_countries):
    """Load, preprocess, save, and return one Scopus publication dataset."""
    output_dir = Path(path_save)
    output_dir.mkdir(parents=True, exist_ok=True)

    merged = return_merged_file(path)
    processed = preprocess_merged_data(merged, author_ids, european_countries)
    processed.to_csv(output_dir / filename, index=False)
    return processed


def european_subset(df):
    """Return publications with at least one affiliation in the predefined European set."""
    return df[df["Binary_EU"] == 1].copy()


# =============================================================================
# 2. DESCRIPTIVE PUBLICATION OUTPUT / TABLE 1
# =============================================================================


def unique_author_ids(df, authors_col="Author_IDs"):
    """Return the set of unique Scopus Author IDs represented in a dataset."""
    authors = set()
    for ids in df[authors_col].dropna():
        if isinstance(ids, list):
            authors.update(str(a).strip() for a in ids if str(a).strip())
    return authors


def _pct_change(pre, post):
    return ((post / pre) - 1.0) * 100 if pre else np.nan


def build_table1(df_pre, df_post, df_pre_eu, df_post_eu, bd_ids):
    """Build the manuscript Table 1 publication/author summary."""
    bd_set = {str(a).strip() for a in bd_ids}

    pre_world_authors = unique_author_ids(df_pre)
    post_world_authors = unique_author_ids(df_post)
    pre_eu_authors = unique_author_ids(df_pre_eu)
    post_eu_authors = unique_author_ids(df_post_eu)

    pre_eu_bd = int((df_pre_eu["BD-Authors"] != 0).sum())
    post_eu_bd = int((df_post_eu["BD-Authors"] != 0).sum())
    pre_eu_nonbd = int((df_pre_eu["BD-Authors"] == 0).sum())
    post_eu_nonbd = int((df_post_eu["BD-Authors"] == 0).sum())

    pre_members = len(pre_eu_authors & bd_set)
    post_members = len(post_eu_authors & bd_set)
    n_registry = len(bd_set)

    rows = [
        ("Worldwide publications", len(df_pre), len(df_post), True),
        ("Worldwide unique authors", len(pre_world_authors), len(post_world_authors), True),
        ("European publications", len(df_pre_eu), len(df_post_eu), True),
        ("European unique authors", len(pre_eu_authors), len(post_eu_authors), True),
        ("European publications involving ≥1 BD Network member", pre_eu_bd, post_eu_bd, True),
        ("European publications without BD Network members", pre_eu_nonbd, post_eu_nonbd, True),
    ]

    table = pd.DataFrame(
        [
            {
                "Measure": label,
                "1992–2008": pre,
                "2009–2025": post,
                "Change": f"{_pct_change(pre, post):+.1f}%" if change else "—",
            }
            for label, pre, post, change in rows
        ]
    )

    table.loc[len(table)] = {
        "Measure": "Registered BD Network members represented in the European dataset",
        "1992–2008": f"{pre_members} / {n_registry}",
        "2009–2025": f"{post_members} / {n_registry}",
        "Change": "—",
    }
    return table


# =============================================================================
# 3. INTERRUPTED TIME-SERIES / FIGURE 1
# =============================================================================


def annual_publication_counts(df, start_year=1992, end_year=2025):
    """Count publications per calendar year, filling missing years with zero."""
    return (
        df.groupby("Year")
        .size()
        .reindex(range(start_year, end_year + 1), fill_value=0)
        .rename("Publications")
        .reset_index()
    )


def fit_segmented_model(df_yearly, intervention_year=2009):
    """Fit Publications ~ Time * Post with Time centered on the intervention year."""
    data = df_yearly.copy()
    data["Time"] = data["Year"] - intervention_year
    data["Post"] = (data["Year"] >= intervention_year).astype(int)

    model = smf.ols("Publications ~ Time * Post", data=data).fit()
    data["Fitted"] = model.predict(data)
    return data, model


def segmented_model_summary(model):
    """Return interpretable pre/post slopes, level change, p-values, and R²."""
    pre_slope = model.params["Time"]
    slope_change = model.params["Time:Post"]
    post_slope = pre_slope + slope_change

    return {
        "pre_slope": pre_slope,
        "post_slope": post_slope,
        "slope_change": slope_change,
        "slope_change_p": model.pvalues["Time:Post"],
        "level_change": model.params["Post"],
        "level_change_p": model.pvalues["Post"],
        "r2": model.rsquared,
    }


def model_coefficient_table(model):
    """Return coefficient, SE, p-value and 95% CI from a statsmodels result."""
    ci = model.conf_int()
    return pd.DataFrame(
        {
            "Coefficient": model.params,
            "SE": model.bse,
            "p": model.pvalues,
            "CI_low": ci[0],
            "CI_high": ci[1],
        }
    )


def plot_segmented_publications(
    ax,
    data,
    stats,
    title,
    color="tab:blue",
    intervention_year=2009,
    tick_size=15,
    label_size=17,
    legend_size=10,
):
    """Plot observed annual counts and pre/post segmented fitted trajectories."""
    pre = data[data["Year"] < intervention_year]
    post = data[data["Year"] >= intervention_year]

    ax.scatter(
        data["Year"], data["Publications"],
        color=color, s=45, alpha=0.75, label="Observed", zorder=3,
    )
    ax.plot(
        pre["Year"], pre["Fitted"],
        color=color, linewidth=3, linestyle="-",
        label=f"Pre-2009 fit ({stats['pre_slope']:.1f} publications/year)",
    )
    ax.plot(
        post["Year"], post["Fitted"],
        color=color, linewidth=3, linestyle="--",
        label=f"Post-2009 fit ({stats['post_slope']:.1f} publications/year)",
    )
    ax.axvline(intervention_year, color="black", linewidth=1.5, linestyle=":", alpha=0.7)

    y_max = ax.get_ylim()[1]
    ax.text(
        intervention_year + 0.3, y_max * 0.75,
        "BD Network\nestablished", fontsize=10, va="top",
    )

    ax.set_title(title, fontsize=label_size)
    ax.set_xlabel("Year", fontsize=label_size)
    ax.set_ylabel("Number of publications", fontsize=label_size)
    ax.tick_params(axis="both", which="major", labelsize=tick_size)
    ax.set_xticks(np.arange(1992, 2026, 4))
    ax.legend(loc="upper left", frameon=False, fontsize=legend_size)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


# =============================================================================
# 4. MATCHED POST-2009 PUBLICATION GROWTH / FIGURE 2
# =============================================================================


def author_publication_counts(df, authors_col="Author_IDs"):
    """Count the number of unique publication rows involving each author."""
    counts = Counter()
    for authors in df[authors_col].dropna():
        if not isinstance(authors, list):
            continue
        for author in set(authors):
            counts[str(author).strip()] += 1
    return counts


def productivity_bin(n):
    """Pre-2009 productivity stratum used for matched randomization."""
    if n <= 2:
        return "1-2"
    if n <= 5:
        return "3-5"
    if n <= 10:
        return "6-10"
    if n <= 20:
        return "11-20"
    if n <= 40:
        return "21-40"
    return "41+"


def annual_growth_rate(counts, years):
    """Estimate APC from a log-linear Poisson model of annual publication counts."""
    counts = np.asarray(counts, dtype=float)
    time = np.asarray(years) - np.asarray(years)[0]
    X = sm.add_constant(time)
    model = sm.GLM(counts, X, family=sm.families.Poisson()).fit()
    beta = model.params[1]
    return (np.exp(beta) - 1.0) * 100.0


def run_matched_growth_analysis(
    df_pre_eu,
    df_post_eu,
    bd_ids,
    n_simulations=10_000,
    seed=42,
    start_year=2009,
    end_year=2025,
):
    """
    Match pre-2009-active BD-Net members to non-Network authors by pre-period
    productivity strata, then compare post-2009 APC with an empirical randomization
    distribution.
    """
    rng = np.random.default_rng(seed)
    years = np.arange(start_year, end_year + 1)
    bd_set = {str(a).strip() for a in bd_ids}

    pre_counts = author_publication_counts(df_pre_eu)
    pre_authors = set(pre_counts)
    bd_pre_active = bd_set & pre_authors
    control_pool = pre_authors - bd_set

    bd_profile = Counter(productivity_bin(pre_counts[a]) for a in bd_pre_active)
    control_bins = {}
    for author in sorted(control_pool):
        control_bins.setdefault(productivity_bin(pre_counts[author]), []).append(author)

    for bin_name, n_needed in bd_profile.items():
        n_available = len(control_bins.get(bin_name, []))
        if n_available < n_needed:
            raise ValueError(
                f"Not enough controls in productivity bin {bin_name}: "
                f"need {n_needed}, found {n_available}"
            )

    post = df_post_eu.reset_index(drop=True)
    papers_by_year_author = {year: {} for year in years}
    for idx, row in post.iterrows():
        year = int(row["Year"])
        if year not in papers_by_year_author:
            continue
        authors = row["Author_IDs"]
        if not isinstance(authors, list):
            continue
        for author in set(str(a).strip() for a in authors if str(a).strip()):
            papers_by_year_author[year].setdefault(author, set()).add(idx)

    def annual_group_counts(selected_authors):
        counts = []
        for year in years:
            papers = set()
            lookup = papers_by_year_author[year]
            for author in selected_authors:
                papers.update(lookup.get(author, set()))
            counts.append(len(papers))
        return np.asarray(counts, dtype=float)

    def sample_matched_group():
        sampled = []
        for bin_name, n_needed in sorted(bd_profile.items()):
            candidates = np.asarray(sorted(control_bins[bin_name]), dtype=object)
            sampled.extend(rng.choice(candidates, size=n_needed, replace=False).tolist())
        return set(sampled)

    bd_annual_counts = annual_group_counts(bd_pre_active)
    bd_growth_rate = annual_growth_rate(bd_annual_counts, years)
    bd_total_output = bd_annual_counts.sum()

    random_annual_counts = np.zeros((n_simulations, len(years)), dtype=float)
    random_growth_rates = np.zeros(n_simulations, dtype=float)
    random_total_outputs = np.zeros(n_simulations, dtype=float)

    for i in range(n_simulations):
        random_group = sample_matched_group()
        counts = annual_group_counts(random_group)
        random_annual_counts[i] = counts
        random_total_outputs[i] = counts.sum()
        random_growth_rates[i] = annual_growth_rate(counts, years)

    p_upper = (1 + np.sum(random_growth_rates >= bd_growth_rate)) / (n_simulations + 1)
    p_lower = (1 + np.sum(random_growth_rates <= bd_growth_rate)) / (n_simulations + 1)
    p_two_sided = min(1.0, 2 * min(p_upper, p_lower))
    p_output = (1 + np.sum(random_total_outputs >= bd_total_output)) / (n_simulations + 1)

    return {
        "years": years,
        "n_simulations": n_simulations,
        "bd_pre_active": bd_pre_active,
        "control_pool_size": len(control_pool),
        "bd_profile": bd_profile,
        "bd_annual_counts": bd_annual_counts,
        "bd_growth_rate": bd_growth_rate,
        "bd_total_output": bd_total_output,
        "random_annual_counts": random_annual_counts,
        "random_growth_rates": random_growth_rates,
        "random_total_outputs": random_total_outputs,
        "matched_growth_median": float(np.median(random_growth_rates)),
        "matched_growth_interval": np.percentile(random_growth_rates, [2.5, 97.5]),
        "matched_total_median": float(np.median(random_total_outputs)),
        "growth_difference": float(bd_growth_rate - np.median(random_growth_rates)),
        "percentile_rank": float(np.mean(random_growth_rates < bd_growth_rate) * 100),
        "p_growth_greater": float(p_upper),
        "p_growth_two_sided": float(p_two_sided),
        "p_output_greater": float(p_output),
    }


def plot_matched_growth(
    result,
    figsize=(16, 6.5),
    tick_size=18,
    label_size=19,
    legend_size=15,
    panel_size=20,
    bd_color="orange",
    matched_color="#4C72B0",
):
    """Create Figure 2: indexed trajectories (A) and APC distribution (B)."""
    years = result["years"]
    bd_counts = result["bd_annual_counts"]
    random_counts = result["random_annual_counts"]
    random_rates = result["random_growth_rates"]
    bd_rate = result["bd_growth_rate"]

    valid = random_counts[:, 0] > 0
    random_valid = random_counts[valid]
    random_indexed = random_valid / random_valid[:, [0]] * 100
    bd_indexed = bd_counts / bd_counts[0] * 100

    random_median = np.median(random_indexed, axis=0)
    random_low = np.percentile(random_indexed, 2.5, axis=0)
    random_high = np.percentile(random_indexed, 97.5, axis=0)

    growth_median = np.median(random_rates)
    growth_low, growth_high = np.percentile(random_rates, [2.5, 97.5])

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    ax_a.fill_between(
        years, random_low, random_high,
        color=matched_color, alpha=0.18, linewidth=0,
        label="95% matched-group interval",
    )
    ax_a.plot(
        years, random_median,
        color=matched_color, linestyle="--", linewidth=3,
        label="Median matched group",
    )
    ax_a.plot(
        years, bd_indexed,
        color=bd_color, marker="o", markersize=6, linewidth=3.5,
        label="BD Network",
    )
    ax_a.axhline(100, color="black", linestyle=":", linewidth=1.5, alpha=0.6)
    ax_a.set_xlabel("Year", fontsize=label_size)
    ax_a.set_ylabel("Publication output index\n(2009 = 100)", fontsize=label_size)
    ax_a.set_xticks([2009, 2013, 2017, 2021, 2025])
    ax_a.tick_params(axis="both", labelsize=tick_size, width=1.2, length=6)
    ax_a.legend(frameon=False, fontsize=legend_size, loc="upper left")
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    ax_a.text(-0.11, 1.04, "A", transform=ax_a.transAxes,
              fontsize=panel_size, fontweight="bold", va="top")

    ax_b.hist(
        random_rates, bins=40,
        color=matched_color, alpha=0.75, edgecolor="white", linewidth=0.7,
    )
    ax_b.axvspan(
        growth_low, growth_high,
        color=matched_color, alpha=0.12,
        label="95% matched-group interval",
    )
    ax_b.axvline(
        growth_median, color=matched_color, linestyle="--", linewidth=3,
        label=f"Matched median ({growth_median:.2f}%)",
    )
    ax_b.axvline(
        bd_rate, color=bd_color, linewidth=3.5,
        label=f"BD Network ({bd_rate:.2f}%)",
    )
    ax_b.set_xlabel("Annual percentage change in publication output (%)", fontsize=label_size)
    ax_b.set_ylabel("Number of matched groups", fontsize=label_size)
    ax_b.tick_params(axis="both", labelsize=tick_size, width=1.2, length=6)
    ax_b.legend(frameon=False, fontsize=legend_size, loc="upper left")
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    ax_b.text(-0.11, 1.04, "B", transform=ax_b.transAxes,
              fontsize=panel_size, fontweight="bold", va="top")

    return fig, (ax_a, ax_b), int(valid.sum())


# =============================================================================
# 5. ANNUALIZED FRACTIONAL AUTHOR PRODUCTIVITY / FIGURE 3
# =============================================================================


def _parse_author_list(value):
    if not isinstance(value, list):
        return []
    return list(dict.fromkeys(str(a).strip() for a in value if str(a).strip()))


def fractional_output_by_author(df, bd_ids, authors_col="Author_IDs"):
    """Sum 1/n_authors fractional publication contributions for each BD-Net author."""
    bd_set = {str(a).strip() for a in bd_ids}
    output = {author: 0.0 for author in bd_set}

    for authors in df[authors_col].dropna():
        authors = _parse_author_list(authors)
        if not authors:
            continue
        contribution = 1.0 / len(authors)
        for author in set(authors) & bd_set:
            output[author] += contribution
    return output


def build_period_afp(
    df,
    bd_ids,
    first_publication_year,
    period_label,
    start_year,
    end_year,
    min_observable_years=5,
):
    """Build author-level Annualized Fractional Publication Rate (AFP) for one period."""
    bd_set = {str(a).strip() for a in bd_ids}
    first_year = {
        str(k).strip(): int(v)
        for k, v in first_publication_year.items()
        if pd.notna(v)
    }
    fractional = fractional_output_by_author(df, bd_set)

    rows = []
    for author in sorted(bd_set):
        if author not in first_year:
            continue

        observable_start = max(start_year, first_year[author])
        observable_years = max(0, end_year - observable_start + 1)
        frac = fractional.get(author, 0.0)
        afp = frac / observable_years if observable_years > 0 else np.nan

        rows.append(
            {
                "Author_ID": author,
                "Period": period_label,
                "FirstPublicationYear": first_year[author],
                "ObservableYears": observable_years,
                "FractionalOutput": frac,
                "AFP": afp,
                "Eligible": observable_years >= min_observable_years,
            }
        )
    return pd.DataFrame(rows)


def run_afp_analysis(
    df_pre,
    df_post,
    bd_ids,
    first_publication_year,
    min_observable_years=5,
):
    """Run the paired pre/post AFP analysis and two-sided Wilcoxon signed-rank test."""
    pre = build_period_afp(
        df_pre, bd_ids, first_publication_year,
        "1992–2008", 1992, 2008, min_observable_years,
    )
    post = build_period_afp(
        df_post, bd_ids, first_publication_year,
        "2009–2025", 2009, 2025, min_observable_years,
    )

    paired = (
        pre[["Author_ID", "ObservableYears", "FractionalOutput", "AFP", "Eligible"]]
        .rename(columns={
            "ObservableYears": "ObservableYears_pre",
            "FractionalOutput": "FractionalOutput_pre",
            "AFP": "AFP_pre",
            "Eligible": "Eligible_pre",
        })
        .merge(
            post[["Author_ID", "ObservableYears", "FractionalOutput", "AFP", "Eligible"]]
            .rename(columns={
                "ObservableYears": "ObservableYears_post",
                "FractionalOutput": "FractionalOutput_post",
                "AFP": "AFP_post",
                "Eligible": "Eligible_post",
            }),
            on="Author_ID",
            how="inner",
        )
    )

    paired = paired[paired["Eligible_pre"] & paired["Eligible_post"]].copy()
    paired = paired.sort_values("AFP_post", ascending=False).reset_index(drop=True)
    paired["AnonID"] = [f"A{i:02d}" for i in range(1, len(paired) + 1)]

    pre_values = paired["AFP_pre"].to_numpy(float)
    post_values = paired["AFP_post"].to_numpy(float)

    wilcox = wilcoxon(
        pre_values, post_values,
        alternative="two-sided", zero_method="wilcox", method="auto",
    )

    def summary(values):
        return {
            "median": float(np.median(values)),
            "q1": float(np.percentile(values, 25)),
            "q3": float(np.percentile(values, 75)),
        }

    return {
        "paired": paired,
        "pre_summary": summary(pre_values),
        "post_summary": summary(post_values),
        "W": float(wilcox.statistic),
        "p": float(wilcox.pvalue),
        "n": len(paired),
        "n_increased": int(np.sum(post_values > pre_values)),
    }


def plot_afp_productivity(
    result,
    figsize=(7, 5.5),
    tick_size=17,
    label_size=17,
    legend_size=15,
    seed=42,
):
    """Create Figure 3: anonymized paired AFP observations with boxplots."""
    paired = result["paired"]
    pre_summary = result["pre_summary"]
    post_summary = result["post_summary"]

    rng = np.random.default_rng(seed)
    x_pre = rng.normal(0, 0.03, len(paired))
    x_post = 1 + rng.normal(0, 0.03, len(paired))

    fig, ax = plt.subplots(figsize=figsize)

    for i, row in paired.iterrows():
        ax.plot(
            [x_pre[i], x_post[i]], [row["AFP_pre"], row["AFP_post"]],
            linewidth=1, alpha=0.30, color="gray", zorder=1,
        )

    ax.boxplot(
        [paired["AFP_pre"], paired["AFP_post"]],
        positions=[0, 1], widths=0.22, showfliers=False, patch_artist=False,
        medianprops=dict(linewidth=2), boxprops=dict(linewidth=1.5),
        whiskerprops=dict(linewidth=1.2), capprops=dict(linewidth=1.2),
    )

    ax.scatter(
        x_pre, paired["AFP_pre"], s=45, alpha=0.9,
        label=f"1992–2008 (n = {len(paired)})", zorder=3,
    )
    ax.scatter(
        x_post, paired["AFP_post"], s=45, alpha=0.9,
        label=f"2009–2025 (n = {len(paired)})", zorder=3,
    )

    ax.text(0.15, pre_summary["median"],
            f"Median = {pre_summary['median']:.3f}", fontsize=13, va="bottom")
    ax.text(1.15, post_summary["median"],
            f"Median = {post_summary['median']:.3f}", fontsize=13, va="bottom")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["1992–2008", "2009–2025"], fontsize=tick_size)
    ax.tick_params(axis="y", labelsize=tick_size)
    ax.set_ylabel("Annualized fractional publication rate", fontsize=label_size)
    ax.set_title("Distribution of BD Network member productivity", fontsize=label_size)
    ax.legend(frameon=False, loc="upper left", fontsize=legend_size)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig, ax


# =============================================================================
# 6. CO-AUTHORSHIP NETWORK ANALYSIS / FIGURE 4 / TABLE 2
# =============================================================================


def _bd_collaborations(df, author_list):
    """Return registered BD-Net IDs on each paper when at least two are present."""
    bd_set = {str(a).strip() for a in author_list}

    if "BD-Authors" in df.columns:
        matched = df["BD-Authors"]
    else:
        matched = df["Author_IDs"].apply(
            lambda ids: [a for a in ids if a in bd_set] if isinstance(ids, list) else []
        )

    return matched.apply(
        lambda ids: ids if isinstance(ids, list) and len(ids) > 1 else 0
    ).to_numpy()


def coauthor_matrix(df, all_authors, author_list):
    """Build a symmetric BD-Net member × member co-authorship count matrix."""
    all_authors = [str(a).strip() for a in all_authors]
    M = pd.DataFrame(0, index=all_authors, columns=all_authors, dtype=int)

    for entry in _bd_collaborations(df, author_list):
        if not isinstance(entry, list):
            continue
        authors = list(dict.fromkeys(str(a).strip() for a in entry if str(a).strip()))
        for a, b in combinations(authors, 2):
            if a in M.index and b in M.index:
                M.loc[a, b] += 1
                M.loc[b, a] += 1
    return M


def aggregated_network_stats(G):
    """Descriptive statistics for an aggregated BD-Net co-authorship graph."""
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    if n_nodes == 0:
        return {
            "Nodes": 0,
            "Edges": 0,
            "Density": np.nan,
            "Average_degree": np.nan,
            "Average_weighted_degree": np.nan,
            "Weighted_clustering": np.nan,
            "Components": 0,
            "Largest_component": 0,
            "Average_shortest_path_largest_component": np.nan,
        }

    degrees = dict(G.degree())
    weighted_degrees = dict(G.degree(weight="weight"))
    components = list(nx.connected_components(G))
    largest_nodes = max(components, key=len)
    largest_graph = G.subgraph(largest_nodes).copy()
    largest_size = largest_graph.number_of_nodes()

    return {
        "Nodes": n_nodes,
        "Edges": n_edges,
        "Density": nx.density(G),
        "Average_degree": sum(degrees.values()) / n_nodes,
        "Average_weighted_degree": sum(weighted_degrees.values()) / n_nodes,
        "Weighted_clustering": nx.average_clustering(G, weight="weight"),
        "Components": len(components),
        "Largest_component": largest_size,
        "Average_shortest_path_largest_component": (
            nx.average_shortest_path_length(largest_graph) if largest_size > 1 else 0.0
        ),
    }


def plot_coauthor_network_clustered(
    M,
    min_weight=1,
    figsize=(24, 14),
    seed=42,
    cluster_radius=10.0,
    internal_scale=20.0,
    label_map=None,
    font_size=18,
    ax=None,
    title=None,
    title_font_size=24,
    edge_alpha=0.10,
):
    """Plot the non-isolated portion of an aggregated BD-Net network by community."""
    A = M.copy()
    A = A.mask(np.eye(len(A), dtype=bool), 0)
    G = nx.from_pandas_adjacency(A)

    G.remove_edges_from(
        [(u, v) for u, v, d in G.edges(data=True) if d.get("weight", 0) < min_weight]
    )
    G.remove_nodes_from(list(nx.isolates(G)))

    if G.number_of_nodes() == 0:
        return G, []

    communities = list(greedy_modularity_communities(G, weight="weight"))
    node_to_comm = {
        node: i for i, community in enumerate(communities) for node in community
    }

    angles = np.linspace(0, 2 * np.pi, len(communities), endpoint=False)
    pos = {}

    for i, community in enumerate(communities):
        sub = G.subgraph(community)
        sub_pos = nx.spring_layout(
            sub,
            seed=seed,
            k=2.5 / np.sqrt(max(sub.number_of_nodes(), 2)),
            iterations=200,
            weight="weight",
        )
        cx = cluster_radius * np.cos(angles[i])
        cy = cluster_radius * np.sin(angles[i])
        for node, (x, y) in sub_pos.items():
            pos[node] = (cx + internal_scale * x, cy + internal_scale * y)

    strength = dict(G.degree(weight="weight"))
    node_sizes = [50 * np.sqrt(strength[node]) for node in G.nodes()]
    edge_widths = [0.10 * G[u][v]["weight"] for u, v in G.edges()]
    colors = [node_to_comm[node] for node in G.nodes()]

    created_fig = False
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
        created_fig = True

    nx.draw_networkx_edges(
        G, pos, width=edge_widths, alpha=edge_alpha, ax=ax,
    )
    nx.draw_networkx_nodes(
        G, pos, node_size=node_sizes, node_color=colors,
        cmap=plt.cm.tab20, alpha=0.9, ax=ax,
    )

    texts = []
    for node, (x, y) in pos.items():
        label = label_map.get(node, node) if label_map else node
        texts.append(
            ax.text(
                x, y, label,
                fontsize=font_size, ha="center", va="center", zorder=5,
            )
        )

    try:
        from adjustText import adjust_text
    except ImportError:
        adjust_text = None

    if adjust_text is not None:
        adjust_text(
            texts,
            ax=ax,
            expand=(1.4, 1.6),
            force_text=(1.2, 1.5),
            force_static=(0.5, 0.7),
            force_pull=(0.02, 0.02),
            arrowprops=dict(arrowstyle="-", alpha=0.25, lw=0.7),
        )

    if title is not None:
        ax.set_title(title, fontsize=title_font_size)
    ax.axis("off")

    if created_fig:
        plt.tight_layout()
        plt.show()

    return G, communities


def build_graph(df, authors_col="Author_IDs"):
    """Build a weighted co-authorship graph from all authors on each publication."""
    G = nx.Graph()
    for authors in df[authors_col].dropna():
        if not isinstance(authors, list):
            continue
        authors = list(dict.fromkeys(str(a).strip() for a in authors if str(a).strip()))
        for author in authors:
            G.add_node(author)
        for u, v in combinations(authors, 2):
            if G.has_edge(u, v):
                G[u][v]["weight"] += 1
            else:
                G.add_edge(u, v, weight=1)
    return G


def annual_network_statistics(df_eu, start_year=1992, end_year=2025):
    """Calculate yearly European co-authorship network metrics."""
    rows = []

    for year in range(start_year, end_year + 1):
        G = build_graph(df_eu[df_eu["Year"] == year])
        n_nodes = G.number_of_nodes()
        if n_nodes < 2:
            continue

        components = list(nx.connected_components(G))
        largest_nodes = max(components, key=len)
        largest_graph = G.subgraph(largest_nodes).copy()
        degrees = dict(G.degree())
        weighted_degrees = dict(G.degree(weight="weight"))

        rows.append(
            {
                "Year": year,
                "Nodes": n_nodes,
                "Edges": G.number_of_edges(),
                "Density": nx.density(G),
                "Average_degree": sum(degrees.values()) / n_nodes,
                "Weighted_degree": sum(weighted_degrees.values()) / n_nodes,
                "Clustering": nx.average_clustering(G, weight="weight"),
                "Components": len(components),
                "Largest_component": largest_graph.number_of_nodes(),
            }
        )

    return pd.DataFrame(rows).sort_values("Year").reset_index(drop=True)


def compare_network_periods(stats, intervention_year=2009):
    """Mann–Whitney + Holm + Cohen's d + Cliff's delta for five prespecified metrics."""
    before = stats[stats["Year"] < intervention_year]
    after = stats[stats["Year"] >= intervention_year]

    variables = [
        "Density",
        "Average_degree",
        "Weighted_degree",
        "Clustering",
        "Largest_component",
    ]
    labels = {
        "Density": "Network density",
        "Average_degree": "Average degree",
        "Weighted_degree": "Average weighted degree",
        "Clustering": "Weighted clustering coefficient",
        "Largest_component": "Largest connected component, nodes",
    }

    rows = []
    for var in variables:
        pre = before[var].dropna().astype(float)
        post = after[var].dropna().astype(float)

        U, p_raw = mannwhitneyu(pre, post, alternative="two-sided", method="auto")

        # Cohen's d, pre minus post, using pooled sample SD.
        n1, n2 = len(pre), len(post)
        pooled_sd = np.sqrt(
            ((n1 - 1) * pre.var(ddof=1) + (n2 - 1) * post.var(ddof=1))
            / (n1 + n2 - 2)
        )
        cohen_d = (pre.mean() - post.mean()) / pooled_sd

        cliff_value, cliff_magnitude = _cliffs_delta(pre.to_numpy(), post.to_numpy())

        rows.append(
            {
                "Metric": var,
                "Metric_label": labels[var],
                "Pre_mean": pre.mean(),
                "Pre_SD": pre.std(ddof=1),
                "Pre_median": pre.median(),
                "Post_mean": post.mean(),
                "Post_SD": post.std(ddof=1),
                "Post_median": post.median(),
                "U": U,
                "p_raw": p_raw,
                "Cohen_d": cohen_d,
                "Cliff_delta": cliff_value,
                "Cliff_magnitude": cliff_magnitude,
            }
        )

    result = pd.DataFrame(rows)
    reject, p_holm, _, _ = multipletests(result["p_raw"], alpha=0.05, method="holm")
    result["p_Holm"] = p_holm
    result["Significant_Holm"] = reject
    return result


def manuscript_network_table(network_comparison):
    """Format the main-manuscript Table 2."""
    out = network_comparison.copy()

    def fmt_mean_sd(row, prefix):
        metric = row["Metric"]
        mean = row[f"{prefix}_mean"]
        sd = row[f"{prefix}_SD"]
        if metric == "Density":
            return f"{mean:.3f} ± {sd:.3f}"
        if metric == "Clustering":
            return f"{mean:.2f} ± {sd:.2f}"
        if metric in {"Average_degree", "Weighted_degree"}:
            return f"{mean:.1f} ± {sd:.1f}"
        return f"{mean:.0f} ± {sd:.0f}"

    out["Pre_mean_SD"] = out.apply(lambda r: fmt_mean_sd(r, "Pre"), axis=1)
    out["Post_mean_SD"] = out.apply(lambda r: fmt_mean_sd(r, "Post"), axis=1)
    out["p_Holm_formatted"] = out["p_Holm"].map(lambda p: f"{p:.2e}")

    table = out[
        [
            "Metric_label", "Pre_mean_SD", "Post_mean_SD", "U",
            "p_Holm_formatted", "Cohen_d", "Cliff_delta",
        ]
    ].copy()
    table["U"] = table["U"].map(lambda x: f"{x:.1f}")
    table["Cohen_d"] = table["Cohen_d"].map(lambda x: f"{x:.2f}")
    table["Cliff_delta"] = table["Cliff_delta"].map(lambda x: f"{x:.2f}")
    table.columns = [
        "Metric",
        "1992–2008 mean ± SD",
        "2009–2025 mean ± SD",
        "Mann–Whitney U",
        "Holm-adjusted p",
        "Cohen's d",
        "Cliff's δ",
    ]
    return table


def fit_clustering_temporal_model(stats, intervention_year=2009):
    """Fit Clustering ~ Time * Post with Time centered on 2009."""
    data = stats.copy()
    data["Post"] = (data["Year"] >= intervention_year).astype(int)
    data["Time"] = data["Year"] - intervention_year
    model = smf.ols("Clustering ~ Time * Post", data=data).fit()

    pre_slope = model.params["Time"]
    slope_change = model.params["Time:Post"]
    return data, model, {
        "pre_slope": pre_slope,
        "slope_change": slope_change,
        "post_slope": pre_slope + slope_change,
        "interaction_p": model.pvalues["Time:Post"],
    }
