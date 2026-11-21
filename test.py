# Significance tests for the "Patient profile" slide (age bucket and primary payer by cohort).
# One chi-square test per panel for the whole distribution, then a two-proportion z-test for
# every category and cohort pair, with a Benjamini-Hochberg correction across all pairs.

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact, norm
from statsmodels.stats.multitest import multipletests

# --------------------------------------------------
# 1. Input data (read off the slide)
# --------------------------------------------------

significance_level = 0.05
output_file = "patient_profile_significance_tests.xlsx"

cohort_sizes = {
    "Known treated": 515,
    ">80% look-alike": 271,
    "60-79% look-alike": 729,
    "<60% rest of diagnosed": 7119,
}
slide_cohorts = list(cohort_sizes.keys())

# Percent of each cohort that falls in each category (rows = categories, columns = cohorts)
age_percentages = pd.DataFrame(
    [
        [17, 25, 18, 14],
        [34, 34, 44, 27],
        [24, 19, 19, 15],
        [23, 21, 18, 28],
        [1, 0, 1, 16],
    ],
    index=["0 to 3", "4 to 9", "10 to 12", "13 to 18", "18+"],
    columns=slide_cohorts,
)

payer_percentages = pd.DataFrame(
    [
        [64, 58, 61, 43],
        [35, 41, 38, 51],
        [1, 0, 1, 6],
    ],
    index=["Commercial", "Medicaid", "Others"],
    columns=slide_cohorts,
)

panel_percentages = {"Age bucket": age_percentages, "Primary payer": payer_percentages}

# Cohort pairs to compare. The combined look-alike cohort is never compared with its own two
# bands, because those groups share patients and the tests below need independent groups.
combined_cohort = "All look-alike (>=60%)"
comparisons = [
    ("Known treated", combined_cohort),
    ("Known treated", ">80% look-alike"),
    ("Known treated", "60-79% look-alike"),
    ("Known treated", "<60% rest of diagnosed"),
    (combined_cohort, "<60% rest of diagnosed"),
    (">80% look-alike", "60-79% look-alike"),
    (">80% look-alike", "<60% rest of diagnosed"),
    ("60-79% look-alike", "<60% rest of diagnosed"),
]

# --------------------------------------------------
# 2. Convert percentages into patient counts
# --------------------------------------------------

# The slide only shows rounded percentages, so each count can be off by about 0.5% of the
# cohort size. Replace these with exact counts if they become available.
panel_counts = {}
for panel_name, percentages in panel_percentages.items():
    counts = pd.DataFrame(index=percentages.index)
    for cohort in slide_cohorts:
        counts[cohort] = (percentages[cohort] / 100 * cohort_sizes[cohort]).round().astype(int)

    # The slide callouts pool the two look-alike bands (271 + 729 = 1,000 patients)
    counts[combined_cohort] = counts[">80% look-alike"] + counts["60-79% look-alike"]
    panel_counts[panel_name] = counts

cohort_sizes[combined_cohort] = cohort_sizes[">80% look-alike"] + cohort_sizes["60-79% look-alike"]

# --------------------------------------------------
# 3. Chi-square test: does the whole distribution differ between cohorts?
# --------------------------------------------------

chi_square_groups = {
    "All four cohorts": slide_cohorts,
    "Known treated vs all look-alike": ["Known treated", combined_cohort],
}

chi_square_rows = []
for panel_name, counts in panel_counts.items():
    for group_name, group_cohorts in chi_square_groups.items():
        contingency_table = counts[group_cohorts]
        chi_square, p_value, degrees_of_freedom, expected_counts = chi2_contingency(contingency_table)
        chi_square_rows.append({
            "Panel": panel_name,
            "Cohorts compared": group_name,
            "Chi-square": chi_square,
            "Degrees of freedom": degrees_of_freedom,
            "P-value": p_value,
            "Significant": p_value < significance_level,
            # the chi-square approximation is weak when expected counts fall below 5
            "Smallest expected count": expected_counts.min(),
        })

df_chi_square = pd.DataFrame(chi_square_rows)

# --------------------------------------------------
# 4. Pairwise test for every category and cohort pair
# --------------------------------------------------

# "Under 10" is the slide callout (0 to 3 plus 4 to 9). It is added after the chi-square test
# so that the same patients are not counted twice in the contingency table.
age_counts = panel_counts["Age bucket"]
age_counts.loc["Under 10"] = age_counts.loc["0 to 3"] + age_counts.loc["4 to 9"]

pairwise_rows = []
for panel_name, counts in panel_counts.items():
    for category in counts.index:
        for cohort_1, cohort_2 in comparisons:
            count_1 = counts.loc[category, cohort_1]
            count_2 = counts.loc[category, cohort_2]
            n_1 = cohort_sizes[cohort_1]
            n_2 = cohort_sizes[cohort_2]
            proportion_1 = count_1 / n_1
            proportion_2 = count_2 / n_2
            difference = proportion_1 - proportion_2

            # Under the null hypothesis both cohorts share one proportion, estimated by pooling them
            pooled_proportion = (count_1 + count_2) / (n_1 + n_2)
            smallest_expected_count = min(n_1, n_2) * min(pooled_proportion, 1 - pooled_proportion)

            if smallest_expected_count >= 5:
                pooled_standard_error = np.sqrt(
                    pooled_proportion * (1 - pooled_proportion) * (1 / n_1 + 1 / n_2)
                )
                z_statistic = difference / pooled_standard_error
                p_value = 2 * norm.sf(abs(z_statistic))
                test_used = "Two-proportion z-test"
            else:
                # Too few patients for the normal approximation, so use the exact test on the 2x2 table
                two_by_two_table = [[count_1, n_1 - count_1], [count_2, n_2 - count_2]]
                z_statistic = np.nan
                p_value = fisher_exact(two_by_two_table)[1]
                test_used = "Fisher exact test"

            # 95% confidence interval of the difference (each cohort keeps its own variance here)
            unpooled_standard_error = np.sqrt(
                proportion_1 * (1 - proportion_1) / n_1 + proportion_2 * (1 - proportion_2) / n_2
            )
            margin_of_error = norm.ppf(0.975) * unpooled_standard_error

            pairwise_rows.append({
                "Panel": panel_name,
                "Category": category,
                "Group 1": cohort_1,
                "Group 2": cohort_2,
                "Count 1": count_1,
                "Count 2": count_2,
                "N1": n_1,
                "N2": n_2,
                "Proportion 1 (%)": proportion_1 * 100,
                "Proportion 2 (%)": proportion_2 * 100,
                "Difference (pp)": difference * 100,
                "95% CI low (pp)": (difference - margin_of_error) * 100,
                "95% CI high (pp)": (difference + margin_of_error) * 100,
                "Test used": test_used,
                "Z-statistic": z_statistic,
                "P-value": p_value,
            })

df_pairwise = pd.DataFrame(pairwise_rows)

# --------------------------------------------------
# 5. Multiple testing correction
# --------------------------------------------------

# With this many tests some will fall below 0.05 by chance alone. Benjamini-Hochberg keeps the
# expected share of false positives among the significant results at the significance level.
df_pairwise["Adjusted P-value"] = multipletests(df_pairwise["P-value"], method="fdr_bh")[1]
df_pairwise["Significant"] = df_pairwise["P-value"] < significance_level
df_pairwise["Significant after FDR"] = df_pairwise["Adjusted P-value"] < significance_level

# --------------------------------------------------
# 6. Print the slide callouts and export everything to Excel
# --------------------------------------------------

is_callout = (
    (df_pairwise["Group 1"] == "Known treated")
    & (df_pairwise["Group 2"] == combined_cohort)
    & df_pairwise["Category"].isin(["Under 10", "Medicaid"])
)
callout_columns = [
    "Category", "Proportion 1 (%)", "Proportion 2 (%)", "Difference (pp)",
    "95% CI low (pp)", "95% CI high (pp)", "P-value", "Adjusted P-value",
]

print("Chi-square tests")
print(df_chi_square.round(4).to_string(index=False))
print("\nSlide callouts: Known treated (1) vs all look-alike (2)")
print(df_pairwise.loc[is_callout, callout_columns].round(4).to_string(index=False))
print(f"\n{df_pairwise['Significant after FDR'].sum()} of {len(df_pairwise)} pairwise tests "
      f"are significant after FDR correction")

with pd.ExcelWriter(output_file) as writer:
    df_chi_square.to_excel(writer, sheet_name="Chi-square", index=False)
    df_pairwise.to_excel(writer, sheet_name="Pairwise tests", index=False)
    panel_counts["Age bucket"].to_excel(writer, sheet_name="Age counts")
    panel_counts["Primary payer"].to_excel(writer, sheet_name="Payer counts")

print(f"Saved {output_file}")
