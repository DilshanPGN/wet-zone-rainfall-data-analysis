# Trend Analysis: Methods, Equations, and Results

This document describes the methods and equations used for trend analysis in this project (Innovative Trend Analysis, Sen's slope, and Mann–Kendall test), and how to interpret the results.

---

## 1. Overview

The project uses two complementary approaches:

1. **Innovative Trend Analysis (ITA)** — a graphical method that splits the time series into two halves and compares them; no distributional assumption.
2. **Mann–Kendall (MK) test** — a non-parametric hypothesis test for monotonic trend, with **Sen's slope** as the trend magnitude.

Both use the same rainfall series (annual totals, or monthly/daily values). Results are exported to `outputs/mann_kendall_results.csv` and shown in the Streamlit app.

---

## 2. Innovative Trend Analysis (ITA)

### 2.1 Method

- The time series \(x_1, x_2, \ldots, x_n\) (e.g. annual rainfall) is split into **first half** and **second half**.
- If \(n\) is even: first half = \(x_1, \ldots, x_{n/2}\); second half = \(x_{n/2+1}, \ldots, x_n\).
- If \(n\) is odd: first half has \((n-1)/2\) points, second half the rest.
- A scatter plot is drawn: **x-axis** = values in the first half (in order), **y-axis** = values in the second half (in order).
- The **1:1 line** (no trend) is the line \(y = x\).

### 2.2 Interpretation

- **Points above the 1:1 line** → second-half values are higher than first-half → **increasing** trend.
- **Points below the 1:1 line** → second-half values are lower → **decreasing** trend.
- **Points on the line** → no change.
- The **percentage of points above/below** the 1:1 line and the **mean change** (second half − first half) summarize the trend. No normality or other distribution is assumed.

**Reference:** Şen, Z. (2012). Innovative trend analysis methodology. *Journal of Hydrologic Engineering*, 17(9), 1042–1046.

---

## 3. Sen's Slope (Theil–Sen Estimator)

Sen's slope is the **median of all pairwise slopes** between time points. It is robust to outliers and is used both in ITA (for the trend line on the time series) and in the Mann–Kendall output.

### 3.1 Equation

For a series \(y_1, y_2, \ldots, y_n\) at times \(t_1, t_2, \ldots, t_n\) (or indices \(1, 2, \ldots, n\)):

\[
\text{slope}_{ij} = \frac{y_j - y_i}{t_j - t_i}, \quad i < j
\]

**Sen's slope** \(\beta\) is the **median** of all such \(\text{slope}_{ij}\):

\[
\beta = \operatorname{median}\bigl\{ \text{slope}_{ij} : i < j \bigr\}
\]

The **intercept** is taken as \(\operatorname{median}(y_i - \beta t_i)\) so that the line \(y = \beta t + \text{intercept}\) fits the data in a robust way.

### 3.2 Units

- For **annual** series: slope has units **mm/year** (change in rainfall per year).
- For **monthly** or **daily** series: slope is per step (month or day).

---

## 4. Mann–Kendall Trend Test

### 4.1 Purpose

The Mann–Kendall test checks whether there is a **monotonic trend** (consistently increasing or decreasing) in the time series. It does **not** assume normality; it uses the ranks of the data.

### 4.2 Test Statistic \(S\)

For data \(x_1, x_2, \ldots, x_n\) (in time order), define the sign of each pair:

\[
\operatorname{sgn}(x_j - x_i) =
\begin{cases}
  +1 & \text{if } x_j > x_i \\
  0  & \text{if } x_j = x_i \\
  -1 & \text{if } x_j < x_i
\end{cases}
\quad (i < j)
\]

The **Mann–Kendall statistic** is:

\[
S = \sum_{i=1}^{n-1} \sum_{j=i+1}^{n} \operatorname{sgn}(x_j - x_i)
\]

- \(S > 0\) → more increases than decreases → **increasing** trend.
- \(S < 0\) → more decreases → **decreasing** trend.
- \(S = 0\) → no trend.

### 4.3 Variance of \(S\) (no ties)

When there are **no tied values**, the variance of \(S\) under the null hypothesis (no trend) is:

\[
\operatorname{Var}(S) = \frac{n(n-1)(2n+5)}{18}
\]

When there **are ties**, a correction is applied (see standard references or the `pymannkendall` implementation).

### 4.4 Standardized Test Statistic \(z\)

\[
z =
\begin{cases}
  \frac{S - 1}{\sqrt{\operatorname{Var}(S)}} & \text{if } S > 0 \\
  0 & \text{if } S = 0 \\
  \frac{S + 1}{\sqrt{\operatorname{Var}(S)}} & \text{if } S < 0
\end{cases}
\]

The continuity correction (\(\pm 1\)) is often used. Under the null hypothesis, \(z\) is approximately standard normal for large \(n\).

### 4.5 p-Value and Trend Decision

- The **p-value** is obtained from the standard normal distribution using \(z\) (two-tailed test: we test for any trend, not only increase or only decrease).
- **Significance level** \(\alpha = 0.05\) is used:
  - If **p < 0.05** → reject the null hypothesis → **significant trend** (reported as "increasing" or "decreasing" depending on the sign of \(S\)).
  - If **p ≥ 0.05** → do not reject → **no significant trend** (reported as "no trend").

So **"no trend"** in the output means **no statistically significant monotonic trend at α = 0.05**, not that the slope is exactly zero. Sen's slope can still be positive or negative.

### 4.6 Sen's Slope in MK

The Mann–Kendall procedure is often reported together with **Sen's slope** (same formula as in Section 3) as the **magnitude** of the trend. The project uses `pymannkendall`, which returns both the MK test result and Sen's slope.

---

## 5. Results: Output Files and Interpretation

### 5.1 Exported File: `outputs/mann_kendall_results.csv`

| Column | Meaning |
|--------|--------|
| **Station** | Rain gauge (city) name. |
| **trend** | MK result: `"increasing"`, `"decreasing"`, or `"no trend"` (no significant trend at α = 0.05). |
| **p_value** | Mann–Kendall two-tailed p-value. |
| **Sen_slope_mm_per_year** | Sen's slope in mm/year (for annual series). |
| **significant_alpha_005** | `True` if p < 0.05, else `False`. |
| **n** | Number of years (or steps) in the series. |

### 5.2 Interpretation of Current Results

For the wet-zone rainfall dataset (annual totals, full year range):

- **All six stations** show **trend = "no trend"** and **p_value > 0.05** (roughly 0.25–0.87).
- **Sen's slope** is non-zero for all stations (positive for Colombo, Galle, Maliboda, Deniyaya; negative for Nuwara Eliya, Rathnapura), but **none of these slopes are statistically significant** at α = 0.05.
- So: **there is no statistically significant monotonic trend** in annual rainfall over the period covered by the data for any of the stations. High year-to-year variability and/or a weak trend lead to p ≥ 0.05.

### 5.3 How to Read the App and CSV Together

- **ITA** (chart and % above/below 1:1 line): gives a **visual and descriptive** view of trend (first half vs second half).
- **Mann–Kendall** (p-value, trend, Sen's slope): gives a **statistical test** and **magnitude**.
- Use both: e.g. ITA may suggest an increase, but MK confirms whether that increase is **significant** (p < 0.05) or not (p ≥ 0.05 → "no trend").

---

## 6. Summary of Equations (Quick Reference)

| Item | Equation or rule |
|------|------------------|
| **ITA** | Split series into two halves; plot second half vs first half; compare to 1:1 line \(y = x\). |
| **Sen's slope** | \(\beta = \operatorname{median}\bigl\{ (y_j - y_i)/(t_j - t_i) : i < j \bigr\}\). |
| **MK statistic** | \(S = \sum_{i<j} \operatorname{sgn}(x_j - x_i)\). |
| **MK variance (no ties)** | \(\operatorname{Var}(S) = n(n-1)(2n+5)/18\). |
| **MK z** | \(z = (S \mp 1) / \sqrt{\operatorname{Var}(S)}\) (with continuity correction). |
| **Trend decision** | p < 0.05 → significant trend (increasing/decreasing); p ≥ 0.05 → "no trend". |

---

## 7. References and Code

- **ITA:** Şen (2012), *J. Hydrol. Eng.* 17(9), 1042–1046.
- **Mann–Kendall:** Standard non-parametric trend test; implementation via `pymannkendall` (e.g. `mk.original_test()`).
- **Sen's slope:** Theil–Sen estimator; implemented in `ita.sens_slope()` and reported by `pymannkendall`.

Scripts: `ita.py` (ITA, Sen's slope, MK wrapper), `export_mk_results.py` (writes `outputs/mann_kendall_results.csv`), `app_trend_analysis.py` (Streamlit UI).
