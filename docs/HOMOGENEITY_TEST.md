# Homogeneity Test — Documentation

This document describes the homogeneity test script and the tests used on the updated (filled) rainfall dataset.

---

## Purpose

A **homogeneity test** checks whether a rainfall time series has **non-climatic breaks** (e.g. from station relocation, instrument change, or observer change). If such breaks are present, trend analysis (e.g. Mann–Kendall, ITA) can be biased. The test helps identify stations or periods that may need correction or separate analysis.

---

## Script and Data

- **Script:** `homogeneity_test.py`
- **Input:** Uses the **filled** dataset if present: `dataset/Wet zone rainfall data_filled.xlsx`. Otherwise falls back to `dataset/Wet zone rainfall data.xlsx`.
- **Aggregation:** Daily rainfall is **aggregated to annual totals** (sum per year per station). Homogeneity tests are run on these annual series to reduce noise.
- **Output:** Console summary and `homogeneity_test_results.csv` in the project root.

---

## Tests Used

The script runs two standard homogeneity tests (from the [pyhomogeneity](https://github.com/mmhs013/pyHomogeneity) package):

### 1. Pettitt test

- **Type:** Non-parametric change-point test (based on Mann–Whitney).
- **Detects:** A single step change in the mean (breakpoint).
- **Result:** \(h\) = True if non-homogeneous at \(\alpha\); \(cp\) = probable change-point index; \(p\) = p-value; means before (\(\mu_1\)) and after (\(\mu_2\)) the break.

### 2. SNHT (Standard Normal Homogeneity Test)

- **Type:** Parametric test (assumes approximate normality of annual totals).
- **Detects:** A single shift in the mean.
- **Result:** \(h\) = True if non-homogeneous at \(\alpha\); \(cp\) = change-point index; \(p\) = p-value.

**Significance level:** \(\alpha = 0.05\) (default). If \(p \le 0.05\), the series is classified as **non-homogeneous** (break detected).

---

## How to Run

From the project root:

```bash
pip install -r requirements.txt
python homogeneity_test.py
```

---

## Interpreting Results

- **Homogeneous:** No significant break detected at \(\alpha = 0.05\). The series is suitable for trend analysis as a single segment (unless you have other reasons to segment).
- **Non-homogeneous:** A significant break is detected. Consider:
  - **Correcting** the series (e.g. adjust level after the break using a reference series),
  - **Segmenting** (run trend analysis before and after the break separately),
  - **Excluding** the faulty period or station,
  - **Documenting** the break and qualifying conclusions.

The CSV and console output report the **most probable break year** for each test even when the result is homogeneous; use it only when the test is significant.

---

## References

- Pettitt, A.N. (1979). A non-parametric approach to the change-point problem. *Journal of the Royal Statistical Society: Series C*, 28(2), 126–135.
- Alexandersson, H. (1986). A homogeneity test applied to precipitation data. *Journal of Climatology*, 6(6), 661–675.
- pyHomogeneity: [GitHub](https://github.com/mmhs013/pyHomogeneity), [PyPI](https://pypi.org/project/pyhomogeneity/).
