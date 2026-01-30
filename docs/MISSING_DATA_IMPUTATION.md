# Missing Rainfall Data Imputation — Method and Script Documentation

This document describes how missing rainfall data are predicted and filled: the standard equations, the method used, and how the script implements them.

---

## 1. Standard Method: Normal Ratio Method

The script uses the **Normal Ratio Method**, a standard approach in hydrology for estimating missing precipitation at one station using observed values at one or more reference stations. It is recommended in guidelines such as the World Meteorological Organization (WMO) and is widely used in rainfall data infilling.

### 1.1 Single-reference equation

For one target station \(X\) and one reference station \(R\), the estimate for a missing value at the target on a given day is:

\[
P_X = \frac{\bar{P}_X^{(m)}}{\bar{P}_R^{(m)}} \cdot P_R
\]

Where:

- \(P_X\) = estimated rainfall at the target station (mm)
- \(P_R\) = observed rainfall at the reference station on the same day (mm)
- \(\bar{P}_X^{(m)}\) = long-term mean rainfall at the target station for month \(m\)
- \(\bar{P}_R^{(m)}\) = long-term mean rainfall at the reference station for month \(m\)
- \(m\) = month of the missing day (1–12)

The ratio \(\bar{P}_X^{(m)} / \bar{P}_R^{(m)}\) is the **normal ratio** for that month. Using monthly means preserves seasonality (e.g. wet vs dry months).

### 1.2 Multiple reference stations

When several reference stations have data on the same day, the script uses the **average of the single-reference estimates** (modified normal ratio):

\[
P_X = \frac{1}{N} \sum_{i=1}^{N} \left( \frac{\bar{P}_X^{(m)}}{\bar{P}_{R_i}^{(m)}} \cdot P_{R_i} \right)
\]

Where:

- \(N\) = number of reference stations with valid data on that day
- \(R_i\) = reference station \(i\)
- \(P_{R_i}\) = observed rainfall at station \(R_i\) on that day

This reduces the influence of any one reference and uses all available information.

### 1.3 Fallback: linear interpolation

If **no** reference station has data on that day (e.g. all are missing), the script falls back to **linear interpolation** along the time series of the target station:

- Missing values are interpolated from the previous and next known values in the same column (by row order, i.e. by date).
- This is a standard time-series infilling method when no spatial reference is available.

---

## 2. How the script uses these equations

### 2.1 Data and stations

- **Input:** Excel file `Wet zone rainfall data.xlsx`, sheet `All`.
- **Structure:** Rows = daily records; columns include `Year`, `Month`, `Date`, and rainfall (mm) for: Colombo, Galle, Nuwara Eliya, Rathnapura, Maliboda, Deniyaya.
- **Target columns:** All six rainfall columns are checked for missing values; Maliboda and Deniyaya have the largest gaps (as in INSTRUCTIONS.md).

### 2.2 Step-by-step procedure

1. **Load data**  
   The script reads the sheet into a pandas DataFrame.

2. **Compute monthly means**  
   For each rainfall column and each month (1–12), it computes the long-term mean over all years (using only non-NaN values).  
   Formula: \(\bar{P}_{\text{station}}^{(\text{month})} = \text{mean of all observed values in that station and month}\).

3. **Imputation for each target column**  
   For each row where the target station has a missing value:
   - **Normal ratio:** For every other rainfall station that has a non-missing value on that row, compute the single-reference estimate using the equation in §1.1 (using the month of that row).
   - **Average:** Replace the missing value by the **average** of these estimates (equation in §1.2). If at least one reference had data, the cell is marked as “imputed”.
   - **Fallback:** If no reference had data on that row, the script uses **linear interpolation** along that column and marks those cells as imputed as well.

4. **Write output and highlight**  
   - The script **does not** overwrite the original file. It **reads** the original workbook with `openpyxl`, writes the **imputed values** into the same sheet (same cells), and **saves** to a new file: `Wet zone rainfall data_filled.xlsx`.
   - Only cells that were actually filled (imputed) get their value updated and are **highlighted** with a light yellow fill (color code `#FFFF99`) so you can see which values are estimated.

### 2.3 Summary of equations implemented

| Situation | Equation / method |
|----------|---------------------|
| One or more references have data | \(P_X = \frac{1}{N} \sum_i \frac{\bar{P}_X^{(m)}}{\bar{P}_{R_i}^{(m)}} P_{R_i}\) (month \(m\)) |
| No reference has data | Linear interpolation along the target station’s time series |

---

## 3. How to run the script

**Requirements:** Python 3.9+, `pandas`, `openpyxl` (see `requirements.txt`).

From the project root:

```bash
pip install -r requirements.txt
python fill_missing_rainfall.py
```

- **Input:** `dataset/Wet zone rainfall data.xlsx`  
- **Output:** `dataset/Wet zone rainfall data_filled.xlsx` (imputed values, imputed cells in light yellow).

---

## 4. Assumptions and limitations

- **Spatial correlation:** The normal ratio method assumes that rainfall at the target and reference stations is correlated and that the ratio of long-term monthly means is stable. This is reasonable for nearby stations in the same climatic region (e.g. wet zone).
- **Monthly means:** Means are taken over all available years; if a station has many missing years, its monthly means may be less representative.
- **Interpolation fallback:** Used only when no reference has data; it does not use spatial information and is less reliable for long gaps.
- **Rounding:** Imputed values are written to Excel rounded to 2 decimal places.

---

## 5. References (conceptual)

- WMO (World Meteorological Organization) guidelines on filling missing precipitation data.
- Standard hydrology texts on precipitation network design and data infilling (e.g. normal ratio and regression-based methods).
