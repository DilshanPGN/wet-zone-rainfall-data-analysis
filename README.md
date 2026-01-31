# Wet Zone Rainfall Data Analysis

Analysis of wet-zone rainfall data for Sri Lanka (past ~30 years): missing-data imputation, homogeneity testing, Innovative Trend Analysis (ITA), and Mann–Kendall trend tests.

---

## Table of contents

1. [Prerequisites](#1-prerequisites)
2. [Setup (from the beginning)](#2-setup-from-the-beginning)
3. [Where to add the dataset](#3-where-to-add-the-dataset)
4. [How to fill missing values](#4-how-to-fill-missing-values)
5. [How to run calculation scripts](#5-how-to-run-calculation-scripts)
6. [How to run the UI](#6-how-to-run-the-ui)
7. [Project structure](#7-project-structure)
8. [Further documentation](#8-further-documentation)

---

## 1. Prerequisites

- **Python 3.10+** (3.11 or 3.12 recommended)
- **Excel input:** one workbook with sheet `"All"` containing columns: `Year`, `Month`, and rainfall station columns (see [Where to add the dataset](#3-where-to-add-the-dataset))

---

## 2. Setup (from the beginning)

From a clean clone or download of the project:

**Step 1 — Go to the project folder**

```bash
cd wet-zone-rainfall-data-analysis
```

**Step 2 — Create a virtual environment (recommended)**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

**Step 3 — Install dependencies**

```bash
pip install -r requirements.txt
```

**Step 4 — Add your dataset**

Place your Excel file in the `dataset` folder (see next section). The app and scripts expect either the original file or the filled file to be present.

---

## 3. Where to add the dataset

- **Folder:** `dataset/` at the **project root** (same level as `ita.py`, `app_trend_analysis.py`).

- **Original file (required for imputation):**  
  `dataset/Wet zone rainfall data.xlsx`

- **Filled file (optional, created by the fill script):**  
  `dataset/Wet zone rainfall data_filled.xlsx`  
  If this file exists, the app and trend scripts use it; otherwise they use the original file.

- **Excel structure:**
  - **Sheet name:** `All`
  - **Columns:**  
    - `Year` (e.g. 1991–2021)  
    - `Month` (1–12)  
    - Optional: `Date` (for daily data)  
    - Rainfall (mm) per station: `Colombo`, `Galle`, `Nuwara Eliya`, `Rathnapura`, `Maliboda`, `Deniyaya`

Example layout:

| Year | Month | Date | Colombo | Galle | Nuwara Eliya | Rathnapura | Maliboda | Deniyaya |
|------|-------|------|---------|-------|--------------|------------|----------|----------|
| 1991 | 1     | 1    | 12.5    | 8.2   | ...          | ...        | ...      | ...      |

If you use different file or sheet names, you must change the paths/sheet name in the scripts (`ita.py`, `fill_missing_rainfall.py`, `homogeneity_test.py`).

---

## 4. How to fill missing values

Missing rainfall values are filled using the **Normal Ratio Method** (and linear interpolation where no reference station has data). The script writes a **new** Excel file and **does not overwrite** the original. Imputed cells are **highlighted in light yellow**.

**Command:**

```bash
python fill_missing_rainfall.py
```

- **Input:** `dataset/Wet zone rainfall data.xlsx` (sheet `All`)
- **Output:** `dataset/Wet zone rainfall data_filled.xlsx` (imputed values; imputed cells in yellow)

**Method summary:** For each station, missing values are estimated from other stations using monthly means (Normal Ratio). If no reference has data for that day, linear interpolation along the target station’s series is used. Details: [docs/MISSING_DATA_IMPUTATION.md](docs/MISSING_DATA_IMPUTATION.md).

**When to run:** Run once after placing the original Excel in `dataset/`. After that, the trend app and other scripts will prefer the filled file if it exists.

---

## 5. How to run calculation scripts

All commands are run from the **project root** (where `requirements.txt` and `ita.py` are).

### 5.1 Homogeneity test (Pettitt + SNHT)

Runs homogeneity tests on **annual** rainfall totals per station. Uses the filled dataset if present, otherwise the original.

```bash
python homogeneity_test.py
```

- **Input:** `dataset/Wet zone rainfall data_filled.xlsx` (or original)
- **Output:** `homogeneity_test_results.csv` in the project root (and printed summary)

### 5.2 Export Mann–Kendall results (all stations)

Computes Mann–Kendall trend test for **annual** totals for every station and writes a CSV.

```bash
python export_mk_results.py
```

- **Input:** Same as above (filled or original)
- **Output:** `outputs/mann_kendall_results.csv` (creates `outputs/` if needed)

### 5.3 ITA demo (command-line)

Runs the full ITA pipeline for the first station and saves two PNGs (time series + trend, ITA scatter).

```bash
python ita.py
```

- **Input:** Filled or original dataset
- **Output:** `ita_timeseries_trend.png`, `ita_demo.png` in the project root

---

## 6. How to run the UI

The **Streamlit** app provides interactive trend analysis: station selection, year range, time interval, ITA and Mann–Kendall results, chart export (PNG/PDF), and CSV export of trend data.

**Command:**

```bash
streamlit run app_trend_analysis.py
```

- The terminal will show a local URL, e.g. **http://localhost:8501**. Open it in a browser.
- **Data:** Uses `dataset/Wet zone rainfall data_filled.xlsx` if present, else `dataset/Wet zone rainfall data.xlsx`. Ensure at least one of these exists before starting.

**Features:**

- **Single station analysis:** Interactive time series (drag to select range), Rainfall time series + Sen’s trend, ITA scatter, Mann–Kendall test and chart, AI-style interpretation, export PNG/PDF per chart and CSV trend data (time series + trend, ITA scatter, summary).
- **Trend by station (all stations):** Bar chart of Mann–Kendall trend per station.
- **Export:** Per-chart PNG/PDF, combined ZIP of charts, and CSV downloads (time series + trend, ITA scatter, summary) for drawing your own charts.

---

## 7. Project structure

```
wet-zone-rainfall-data-analysis/
├── dataset/
│   ├── Wet zone rainfall data.xlsx      # Original (you add this)
│   └── Wet zone rainfall data_filled.xlsx  # Created by fill_missing_rainfall.py
├── docs/
│   ├── HOMOGENEITY_TEST.md
│   ├── MISSING_DATA_IMPUTATION.md
│   └── TREND_ANALYSIS_METHODS.md
├── outputs/
│   └── mann_kendall_results.csv          # Created by export_mk_results.py
├── app_trend_analysis.py                # Streamlit UI
├── fill_missing_rainfall.py              # Missing-value imputation
├── homogeneity_test.py                  # Homogeneity tests
├── export_mk_results.py                 # Mann–Kendall export
├── ita.py                               # ITA + Mann–Kendall logic & CLI demo
├── requirements.txt
├── README.md                             # This file
├── INSTRUCTIONS.md                       # Project instructions
└── homogeneity_test_results.csv        # Created by homogeneity_test.py
```

---

## 8. Further documentation

- **Missing data imputation:** [docs/MISSING_DATA_IMPUTATION.md](docs/MISSING_DATA_IMPUTATION.md)
- **Homogeneity tests:** [docs/HOMOGENEITY_TEST.md](docs/HOMOGENEITY_TEST.md)
- **Trend analysis (ITA & Mann–Kendall):** [docs/TREND_ANALYSIS_METHODS.md](docs/TREND_ANALYSIS_METHODS.md)
- **Project instructions and data availability:** [INSTRUCTIONS.md](INSTRUCTIONS.md)

---

## Quick reference — commands from the beginning

```bash
# 1. Enter project and setup
cd wet-zone-rainfall-data-analysis
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate      # macOS/Linux
pip install -r requirements.txt

# 2. Add dataset/Wet zone rainfall data.xlsx (see §3)

# 3. Fill missing values (optional but recommended)
python fill_missing_rainfall.py

# 4. Run analyses (optional)
python homogeneity_test.py
python export_mk_results.py

# 5. Run the UI
streamlit run app_trend_analysis.py
```

Then open **http://localhost:8501** in your browser.
