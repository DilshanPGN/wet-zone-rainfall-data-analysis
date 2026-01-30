"""
Perform homogeneity tests on the updated (filled) rainfall dataset.
Uses Pettitt test and SNHT (Standard Normal Homogeneity Test) on annual series.
"""
from pathlib import Path

import pandas as pd
import pyhomogeneity as hg

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DATASET_DIR = Path(__file__).resolve().parent / "dataset"
# Use filled dataset if available; otherwise use original
EXCEL_FILLED = DATASET_DIR / "Wet zone rainfall data_filled.xlsx"
EXCEL_ORIGINAL = DATASET_DIR / "Wet zone rainfall data.xlsx"
SHEET_NAME = "All"
RAINFALL_COLUMNS = [
    "Colombo", "Galle", "Nuwara Eliya", "Rathnapura", "Maliboda", "Deniyaya"
]
ALPHA = 0.05  # significance level


def load_data(path: Path) -> pd.DataFrame:
    """Load Excel sheet into DataFrame."""
    return pd.read_excel(path, sheet_name=SHEET_NAME)


def annual_totals(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily rainfall to annual totals per station (Year x Station)."""
    return df.groupby("Year")[RAINFALL_COLUMNS].sum()


def run_homogeneity_tests(series: pd.Series, station_name: str) -> dict:
    """
    Run Pettitt and SNHT on a single station's annual series.
    Returns dict with test results and inferred break year.
    """
    x = series.dropna().values
    years = series.dropna().index.tolist()
    if len(x) < 10:
        return {
            "station": station_name,
            "n_years": len(x),
            "pettitt_h": None,
            "pettitt_cp_year": None,
            "pettitt_p": None,
            "pettitt_mu1": None,
            "pettitt_mu2": None,
            "snht_h": None,
            "snht_cp_year": None,
            "snht_p": None,
        }

    # Pettitt test
    pettitt = hg.pettitt_test(x, alpha=ALPHA)
    # cp is 1-based index of change point (first index of second segment)
    cp_idx = pettitt.cp
    pettitt_year = years[cp_idx - 1] if 1 <= cp_idx <= len(years) else None

    # SNHT test
    snht = hg.snht_test(x, alpha=ALPHA)
    snht_cp_idx = snht.cp
    snht_year = years[snht_cp_idx - 1] if 1 <= snht_cp_idx <= len(years) else None

    mu1 = pettitt.avg.mu1 if hasattr(pettitt.avg, "mu1") else None
    mu2 = pettitt.avg.mu2 if hasattr(pettitt.avg, "mu2") else None

    return {
        "station": station_name,
        "n_years": len(x),
        "pettitt_h": pettitt.h,
        "pettitt_cp_year": pettitt_year,
        "pettitt_p": round(pettitt.p, 4),
        "pettitt_mu1": round(mu1, 2) if mu1 is not None else None,
        "pettitt_mu2": round(mu2, 2) if mu2 is not None else None,
        "snht_h": snht.h,
        "snht_cp_year": snht_year,
        "snht_p": round(snht.p, 4),
    }


def main() -> None:
    path = EXCEL_FILLED if EXCEL_FILLED.exists() else EXCEL_ORIGINAL
    if not path.exists():
        raise FileNotFoundError(
            f"Neither filled nor original Excel found in {DATASET_DIR}"
        )
    print(f"Loading: {path.name}")
    df = load_data(path)
    annual = annual_totals(df)

    print("\nHomogeneity tests (alpha = {})".format(ALPHA))
    print("  Pettitt: non-parametric change-point test")
    print("  SNHT: Standard Normal Homogeneity Test")
    print("-" * 80)

    results = []
    for col in RAINFALL_COLUMNS:
        res = run_homogeneity_tests(annual[col], col)
        results.append(res)

    # Summary table
    rows = []
    for r in results:
        pettitt_status = "Non-homogeneous" if r["pettitt_h"] else "Homogeneous"
        snht_status = "Non-homogeneous" if r["snht_h"] else "Homogeneous"
        rows.append({
            "Station": r["station"],
            "N years": r["n_years"],
            "Pettitt": pettitt_status,
            "Pettitt break year": r["pettitt_cp_year"] or "—",
            "Pettitt p-value": r["pettitt_p"] or "—",
            "SNHT": snht_status,
            "SNHT break year": r["snht_cp_year"] or "—",
            "SNHT p-value": r["snht_p"] or "—",
        })
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))

    # Detailed breakdown for non-homogeneous
    non_hom = [r for r in results if r["pettitt_h"] or r["snht_h"]]
    if non_hom:
        print("\n--- Stations with detected break(s) ---")
        for r in non_hom:
            print(f"\n  {r['station']}:")
            if r["pettitt_h"]:
                print(
                    f"    Pettitt: break at {r['pettitt_cp_year']} (p={r['pettitt_p']}); "
                    f"mean before ≈ {r['pettitt_mu1']}, after ≈ {r['pettitt_mu2']} mm/yr"
                )
            if r["snht_h"]:
                print(f"    SNHT: break at {r['snht_cp_year']} (p={r['snht_p']})")
    else:
        print("\nAll stations are homogeneous at alpha = {}.".format(ALPHA))

    # Save results to CSV
    out_csv = Path(__file__).resolve().parent / "homogeneity_test_results.csv"
    table.to_csv(out_csv, index=False)
    print(f"\nResults saved to: {out_csv}")


if __name__ == "__main__":
    main()
