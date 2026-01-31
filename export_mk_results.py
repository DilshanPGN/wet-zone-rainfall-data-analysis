"""
Export Mann–Kendall trend test results for all stations to outputs folder.
"""
from pathlib import Path

import pandas as pd

from ita import load_data, get_series, run_mann_kendall, RAINFALL_COLUMNS

OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_CSV = OUTPUTS_DIR / "mann_kendall_results.csv"


def main() -> None:
    OUTPUTS_DIR.mkdir(exist_ok=True)
    df = load_data()
    year_min = int(df["Year"].min())
    year_max = int(df["Year"].max())

    rows = []
    for station in RAINFALL_COLUMNS:
        series = get_series(df, station, year_min, year_max, "annual")
        r = run_mann_kendall(series, alpha=0.05)
        rows.append({
            "Station": station,
            "trend": r.get("trend") or "—",
            "p_value": r.get("p") if r.get("p") is not None else None,
            "Sen_slope_mm_per_year": r.get("slope") if r.get("slope") is not None else None,
            "significant_alpha_005": r.get("significant", False),
            "n": r.get("n", 0),
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"Mann–Kendall results saved to: {OUTPUT_CSV}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
