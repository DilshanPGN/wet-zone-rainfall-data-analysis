"""
Innovative Trend Analysis (ITA) for rainfall time series.

Method: Split the series into first and second halves, plot second-half values
against first-half values. Points above the 1:1 line indicate increase;
points below indicate decrease. No distributional assumption required.

Reference: Şen, Z. (2012). Innovative trend analysis methodology.
Journal of Hydrologic Engineering, 17(9), 1042-1046.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib
matplotlib.use("Agg")  # headless for Streamlit/export
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import pymannkendall as mk
except ImportError:
    mk = None
try:
    import plotly.graph_objects as go
except ImportError:
    go = None

# -----------------------------------------------------------------------------
# Configuration (aligned with project)
# -----------------------------------------------------------------------------
DATASET_DIR = Path(__file__).resolve().parent / "dataset"
EXCEL_FILLED = DATASET_DIR / "Wet zone rainfall data_filled.xlsx"
EXCEL_ORIGINAL = DATASET_DIR / "Wet zone rainfall data.xlsx"
SHEET_NAME = "All"
RAINFALL_COLUMNS = [
    "Colombo", "Galle", "Nuwara Eliya", "Rathnapura", "Maliboda", "Deniyaya",
]


def load_data(path: Path | None = None) -> pd.DataFrame:
    """Load Excel sheet into DataFrame. Prefer filled dataset if available."""
    if path is not None:
        return pd.read_excel(path, sheet_name=SHEET_NAME)
    p = EXCEL_FILLED if EXCEL_FILLED.exists() else EXCEL_ORIGINAL
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found in {DATASET_DIR}")
    return pd.read_excel(p, sheet_name=SHEET_NAME)


def get_series(
    df: pd.DataFrame,
    station: str,
    year_min: int,
    year_max: int,
    interval: Literal["annual", "monthly", "daily"],
) -> pd.Series:
    """
    Extract rainfall series for one station over the given year range.

    - annual: one value per year (annual total rainfall).
    - monthly: chronological monthly values (one value per month across years).
    - daily: chronological daily values (one value per day); uses Date column
      if present, else falls back to row order within Year–Month.

    Returns a Series (index 0,1,... for ITA; original index kept for annual).
    """
    if station not in df.columns:
        raise ValueError(f"Station '{station}' not in data. Choose from {list(df.columns)}.")
    df = df[(df["Year"] >= year_min) & (df["Year"] <= year_max)].copy()
    if df.empty:
        return pd.Series(dtype=float)

    if interval == "annual":
        annual = df.groupby("Year")[station].sum()
        annual = annual.dropna()
        return annual

    # daily or monthly: chronological order
    if interval == "daily" and "Date" in df.columns:
        df = df.sort_values(["Year", "Month", "Date"])
    else:
        df = df.sort_values(["Year", "Month"])
    vals = df[station].dropna()
    vals.index = range(len(vals))
    return vals


def ita_split(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """
    Split the series into first half and second half for ITA.

    Returns (first_half_values, second_half_values). Lengths are equal
    (n // 2 each for even n; second half gets the middle point for odd n).
    """
    vals = series.dropna().values
    n = len(vals)
    if n < 2:
        return np.array([]), np.array([])
    half = n // 2
    first_half = vals[:half]
    second_half = vals[half:]
    return first_half, second_half


def sens_slope(series: pd.Series) -> tuple[float, float]:
    """
    Sen's slope (median of pairwise slopes) and intercept estimate.

    For series index = 0,1,2,... (or years), slope = median((y_j - y_i)/(j - i)).
    Returns (slope, intercept) where intercept is median(y - slope * x).
    """
    y = series.dropna().values
    x = np.arange(len(y), dtype=float)
    n = len(y)
    if n < 2:
        return 0.0, 0.0
    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            if x[j] != x[i]:
                slopes.append((y[j] - y[i]) / (x[j] - x[i]))
    if not slopes:
        return 0.0, float(np.median(y))
    slope = float(np.median(slopes))
    intercept = float(np.median(y - slope * x))
    return slope, intercept


def run_mann_kendall(series: pd.Series, alpha: float = 0.05) -> dict:
    """
    Perform Mann–Kendall trend test on a time series.

    Returns dict with: trend, p, slope (Sen's slope), z, tau, significant (bool),
    and trend_direction (for display). Uses pymannkendall if available.
    """
    x = series.dropna().values
    n = len(x)
    result = {
        "trend": "no trend",
        "p": None,
        "slope": None,
        "z": None,
        "tau": None,
        "significant": False,
        "trend_direction": "no significant trend",
        "n": n,
    }
    if n < 3 or mk is None:
        if n >= 2 and mk is None:
            slope, _ = sens_slope(series)
            result["slope"] = slope
        return result
    try:
        out = mk.original_test(x)
        result["trend"] = getattr(out, "trend", "no trend")
        result["p"] = getattr(out, "p", None)
        result["slope"] = getattr(out, "slope", None)
        result["z"] = getattr(out, "z", None)
        result["tau"] = getattr(out, "Tau", None)
        p = result["p"]
        result["significant"] = p is not None and p < alpha
        if result["significant"]:
            result["trend_direction"] = (
                "significant increasing trend"
                if result["trend"] == "increasing"
                else "significant decreasing trend"
                if result["trend"] == "decreasing"
                else "significant trend"
            )
        else:
            result["trend_direction"] = "no significant trend"
    except Exception:
        slope, _ = sens_slope(series)
        result["slope"] = slope
    return result


def ita_summary_stats(
    first_half: np.ndarray,
    second_half: np.ndarray,
) -> dict:
    """Compute summary stats for ITA: % above/below 1:1 line, mean change."""
    if len(first_half) == 0 or len(second_half) == 0:
        return {"n": 0, "pct_above": 0.0, "pct_below": 0.0, "mean_change": 0.0}
    n = min(len(first_half), len(second_half))
    f, s = first_half[:n], second_half[:n]
    above = np.sum(s > f)
    below = np.sum(s < f)
    on_line = np.sum(s == f)
    pct_above = 100.0 * above / n
    pct_below = 100.0 * below / n
    mean_change = float(np.mean(s - f))
    return {
        "n": n,
        "pct_above": pct_above,
        "pct_below": pct_below,
        "pct_on_line": 100.0 * on_line / n,
        "mean_change": mean_change,
    }


def plot_ita(
    first_half: np.ndarray,
    second_half: np.ndarray,
    title: str = "Innovative Trend Analysis (ITA)",
    xlabel: str = "First half (chronological)",
    ylabel: str = "Second half (chronological)",
) -> plt.Figure:
    """
    Create publication-ready ITA scatter plot with 1:1 line.
    Sen's slope is reported in the UI; the plot shows only the 1:1 reference.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    n = min(len(first_half), len(second_half))
    if n == 0:
        ax.set_title(title)
        return fig
    x, y = first_half[:n], second_half[:n]
    ax.scatter(x, y, alpha=0.7, s=40, edgecolors="k", linewidths=0.5, zorder=2)
    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    margin = (hi - lo) * 0.05 if hi > lo else 1
    lo, hi = lo - margin, hi + margin
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.5, label="1:1 line (no trend)", zorder=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_ita_advanced(
    series: pd.Series,
    first_half: np.ndarray,
    second_half: np.ndarray,
    summary: dict,
    title_base: str,
    xlabel_ita: str,
    ylabel_ita: str,
    step_label: str,
) -> plt.Figure:
    """
    Advanced two-panel chart for analysis:
    - Top: time series of rainfall with Sen's trend line.
    - Bottom: ITA scatter with points colored by increase/decrease, 1:1 line,
      and statistics annotation box. Publication-ready styling.
    """
    fig, (ax_ts, ax_ita) = plt.subplots(2, 1, figsize=(10, 9), height_ratios=[1, 1.1])
    fig.suptitle(title_base, fontsize=14, fontweight="bold", y=1.02)

    n = min(len(first_half), len(second_half))
    slope = summary.get("sens_slope")
    intercept = summary.get("sens_intercept")

    # ---- Panel 1: Time series ----
    x_ts = np.arange(len(series))
    y_ts = series.values
    ax_ts.fill_between(x_ts, y_ts, alpha=0.3, color="steelblue")
    ax_ts.plot(x_ts, y_ts, color="steelblue", linewidth=0.8, label="Rainfall")
    if slope is not None and intercept is not None and len(series) >= 2:
        trend_line = slope * x_ts + intercept
        ax_ts.plot(x_ts, trend_line, color="coral", linewidth=2, linestyle="--", label=f"Sen's trend ({slope:.2f} {step_label})")
    ax_ts.set_ylabel("Rainfall (mm)", fontsize=11)
    ax_ts.set_xlabel(f"Time index ({step_label}s)", fontsize=11)
    ax_ts.set_title("Rainfall time series and trend", fontsize=12)
    ax_ts.legend(loc="upper right", fontsize=9)
    ax_ts.grid(True, alpha=0.4)
    ax_ts.set_xlim(0, len(series) - 1 if len(series) > 1 else 1)

    # ---- Panel 2: ITA scatter with colored points ----
    if n > 0:
        x_ita, y_ita = first_half[:n], second_half[:n]
        above = y_ita > x_ita
        below = y_ita < x_ita
        on_line = np.logical_and(~above, ~below)
        ax_ita.scatter(x_ita[above], y_ita[above], c="green", alpha=0.7, s=36, edgecolors="darkgreen", linewidths=0.6, label="Increase (above 1:1)", zorder=2)
        ax_ita.scatter(x_ita[below], y_ita[below], c="red", alpha=0.7, s=36, edgecolors="darkred", linewidths=0.6, label="Decrease (below 1:1)", zorder=2)
        if np.any(on_line):
            ax_ita.scatter(x_ita[on_line], y_ita[on_line], c="gray", alpha=0.7, s=36, edgecolors="black", linewidths=0.6, label="No change", zorder=2)
        lo = min(x_ita.min(), y_ita.min())
        hi = max(x_ita.max(), y_ita.max())
        margin = (hi - lo) * 0.06 if hi > lo else 1
        lo, hi = lo - margin, hi + margin
        ax_ita.plot([lo, hi], [lo, hi], "k--", lw=2, label="1:1 line (no trend)", zorder=1)
        ax_ita.set_xlim(lo, hi)
        ax_ita.set_ylim(lo, hi)
        ax_ita.set_xlabel(xlabel_ita + " (mm)", fontsize=11)
        ax_ita.set_ylabel(ylabel_ita + " (mm)", fontsize=11)
        ax_ita.set_title("Innovative Trend Analysis (ITA)", fontsize=12)
        ax_ita.legend(loc="upper left", fontsize=9)
        ax_ita.set_aspect("equal")
        ax_ita.grid(True, alpha=0.4)

        # Statistics box
        pct_above = summary.get("pct_above", 0)
        pct_below = summary.get("pct_below", 0)
        mean_change = summary.get("mean_change", 0)
        stats_text = (
            f"Points: {n}\n"
            f"Above 1:1: {pct_above:.1f}%\n"
            f"Below 1:1: {pct_below:.1f}%\n"
            f"Mean change: {mean_change:.2f} mm"
        )
        ax_ita.text(0.98, 0.02, stats_text, transform=ax_ita.transAxes, fontsize=9,
                    verticalalignment="bottom", horizontalalignment="right",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.9))
    else:
        ax_ita.set_title("Innovative Trend Analysis (ITA)", fontsize=12)
        ax_ita.text(0.5, 0.5, "Insufficient data", transform=ax_ita.transAxes, ha="center", va="center")

    fig.tight_layout()
    return fig


def run_ita(
    df: pd.DataFrame,
    station: str,
    year_min: int,
    year_max: int,
    interval: Literal["annual", "monthly", "daily"],
    use_advanced_chart: bool = True,
) -> tuple[pd.Series, np.ndarray, np.ndarray, dict, plt.Figure]:
    """
    Run full ITA pipeline: get series, split, stats, plot.

    Returns (series, first_half, second_half, summary_dict, figure).
    If use_advanced_chart is True, returns a two-panel (time series + ITA) figure.
    """
    series = get_series(df, station, year_min, year_max, interval)
    first_half, second_half = ita_split(series)
    summary = ita_summary_stats(first_half, second_half)
    slope, intercept = sens_slope(series) if len(series) >= 2 else (None, None)
    summary["sens_slope"] = slope
    summary["sens_intercept"] = intercept

    step_label = "year" if interval == "annual" else ("day" if interval == "daily" else "step")
    xlabel_ita = f"First half ({step_label})"
    ylabel_ita = f"Second half ({step_label})"
    title_base = f"ITA — {station} ({year_min}–{year_max}, {interval})"

    if use_advanced_chart:
        fig = plot_ita_advanced(
            series=series,
            first_half=first_half,
            second_half=second_half,
            summary=summary,
            title_base=title_base,
            xlabel_ita=xlabel_ita,
            ylabel_ita=ylabel_ita,
            step_label=step_label,
        )
    else:
        fig = plot_ita(
            first_half,
            second_half,
            title=title_base,
            xlabel=xlabel_ita,
            ylabel=ylabel_ita,
        )
    return series, first_half, second_half, summary, fig


def run_ita_from_series(
    series: pd.Series,
    station: str,
    interval: Literal["annual", "monthly", "daily"],
    title_suffix: str = "selected range",
    use_advanced_chart: bool = True,
) -> tuple[pd.Series, np.ndarray, np.ndarray, dict, plt.Figure]:
    """
    Run ITA on an already-extracted series (e.g. a subset from box selection).

    Returns (series, first_half, second_half, summary_dict, figure).
    """
    series = series.dropna()
    if len(series) < 2:
        empty_summary = {
            "n": 0, "pct_above": 0.0, "pct_below": 0.0, "mean_change": 0.0,
            "sens_slope": None, "sens_intercept": None,
        }
        fig = plot_ita_advanced(
            series=series, first_half=np.array([]), second_half=np.array([]),
            summary=empty_summary, title_base="", xlabel_ita="", ylabel_ita="",
            step_label="step",
        )
        return series, np.array([]), np.array([]), empty_summary, fig
    first_half, second_half = ita_split(series)
    summary = ita_summary_stats(first_half, second_half)
    slope, intercept = sens_slope(series)
    summary["sens_slope"] = slope
    summary["sens_intercept"] = intercept
    step_label = "year" if interval == "annual" else ("day" if interval == "daily" else "step")
    xlabel_ita = f"First half ({step_label})"
    ylabel_ita = f"Second half ({step_label})"
    title_base = f"ITA — {station} ({title_suffix}, {interval})"
    fig = plot_ita_advanced(
        series=series,
        first_half=first_half,
        second_half=second_half,
        summary=summary,
        title_base=title_base,
        xlabel_ita=xlabel_ita,
        ylabel_ita=ylabel_ita,
        step_label=step_label,
    )
    return series, first_half, second_half, summary, fig


def mk_timeseries_plotly(
    series: pd.Series,
    mk_result: dict,
    x_values: list | np.ndarray | None = None,
    title: str = "Mann–Kendall trend analysis",
    y_label: str = "Rainfall (mm)",
    x_label: str = "Year",
) -> "go.Figure | None":
    """
    Build a Plotly figure: time series (line + markers) with Sen's trend line overlaid.
    Like the reference image: temporal trend with clear linear trend line.
    Returns None if plotly is not available.
    """
    if go is None:
        return None
    n = len(series)
    if n == 0:
        return None
    y_vals = series.values.tolist()
    x_vals = (x_values if x_values is not None else list(range(n)))
    if len(x_vals) != n:
        x_vals = list(range(n))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="lines+markers",
            name="Rainfall",
            line=dict(color="steelblue", width=2),
            marker=dict(size=6),
        )
    )
    slope = mk_result.get("slope")
    if slope is not None and n >= 2:
        _, intercept = sens_slope(series)
        trend_line = slope * np.arange(n) + intercept
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=trend_line.tolist(),
                mode="lines",
                name=f"Sen's trend (slope ≈ {slope:.2f})",
                line=dict(color="coral", width=3, dash="dash"),
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, b=50),
    )
    return fig


def get_all_stations_mk(
    df: pd.DataFrame,
    year_min: int,
    year_max: int,
) -> pd.DataFrame:
    """
    Run Mann–Kendall test for all stations (annual totals) and return a DataFrame
    with Station, trend, p_value, slope, significant, n.
    """
    rows = []
    for stn in RAINFALL_COLUMNS:
        series = get_series(df, stn, year_min, year_max, "annual")
        r = run_mann_kendall(series, alpha=0.05)
        rows.append({
            "Station": stn,
            "trend": r.get("trend") or "no trend",
            "p_value": r.get("p"),
            "slope": r.get("slope"),
            "significant": r.get("significant", False),
            "n": r.get("n", 0),
        })
    return pd.DataFrame(rows)


def mk_all_stations_bar_plotly(mk_df: pd.DataFrame) -> "go.Figure | None":
    """
    Bar chart: Sen's slope by station, colored by trend (increasing=red/orange,
    decreasing=blue, no trend=gray).
    """
    if go is None or mk_df.empty:
        return None
    stations = mk_df["Station"].tolist()
    slopes = mk_df["slope"].fillna(0).tolist()
    trend_colors = []
    for t in mk_df["trend"]:
        if t == "increasing":
            trend_colors.append("rgba(220, 53, 69, 0.8)")
        elif t == "decreasing":
            trend_colors.append("rgba(13, 110, 253, 0.8)")
        else:
            trend_colors.append("rgba(108, 117, 125, 0.8)")
    fig = go.Figure(
        data=[go.Bar(x=stations, y=slopes, marker_color=trend_colors, name="Sen's slope (mm/yr)")]
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title="Trend by station (Mann–Kendall Sen's slope)",
        xaxis_title="Station",
        yaxis_title="Sen's slope (mm/year)",
        height=380,
        margin=dict(t=50, b=50),
        showlegend=False,
    )
    return fig


if __name__ == "__main__":
    df = load_data()
    year_min = int(df["Year"].min())
    year_max = int(df["Year"].max())
    station = RAINFALL_COLUMNS[0]
    series, f, s, summary, fig = run_ita(df, station, year_min, year_max, "annual")
    print("Summary:", summary)
    fig.savefig("ita_demo.png", dpi=150)
    plt.close(fig)
    print("Saved ita_demo.png")
