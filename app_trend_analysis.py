"""
Streamlit UI for Innovative Trend Analysis (ITA).

Features: city (station) selection, date range, time interval, ITA chart,
export PNG/PDF, and AI Analysis section (rule-based interpretation).
"""
from pathlib import Path
from io import BytesIO
import threading
import time

import streamlit as st

from ita import (
    load_data,
    RAINFALL_COLUMNS,
    run_ita,
)

# -----------------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Wet Zone Rainfall — ITA Trend Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Innovative Trend Analysis (ITA)")
st.caption("Wet zone rainfall data — select station, date range, and interval to view trends and export charts.")

# -----------------------------------------------------------------------------
# Load data once (with progress bar)
# -----------------------------------------------------------------------------
@st.cache_data
def get_df():
    try:
        return load_data()
    except FileNotFoundError as e:
        st.error(str(e))
        return None

progress_bar = st.progress(0, text="Loading data...")
with st.spinner("Reading dataset..."):
    df = get_df()
progress_bar.progress(1.0, text="Data loaded.")
progress_bar.empty()

if df is None:
    st.stop()

year_min_all = int(df["Year"].min())
year_max_all = int(df["Year"].max())

# -----------------------------------------------------------------------------
# Sidebar: controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")

    station = st.selectbox(
        "Station (city)",
        options=RAINFALL_COLUMNS,
        index=0,
        help="Rainfall station to analyze.",
    )

    col1, col2 = st.columns(2)
    with col1:
        year_min = st.number_input(
            "Year from",
            min_value=year_min_all,
            max_value=year_max_all,
            value=year_min_all,
            step=1,
        )
    with col2:
        year_max = st.number_input(
            "Year to",
            min_value=year_min_all,
            max_value=year_max_all,
            value=year_max_all,
            step=1,
        )
    if year_min > year_max:
        st.warning("Year from must be ≤ Year to. Adjusting.")
        year_min, year_max = min(year_min, year_max), max(year_min, year_max)

    interval = st.radio(
        "Time interval",
        options=["annual", "monthly", "daily"],
        index=0,
        format_func=lambda x: {
            "annual": "Annual (one value per year)",
            "monthly": "Monthly (chronological)",
            "daily": "Daily (chronological, per day)",
        }[x],
        help="Annual: yearly totals. Monthly: all months in order. Daily: one value per day (uses Date column).",
    )

# -----------------------------------------------------------------------------
# Run ITA (with filling progress bar while computing)
# -----------------------------------------------------------------------------
compute_bar = st.progress(0, text=f"Computing trend analysis ({interval})...")
ita_result = [None]
ita_error = [None]

def run_ita_thread():
    try:
        ita_result[0] = run_ita(
            df, station, int(year_min), int(year_max), interval
        )
    except Exception as e:
        ita_error[0] = e

thread = threading.Thread(target=run_ita_thread)
thread.start()
p = 0
while thread.is_alive():
    p = min(p + 0.03, 0.92)
    compute_bar.progress(p, text=f"Computing trend analysis ({interval})...")
    time.sleep(0.08)
thread.join()
compute_bar.progress(1.0, text="Done.")
time.sleep(0.2)
compute_bar.empty()

if ita_error[0] is not None:
    st.error(f"ITA failed: {ita_error[0]}")
    st.stop()
series, first_half, second_half, summary, fig = ita_result[0]

if summary["n"] == 0:
    st.warning("No data in the selected range. Widen the year range or choose another station.")
    st.stop()

# -----------------------------------------------------------------------------
# Main: chart and metrics
# -----------------------------------------------------------------------------
col_chart, col_metrics = st.columns([2, 1])

with col_chart:
    st.pyplot(fig)

with col_metrics:
    st.metric("Points (half-series)", summary["n"])
    st.metric("% above 1:1 line (increase)", f"{summary['pct_above']:.1f}%")
    st.metric("% below 1:1 line (decrease)", f"{summary['pct_below']:.1f}%")
    st.metric("Mean change (2nd − 1st half)", f"{summary['mean_change']:.2f} mm")
    slope = summary.get("sens_slope")
    if slope is not None:
        unit = "mm/year" if interval == "annual" else ("mm/day" if interval == "daily" else "mm/step")
        st.metric("Sen's slope", f"{slope:.2f} {unit}")

# -----------------------------------------------------------------------------
# AI Analysis section (rule-based interpretation)
# -----------------------------------------------------------------------------
st.subheader("AI Analysis")
slope = summary.get("sens_slope") or 0.0
pct_above = summary["pct_above"]
pct_below = summary["pct_below"]
mean_change = summary["mean_change"]

if pct_above > 55:
    trend_ita = "increasing"
    reason = f"Most points ({pct_above:.0f}%) lie above the 1:1 line: second-half values are higher than first-half."
elif pct_below > 55:
    trend_ita = "decreasing"
    reason = f"Most points ({pct_below:.0f}%) lie below the 1:1 line: second-half values are lower than first-half."
else:
    trend_ita = "no clear trend or mixed"
    reason = f"Points are roughly balanced above ({pct_above:.0f}%) and below ({pct_below:.0f}%) the 1:1 line."

if slope > 0.5:
    trend_sen = "positive (increasing over time)"
elif slope < -0.5:
    trend_sen = "negative (decreasing over time)"
else:
    trend_sen = "near zero (stable)"

unit_slope = "mm/year" if interval == "annual" else ("mm/day" if interval == "daily" else "mm/step")
interpretation = f"""
**ITA interpretation:** {trend_ita.capitalize()}. {reason}

**Sen's slope:** {trend_sen} (slope ≈ {slope:.2f} {unit_slope}).  
**Mean change** between first and second half: {mean_change:.2f} mm.

**Conclusion:** For **{station}** over {year_min}–{year_max} ({interval} scale), the series shows a **{trend_ita}** pattern. Use this together with Mann–Kendall (if run separately) for significance.
"""
st.markdown(interpretation)

# -----------------------------------------------------------------------------
# Export: PNG / PDF
# -----------------------------------------------------------------------------
st.divider()
st.subheader("Export chart")

buf_png = BytesIO()
buf_pdf = BytesIO()
fig.savefig(buf_png, format="png", dpi=150, bbox_inches="tight")
fig.savefig(buf_pdf, format="pdf", bbox_inches="tight")
png_bytes = buf_png.getvalue()
pdf_bytes = buf_pdf.getvalue()

col_png, col_pdf, _ = st.columns(3)
with col_png:
    st.download_button(
        label="Download as PNG",
        data=png_bytes,
        file_name=f"ita_{station}_{year_min}_{year_max}_{interval}.png",
        mime="image/png",
    )
with col_pdf:
    st.download_button(
        label="Download as PDF",
        data=pdf_bytes,
        file_name=f"ita_{station}_{year_min}_{year_max}_{interval}.pdf",
        mime="application/pdf",
    )

# Optional: close figure to free memory (Streamlit may keep it)
import matplotlib.pyplot as plt
plt.close(fig)
