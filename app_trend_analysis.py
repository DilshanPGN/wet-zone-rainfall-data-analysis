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

import plotly.graph_objects as go

from ita import (
    load_data,
    RAINFALL_COLUMNS,
    run_ita,
    run_ita_from_series,
    run_mann_kendall,
    mk_timeseries_plotly,
    get_all_stations_mk,
    mk_all_stations_bar_plotly,
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

tab_single, tab_all = st.tabs(["Single station analysis", "Trend by station (all stations)"])

with tab_single:
    # -----------------------------------------------------------------------------
    # Run ITA (progress bar while computing — trend can take a moment to appear)
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
    series_full, first_half_full, second_half_full, summary_full, fig_full = ita_result[0]
    
    if summary_full["n"] == 0:
        st.warning("No data in the selected range. Widen the year range or choose another station.")
        st.stop()
    
    # -----------------------------------------------------------------------------
    # Selection state and interactive time series (drag to select range)
    # -----------------------------------------------------------------------------
    if "ita_selection" not in st.session_state:
        st.session_state.ita_selection = None
    if "ita_chart_key" not in st.session_state:
        st.session_state.ita_chart_key = 0
    
    # Plotly time series for box-select (drag to select a range)
    x_ts = list(range(len(series_full)))
    y_ts = series_full.values.tolist()
    fig_plotly = go.Figure()
    fig_plotly.add_trace(
        go.Scatter(
            x=x_ts,
            y=y_ts,
            mode="lines+markers",
            name="Rainfall",
            line=dict(color="steelblue", width=1.5),
            marker=dict(size=4),
        )
    )
    fig_plotly.update_layout(
        title="Time series — drag a box to select a range (results below update)",
        xaxis_title="Time index",
        yaxis_title="Rainfall (mm)",
        height=350,
        dragmode="select",
        margin=dict(t=50, b=50),
    )
    fig_plotly.update_xaxes(rangeslider_visible=False)
    
    event = st.plotly_chart(
        fig_plotly,
        key=f"ita_ts_select_{st.session_state.ita_chart_key}",
        on_select="rerun",
        selection_mode=("box",),
        use_container_width=True,
    )
    
    # Update ita_selection from chart selection (point_indices = time indices)
    if event and getattr(event, "selection", None):
        sel = event.selection
        indices = sel.get("point_indices", []) if isinstance(sel, dict) else getattr(sel, "point_indices", [])
        if indices:
            st.session_state.ita_selection = (min(indices), max(indices))
    
    # Reset button (new chart key clears the drawn box on the Plotly chart)
    if st.button("Reset to full range", type="secondary"):
        st.session_state.ita_selection = None
        st.session_state.ita_chart_key = (st.session_state.ita_chart_key + 1) % 10000
        st.rerun()
    
    # Use subset or full results
    if st.session_state.ita_selection is not None:
        start_idx, end_idx = st.session_state.ita_selection
        series_sub = series_full.iloc[start_idx : end_idx + 1].reset_index(drop=True)
        if len(series_sub) >= 2:
            series, first_half, second_half, summary, fig = run_ita_from_series(
                series_sub,
                station,
                interval,
                title_suffix=f"indices {start_idx}–{end_idx}",
            )
            st.info(f"Showing **selected range**: indices {start_idx}–{end_idx} ({len(series_sub)} points). Results, AI analysis, and export reflect this subset. Use *Reset to full range* to clear.")
        else:
            series, first_half, second_half, summary, fig = series_full, first_half_full, second_half_full, summary_full, fig_full
            st.warning("Selected range too short (need ≥2 points). Showing full range.")
    else:
        series, first_half, second_half, summary, fig = series_full, first_half_full, second_half_full, summary_full, fig_full
    
    # -----------------------------------------------------------------------------
    # Main: chart and metrics
    # -----------------------------------------------------------------------------
    st.subheader("Results (full range or selected range)")
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
    # Mann–Kendall trend test
    # -----------------------------------------------------------------------------
    mk_result = run_mann_kendall(series, alpha=0.05)
    st.subheader("Mann–Kendall trend test")
    mk_col1, mk_col2, mk_col3 = st.columns(3)
    with mk_col1:
        p_val = mk_result.get("p")
        st.metric("p-value", f"{p_val:.4f}" if p_val is not None else "—")
    with mk_col2:
        st.metric("Trend (MK)", mk_result.get("trend", "—") or "—")
    with mk_col3:
        mk_slope = mk_result.get("slope")
        if mk_slope is not None:
            unit = "mm/year" if interval == "annual" else ("mm/day" if interval == "daily" else "mm/step")
            st.metric("Sen's slope (MK)", f"{mk_slope:.4f} {unit}")
        else:
            st.metric("Sen's slope (MK)", "—")
    if mk_result.get("significant") is True:
        st.success(f"**Statistically significant** at α = 0.05: {mk_result.get('trend_direction', '')}.")
    elif mk_result.get("p") is not None:
        st.caption("Not significant at α = 0.05 (p ≥ 0.05).")
    
    # -----------------------------------------------------------------------------
    # Mann–Kendall trend chart (time series + trend line, like reference image)
    # -----------------------------------------------------------------------------
    st.subheader("Mann–Kendall trend: time series and trend line")
    if interval == "annual" and hasattr(series.index, "tolist") and len(series.index) == len(series):
        try:
            x_vals_mk = series.index.tolist()
            x_label_mk = "Year"
        except Exception:
            x_vals_mk = list(range(len(series)))
            x_label_mk = "Time index"
    else:
        x_vals_mk = list(range(len(series)))
        x_label_mk = "Time index"
    fig_mk_ts = mk_timeseries_plotly(
        series,
        mk_result,
        x_values=x_vals_mk,
        title=f"Mann–Kendall trend — {station} (rainfall + Sen's trend line)",
        y_label="Rainfall (mm)",
        x_label=x_label_mk,
    )
    if fig_mk_ts is not None:
        st.plotly_chart(fig_mk_ts, use_container_width=True)
    
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
    range_desc = f"indices {st.session_state.ita_selection[0]}–{st.session_state.ita_selection[1]}" if st.session_state.ita_selection else f"{year_min}–{year_max}"
    mk_trend = mk_result.get("trend") or "no trend"
    mk_p = mk_result.get("p")
    mk_sig = "significant" if mk_result.get("significant") else "not significant"
    mk_line = f"Mann–Kendall: **{mk_trend}** (p = {mk_p:.4f}); trend is **{mk_sig}** at α = 0.05." if mk_p is not None else "Mann–Kendall: not computed (insufficient data or package missing)."
    interpretation = f"""
    **ITA interpretation:** {trend_ita.capitalize()}. {reason}
    
    **Sen's slope:** {trend_sen} (slope ≈ {slope:.2f} {unit_slope}).  
    **Mean change** between first and second half: {mean_change:.2f} mm.
    
    **Mann–Kendall:** {mk_line}
    
    **Conclusion:** For **{station}** ({range_desc}, {interval} scale), the series shows a **{trend_ita}** pattern. The Mann–Kendall test indicates the trend is **{mk_sig}**.
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
    
    suffix = f"_selected_{st.session_state.ita_selection[0]}_{st.session_state.ita_selection[1]}" if st.session_state.ita_selection else ""
    col_png, col_pdf, _ = st.columns(3)
    with col_png:
        st.download_button(
            label="Download as PNG",
            data=png_bytes,
            file_name=f"ita_{station}_{year_min}_{year_max}_{interval}{suffix}.png",
            mime="image/png",
        )
    with col_pdf:
        st.download_button(
            label="Download as PDF",
            data=pdf_bytes,
            file_name=f"ita_{station}_{year_min}_{year_max}_{interval}{suffix}.pdf",
            mime="application/pdf",
        )
    
    # Optional: close figure to free memory (Streamlit may keep it)
    import matplotlib.pyplot as plt
    plt.close(fig)

with tab_all:
    st.subheader("Trend by station (all stations)")
    st.caption("Mann–Kendall trend for all wet-zone stations (annual totals). Uses the year range from the sidebar.")
    all_bar = st.progress(0, text="Computing trend for all stations...")
    mk_all_result = [None]

    def run_all_stations_mk_thread():
        try:
            mk_all_result[0] = get_all_stations_mk(df, int(year_min), int(year_max))
        except Exception:
            mk_all_result[0] = None

    thread_all = threading.Thread(target=run_all_stations_mk_thread)
    thread_all.start()
    p_all = 0
    while thread_all.is_alive():
        p_all = min(p_all + 0.03, 0.92)
        all_bar.progress(p_all, text="Computing trend for all stations...")
        time.sleep(0.08)
    thread_all.join()
    all_bar.progress(1.0, text="Done.")
    time.sleep(0.2)
    all_bar.empty()
    mk_all_df = mk_all_result[0]
    if mk_all_df is None or mk_all_df.empty:
        st.warning("Could not compute trend for all stations. Check the year range and data.")
    else:
        fig_bar = mk_all_stations_bar_plotly(mk_all_df)
        if fig_bar is not None:
            st.plotly_chart(fig_bar, use_container_width=True)
    st.caption("Sen's slope (mm/year) by station; red = increasing, blue = decreasing, gray = no significant trend.")
