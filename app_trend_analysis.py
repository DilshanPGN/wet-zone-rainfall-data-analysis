"""
Streamlit UI for Innovative Trend Analysis (ITA).

Features: city (station) selection, date range, time interval, ITA chart,
export PNG/PDF, and AI Analysis section (rule-based interpretation).
"""
from pathlib import Path
from io import BytesIO
import threading
import time
import zipfile

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
    export_timeseries_trend_csv,
    export_ita_scatter_csv,
    export_trend_summary_csv,
)

# High-quality export: DPI for matplotlib, scale multiplier for Plotly (2 = 2× resolution)
EXPORT_DPI = 300
EXPORT_PLOTLY_SCALE = 2


def export_plotly_high_quality(fig, format: str = "png") -> bytes | None:
    """Export Plotly figure to PNG or PDF at high resolution. Requires kaleido."""
    if fig is None:
        return None
    try:
        buf = BytesIO()
        fig.write_image(buf, format=format, scale=EXPORT_PLOTLY_SCALE, engine="kaleido")
        return buf.getvalue()
    except Exception:
        return None


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
# Data loading (cached; load only when a section needs it or on Export button)
# -----------------------------------------------------------------------------
@st.cache_data
def get_df():
    try:
        return load_data()
    except FileNotFoundError as e:
        st.error(str(e))
        return None


def _ensure_year_range_in_session(df):
    if df is not None and not df.empty:
        st.session_state["year_min_all"] = int(df["Year"].min())
        st.session_state["year_max_all"] = int(df["Year"].max())


# Default year range until data is loaded (e.g. when only Export tab is open)
if "year_min_all" not in st.session_state:
    st.session_state["year_min_all"] = 1990
if "year_max_all" not in st.session_state:
    st.session_state["year_max_all"] = 2021

year_min_all = st.session_state["year_min_all"]
year_max_all = st.session_state["year_max_all"]

# -----------------------------------------------------------------------------
# Sidebar: controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")

    station = st.selectbox(
        "Station (city)",
        options=["All"] + list(RAINFALL_COLUMNS),
        index=0,
        help="Rainfall station to analyze. Use 'All' in Export all charts to export every station.",
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

# Section selector: only the selected section's code runs (no load/calc for others)
section = st.radio(
    "Section",
    ["Single station analysis", "Trend by station (all stations)", "Export all charts"],
    index=0,
    key="main_section",
    horizontal=True,
    label_visibility="collapsed",
)
# Style like tabs: use columns to show three options
st.markdown("---")

if section == "Single station analysis":
    # Load data only when this section is selected
    progress_bar = st.progress(0, text="Loading data...")
    with st.spinner("Reading dataset..."):
        df = get_df()
    progress_bar.progress(0.5, text="Processing dataset...")
    time.sleep(0.1)
    progress_bar.progress(1.0, text="Data loaded.")
    time.sleep(0.2)
    progress_bar.empty()
    if df is None:
        st.stop()
    _ensure_year_range_in_session(df)
    year_min_all = int(df["Year"].min())
    year_max_all = int(df["Year"].max())

elif section == "Trend by station (all stations)":
    # Load data only when this section is selected
    progress_bar = st.progress(0, text="Loading data...")
    with st.spinner("Reading dataset..."):
        df = get_df()
    progress_bar.progress(0.5, text="Processing dataset...")
    time.sleep(0.1)
    progress_bar.progress(1.0, text="Data loaded.")
    time.sleep(0.2)
    progress_bar.empty()
    if df is None:
        st.stop()
    _ensure_year_range_in_session(df)
    year_min_all = int(df["Year"].min())
    year_max_all = int(df["Year"].max())

else:
    # Export all charts: no data load here; df will be loaded when user clicks Export
    df = None

if section == "Single station analysis" and df is not None:
    if station == "All":
        st.info("Select a specific station above for single-station analysis. Use the **Export all charts** section to export charts for every station.")
        st.stop()
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
    series_full, first_half_full, second_half_full, summary_full, fig_ts_full, fig_ita_full = ita_result[0]
    
    if summary_full["n"] == 0:
        st.warning("No data in the selected range. Widen the year range or choose another station.")
        st.stop()

    # Progress bar for rendering single-station results
    render_bar = st.progress(0, text="Preparing time series chart...")
    
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
            line=dict(color="steelblue", width=0.6),
            marker=dict(size=1, color="black", line=dict(width=0.2, color="black")),
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
    render_bar.progress(0.1, text="Time series chart ready. Preparing selection...")

    event = st.plotly_chart(
        fig_plotly,
        key=f"ita_ts_select_{st.session_state.ita_chart_key}",
        on_select="rerun",
        selection_mode=("box",),
        use_container_width=True,
    )
    # Export: Rainfall time series (high-quality PNG/PDF)
    ts_png = export_plotly_high_quality(fig_plotly, "png")
    ts_pdf = export_plotly_high_quality(fig_plotly, "pdf")
    if ts_png is not None or ts_pdf is not None:
        ex_ts_1, ex_ts_2, _ = st.columns(3)
        with ex_ts_1:
            if ts_png is not None:
                st.download_button("Export: Rainfall time series (PNG)", data=ts_png, file_name=f"rainfall_timeseries_{station}_{year_min}_{year_max}.png", mime="image/png", key="export_ts_png")
        with ex_ts_2:
            if ts_pdf is not None:
                st.download_button("Export: Rainfall time series (PDF)", data=ts_pdf, file_name=f"rainfall_timeseries_{station}_{year_min}_{year_max}.pdf", mime="application/pdf", key="export_ts_pdf")
    else:
        st.caption("Chart export: install kaleido (`pip install kaleido`) for high-quality PNG/PDF.")
    
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
            sel_bar = st.progress(0, text="Updating analysis for selected range...")
            series, first_half, second_half, summary, fig_ts, fig_ita = run_ita_from_series(
                series_sub,
                station,
                interval,
                title_suffix=f"indices {start_idx}–{end_idx}",
            )
            sel_bar.progress(1.0, text="Done.")
            time.sleep(0.15)
            sel_bar.empty()
            st.info(f"Showing **selected range**: indices {start_idx}–{end_idx} ({len(series_sub)} points). Results, AI analysis, and export reflect this subset. Use *Reset to full range* to clear.")
        else:
            series, first_half, second_half, summary, fig_ts, fig_ita = series_full, first_half_full, second_half_full, summary_full, fig_ts_full, fig_ita_full
            st.warning("Selected range too short (need ≥2 points). Showing full range.")
    else:
        series, first_half, second_half, summary, fig_ts, fig_ita = series_full, first_half_full, second_half_full, summary_full, fig_ts_full, fig_ita_full

    render_bar.progress(0.25, text="Preparing ITA charts...")

    suffix_ita = f"_selected_{st.session_state.ita_selection[0]}_{st.session_state.ita_selection[1]}" if st.session_state.ita_selection else ""

    # -----------------------------------------------------------------------------
    # Rainfall time series and trend (Sen's trend line only)
    # -----------------------------------------------------------------------------
    if fig_ts is not None:
        st.subheader("Rainfall time series and trend")
        st.caption("Annual (or selected) rainfall with Sen's trend line.")
        col_ts, col_ts_metrics = st.columns([2, 1])
        with col_ts:
            st.pyplot(fig_ts)
        buf_ts_trend_png = BytesIO()
        buf_ts_trend_pdf = BytesIO()
        fig_ts.savefig(buf_ts_trend_png, format="png", dpi=EXPORT_DPI, bbox_inches="tight")
        fig_ts.savefig(buf_ts_trend_pdf, format="pdf", bbox_inches="tight")
        ex_ts_1, ex_ts_2, _ = st.columns(3)
        with ex_ts_1:
            st.download_button("Export: Rainfall time series and trend (PNG)", data=buf_ts_trend_png.getvalue(), file_name=f"rainfall_timeseries_trend_{station}_{year_min}_{year_max}{suffix_ita}.png", mime="image/png", key="export_ts_trend_png")
        with ex_ts_2:
            st.download_button("Export: Rainfall time series and trend (PDF)", data=buf_ts_trend_pdf.getvalue(), file_name=f"rainfall_timeseries_trend_{station}_{year_min}_{year_max}{suffix_ita}.pdf", mime="application/pdf", key="export_ts_trend_pdf")
        with col_ts_metrics:
            slope = summary.get("sens_slope")
            if slope is not None:
                unit = "mm/year" if interval == "annual" else ("mm/day" if interval == "daily" else "mm/step")
                st.metric("Sen's slope", f"{slope:.2f} {unit}")
    render_bar.progress(0.35, text="Preparing ITA scatter...")

    # -----------------------------------------------------------------------------
    # ITA trend (scatter only: first half vs second half)
    # -----------------------------------------------------------------------------
    st.subheader("ITA trend")
    st.caption("Innovative Trend Analysis: first half vs second half (full range or selected range).")
    col_chart, col_metrics = st.columns([2, 1])

    with col_chart:
        st.pyplot(fig_ita)
    buf_ita_png = BytesIO()
    buf_ita_pdf = BytesIO()
    fig_ita.savefig(buf_ita_png, format="png", dpi=EXPORT_DPI, bbox_inches="tight")
    fig_ita.savefig(buf_ita_pdf, format="pdf", bbox_inches="tight")
    ex_ita_1, ex_ita_2, _ = st.columns(3)
    with ex_ita_1:
        st.download_button("Export: ITA trend (PNG)", data=buf_ita_png.getvalue(), file_name=f"ita_trend_{station}_{year_min}_{year_max}{suffix_ita}.png", mime="image/png", key="export_ita_png")
    with ex_ita_2:
        st.download_button("Export: ITA trend (PDF)", data=buf_ita_pdf.getvalue(), file_name=f"ita_trend_{station}_{year_min}_{year_max}{suffix_ita}.pdf", mime="application/pdf", key="export_ita_pdf")

    with col_metrics:
        st.metric("Points (half-series)", summary["n"])
        st.metric("% above 1:1 line (increase)", f"{summary['pct_above']:.1f}%")
        st.metric("% below 1:1 line (decrease)", f"{summary['pct_below']:.1f}%")
        st.metric("Mean change (2nd − 1st half)", f"{summary['mean_change']:.2f} mm")
        slope = summary.get("sens_slope")
        if slope is not None:
            unit = "mm/year" if interval == "annual" else ("mm/day" if interval == "daily" else "mm/step")
            st.metric("Sen's slope", f"{slope:.2f} {unit}")
    render_bar.progress(0.45, text="Computing Mann–Kendall...")

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
    
    render_bar.progress(0.55, text="Building Mann–Kendall chart...")

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
        render_bar.progress(0.65, text="Preparing export options...")
        # Export: Mann–Kendall trend (high-quality PNG/PDF)
        mk_png = export_plotly_high_quality(fig_mk_ts, "png")
        mk_pdf = export_plotly_high_quality(fig_mk_ts, "pdf")
        if mk_png is not None or mk_pdf is not None:
            ex_mk_1, ex_mk_2, _ = st.columns(3)
            with ex_mk_1:
                if mk_png is not None:
                    st.download_button("Export: Mann–Kendall trend (PNG)", data=mk_png, file_name=f"mann_kendall_trend_{station}_{year_min}_{year_max}.png", mime="image/png", key="export_mk_png")
            with ex_mk_2:
                if mk_pdf is not None:
                    st.download_button("Export: Mann–Kendall trend (PDF)", data=mk_pdf, file_name=f"mann_kendall_trend_{station}_{year_min}_{year_max}.pdf", mime="application/pdf", key="export_mk_pdf")
    
    render_bar.progress(0.7, text="Preparing AI analysis...")

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
    render_bar.progress(0.8, text="Preparing CSV exports...")

    # -----------------------------------------------------------------------------
    # Export trend data (CSV) — for users to draw their own charts
    # -----------------------------------------------------------------------------
    st.divider()
    st.subheader("Export trend data (CSV)")
    st.caption("Download CSV data to draw your own charts in Excel, Python, R, etc.")
    time_col = "year" if interval == "annual" else "time_index"
    df_ts_trend = export_timeseries_trend_csv(series, summary, time_index_name=time_col)
    df_ita = export_ita_scatter_csv(first_half, second_half)
    df_summary = export_trend_summary_csv(summary, mk_result, station, year_min, year_max, interval)
    csv_ts = df_ts_trend.to_csv(index=False)
    csv_ita = df_ita.to_csv(index=False)
    csv_summary = df_summary.to_csv(index=False)
    csv_base = f"trend_data_{station}_{year_min}_{year_max}{suffix_ita}"
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button(
            "Time series + Sen's trend (CSV)",
            data=csv_ts,
            file_name=f"{csv_base}_timeseries_trend.csv",
            mime="text/csv",
            key="export_csv_timeseries",
        )
    with c2:
        st.download_button(
            "ITA scatter: first half vs second half (CSV)",
            data=csv_ita,
            file_name=f"{csv_base}_ita_scatter.csv",
            mime="text/csv",
            key="export_csv_ita",
        )
    with c3:
        st.download_button(
            "Summary statistics (CSV)",
            data=csv_summary,
            file_name=f"{csv_base}_summary.csv",
            mime="text/csv",
            key="export_csv_summary",
        )
    # Optional: ZIP of all CSVs
    zip_csv = BytesIO()
    with zipfile.ZipFile(zip_csv, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("timeseries_trend.csv", csv_ts)
        zf.writestr("ita_scatter.csv", csv_ita)
        zf.writestr("trend_summary.csv", csv_summary)
    st.download_button(
        "Download all trend CSVs (ZIP)",
        data=zip_csv.getvalue(),
        file_name=f"{csv_base}.zip",
        mime="application/zip",
        key="export_csv_zip",
    )

    render_bar.progress(0.9, text="Preparing combined export...")

    # -----------------------------------------------------------------------------
    # Combined export (all single-station charts in one ZIP)
    # -----------------------------------------------------------------------------
    st.divider()
    st.subheader("Combined export")
    st.caption("Download all charts from this tab in one ZIP (Rainfall time series, Rainfall time series + trend, ITA trend, Mann–Kendall trend).")
    ts_png_combined = export_plotly_high_quality(fig_plotly, "png")
    ts_pdf_combined = export_plotly_high_quality(fig_plotly, "pdf")
    ts_trend_png = buf_ts_trend_png.getvalue() if fig_ts is not None else None
    ts_trend_pdf = buf_ts_trend_pdf.getvalue() if fig_ts is not None else None
    ita_png_bytes = buf_ita_png.getvalue()
    ita_pdf_bytes = buf_ita_pdf.getvalue()
    mk_png_combined = export_plotly_high_quality(fig_mk_ts, "png") if fig_mk_ts is not None else None
    mk_pdf_combined = export_plotly_high_quality(fig_mk_ts, "pdf") if fig_mk_ts is not None else None
    suffix_zip = f"_selected_{st.session_state.ita_selection[0]}_{st.session_state.ita_selection[1]}" if st.session_state.ita_selection else ""
    zip_base = f"all_charts_{station}_{year_min}_{year_max}{suffix_zip}"
    if ts_png_combined and ita_png_bytes and mk_png_combined:
        zip_png = BytesIO()
        with zipfile.ZipFile(zip_png, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("1_rainfall_timeseries.png", ts_png_combined)
            if ts_trend_png is not None:
                zf.writestr("2_rainfall_timeseries_trend.png", ts_trend_png)
            zf.writestr("3_ita_trend.png", ita_png_bytes)
            zf.writestr("4_mann_kendall_trend.png", mk_png_combined)
        zip_png_bytes = zip_png.getvalue()
        st.download_button("Download all charts (PNG, ZIP)", data=zip_png_bytes, file_name=f"{zip_base}_png.zip", mime="application/zip", key="export_combined_png_zip")
    if ts_pdf_combined and ita_pdf_bytes and mk_pdf_combined:
        zip_pdf = BytesIO()
        with zipfile.ZipFile(zip_pdf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("1_rainfall_timeseries.pdf", ts_pdf_combined)
            if ts_trend_pdf is not None:
                zf.writestr("2_rainfall_timeseries_trend.pdf", ts_trend_pdf)
            zf.writestr("3_ita_trend.pdf", ita_pdf_bytes)
            zf.writestr("4_mann_kendall_trend.pdf", mk_pdf_combined)
        zip_pdf_bytes = zip_pdf.getvalue()
        st.download_button("Download all charts (PDF, ZIP)", data=zip_pdf_bytes, file_name=f"{zip_base}_pdf.zip", mime="application/zip", key="export_combined_pdf_zip")
    st.caption("Use the export buttons above each chart for high-quality PNG or PDF.")
    render_bar.progress(1.0, text="Done.")
    time.sleep(0.15)
    render_bar.empty()

    # Optional: close figures to free memory (Streamlit may keep them)
    import matplotlib.pyplot as plt
    if fig_ts is not None:
        plt.close(fig_ts)
    plt.close(fig_ita)

elif section == "Trend by station (all stations)" and df is not None:
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
    all_bar.progress(0.95, text="Rendering bar chart...")
    mk_all_df = mk_all_result[0]
    if mk_all_df is None or mk_all_df.empty:
        all_bar.progress(1.0, text="Done.")
        time.sleep(0.2)
        all_bar.empty()
        st.warning("Could not compute trend for all stations. Check the year range and data.")
    else:
        fig_bar = mk_all_stations_bar_plotly(mk_all_df)
        all_bar.progress(1.0, text="Done.")
        time.sleep(0.2)
        all_bar.empty()
        if fig_bar is not None:
            st.plotly_chart(fig_bar, use_container_width=True)
            # Export: Trend by station (all stations) — high-quality PNG/PDF
            all_png = export_plotly_high_quality(fig_bar, "png")
            all_pdf = export_plotly_high_quality(fig_bar, "pdf")
            if all_png is not None or all_pdf is not None:
                ex_all_1, ex_all_2, _ = st.columns(3)
                with ex_all_1:
                    if all_png is not None:
                        st.download_button("Export: Trend by station (PNG)", data=all_png, file_name=f"trend_by_station_{year_min}_{year_max}.png", mime="image/png", key="export_all_png")
                with ex_all_2:
                    if all_pdf is not None:
                        st.download_button("Export: Trend by station (PDF)", data=all_pdf, file_name=f"trend_by_station_{year_min}_{year_max}.pdf", mime="application/pdf", key="export_all_pdf")
    st.caption("Sen's slope (mm/year) by station; red = increasing, blue = decreasing, gray = no significant trend.")

else:
    # Export all charts: no data load or calculation until user presses the button
    st.subheader("Export all charts")
    st.caption("Generate all charts and CSV data for the **selected station** (or **All** for every station), **year range**, and **time interval** (sidebar). Data is loaded and computed only when you press the button below. The ZIP includes PNG charts and CSV files (timeseries_trend, ita_scatter, trend_summary) per station.")
    if st.button("Generate and download all charts (ZIP)", type="primary", key="export_all_btn"):
        import matplotlib.pyplot as plt
        prog = st.progress(0, text="Loading data...")
        status_placeholder = st.empty()
        try:
            status_placeholder.text("Loading dataset...")
            df_export = get_df()
            if df_export is None:
                status_placeholder.error("Could not load dataset. Check that the Excel file exists in the dataset folder.")
                prog.empty()
            else:
                _ensure_year_range_in_session(df_export)
                stations_to_export = list(RAINFALL_COLUMNS) if station == "All" else [station]
                n_stations = len(stations_to_export)
                zip_buf = BytesIO()
                with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    for idx, stn in enumerate(stations_to_export):
                        p_base = 0.1 + (0.75 * idx / n_stations) if n_stations else 0.1
                        status_placeholder.text(f"Processing {stn} ({idx + 1}/{n_stations})...")
                        prog.progress(p_base, text=f"Processing {stn}...")
                        series, first_half, second_half, summary, fig_ts, fig_ita = run_ita(
                            df_export, stn, int(year_min), int(year_max), interval
                        )
                        if summary["n"] == 0:
                            continue
                        prefix = f"{stn.replace(' ', '_')}_"
                        x_ts = list(range(len(series)))
                        y_ts = series.values.tolist()
                        fig_plotly = go.Figure()
                        fig_plotly.add_trace(
                            go.Scatter(
                                x=x_ts,
                                y=y_ts,
                                mode="lines+markers",
                                name="Rainfall",
                                line=dict(color="steelblue", width=0.6),
                                marker=dict(size=1, color="black", line=dict(width=0.2, color="black")),
                            )
                        )
                        fig_plotly.update_layout(
                            title=f"Time series — {stn} ({year_min}–{year_max}, {interval})",
                            xaxis_title="Time index",
                            yaxis_title="Rainfall (mm)",
                            height=350,
                            margin=dict(t=50, b=50),
                        )
                        mk_result = run_mann_kendall(series, alpha=0.05)
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
                        fig_mk = mk_timeseries_plotly(
                            series,
                            mk_result,
                            x_values=x_vals_mk,
                            title=f"Mann–Kendall trend — {stn} ({year_min}–{year_max}, {interval})",
                            y_label="Rainfall (mm)",
                            x_label=x_label_mk,
                        )
                        ts_png = export_plotly_high_quality(fig_plotly, "png")
                        buf_ts_trend = BytesIO()
                        buf_ita = BytesIO()
                        if fig_ts is not None:
                            fig_ts.savefig(buf_ts_trend, format="png", dpi=EXPORT_DPI, bbox_inches="tight")
                        fig_ita.savefig(buf_ita, format="png", dpi=EXPORT_DPI, bbox_inches="tight")
                        ts_trend_png = buf_ts_trend.getvalue() if fig_ts is not None else None
                        ita_png = buf_ita.getvalue()
                        mk_png = export_plotly_high_quality(fig_mk, "png") if fig_mk is not None else None
                        if fig_ts is not None:
                            plt.close(fig_ts)
                        plt.close(fig_ita)
                        if ts_png:
                            zf.writestr(f"{prefix}1_rainfall_timeseries.png", ts_png)
                        if ts_trend_png:
                            zf.writestr(f"{prefix}2_rainfall_timeseries_trend.png", ts_trend_png)
                        zf.writestr(f"{prefix}3_ita_trend.png", ita_png)
                        if mk_png:
                            zf.writestr(f"{prefix}4_mann_kendall_trend.png", mk_png)
                        # CSV data for this station (same as single-station export)
                        time_col = "year" if interval == "annual" else "time_index"
                        df_ts_trend = export_timeseries_trend_csv(series, summary, time_index_name=time_col)
                        df_ita = export_ita_scatter_csv(first_half, second_half)
                        df_summary = export_trend_summary_csv(summary, mk_result, stn, year_min, year_max, interval)
                        zf.writestr(f"{prefix}timeseries_trend.csv", df_ts_trend.to_csv(index=False))
                        zf.writestr(f"{prefix}ita_scatter.csv", df_ita.to_csv(index=False))
                        zf.writestr(f"{prefix}trend_summary.csv", df_summary.to_csv(index=False))
                prog.progress(0.9, text="Finalizing ZIP...")
                status_placeholder.empty()
                prog.progress(1.0, text="Done.")
                time.sleep(0.15)
                prog.empty()
                zip_label = "all_stations" if station == "All" else station.replace(" ", "_")
                zip_name = f"all_charts_{zip_label}_{year_min}_{year_max}_{interval}.zip"
                st.success("Charts generated. Download the ZIP below.")
                st.download_button(
                    "Download all charts (PNG, ZIP)",
                    data=zip_buf.getvalue(),
                    file_name=zip_name,
                    mime="application/zip",
                    key="download_export_all_zip",
                )
        except Exception as e:
            status_placeholder.error(f"Export failed: {e}")
            prog.empty()
