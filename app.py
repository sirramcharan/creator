import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

st.set_page_config(page_title="YouTube Creator Day Planner", layout="wide")


# ---------- HELPERS ----------

def debug_columns(df_raw: pd.DataFrame):
    st.write("Detected columns:", list(df_raw.columns))
    if "Video publish time" in df_raw.columns:
        st.write("Sample 'Video publish time' values:")
        st.write(df_raw["Video publish time"].head(10))


def load_and_clean(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Clean YouTube Studio export based on the example screenshot.
    """
    df = df_raw.copy()

    # 0. Show columns to help debugging
    debug_columns(df_raw)

    # 1. Drop 'Total' summary row
    if "Content" in df.columns:
        before = len(df)
        df = df[df["Content"] != "Total"]
        st.write(f"Rows after removing 'Total' row: {len(df)} (was {before})")

    # 2. Parse publish date from 'Video publish time'
    if "Video publish time" not in df.columns:
        st.error("Expected column 'Video publish time' not found. Check CSV headers.")
        return pd.DataFrame()

    # Try multiple date formats commonly used by YouTube exports
    date_series = df["Video publish time"].astype(str).str.strip()
    parsed = None
    tried_formats = ["%d-%b-%y", "%d-%b-%Y", "%Y-%m-%d", None]  # None = let pandas infer

    for fmt in tried_formats:
        if fmt is None:
            tmp = pd.to_datetime(date_series, errors="coerce")
            fmt_name = "infer"
        else:
            tmp = pd.to_datetime(date_series, format=fmt, errors="coerce")
            fmt_name = fmt
        valid = tmp.notna().sum()
        st.write(f"Trying date format {fmt_name}: parsed {valid} non-null dates")
        if valid > 0:
            parsed = tmp
            break

    if parsed is None or parsed.notna().sum() == 0:
        st.error("Could not parse any dates from 'Video publish time'.")
        return pd.DataFrame()

    df["publish_date"] = parsed.dt.date
    df["dow"] = parsed.dt.day_name()

    # 3. Rename numeric columns to simpler names
    rename_map = {
        "Views": "views",
        "Watch time (hours)": "watch_time_hours",
        "Impressions": "impressions",
        "Impressions click-through rate (%)": "ctr_percent",
        "Subscribers": "subscribers",
        "Likes": "likes",
        "Dislikes": "dislikes",
        "Shares": "shares",
        "Comments added": "comments",
    }
    df = df.rename(columns=rename_map)

    # 4. Convert numeric columns
    numeric_cols = [
        "views",
        "watch_time_hours",
        "impressions",
        "ctr_percent",
        "subscribers",
        "likes",
        "dislikes",
        "shares",
        "comments",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 5. Drop rows missing key fields
    before = len(df)
    df = df.dropna(subset=["publish_date"])
    st.write(f"Rows after dropping missing publish_date: {len(df)} (was {before})")

    if "views" in df.columns:
        before = len(df)
        df = df.dropna(subset=["views"])
        st.write(f"Rows after dropping missing views: {len(df)} (was {before})")
    else:
        st.error("Column 'Views' not found (or renamed incorrectly).")
        return pd.DataFrame()

    return df


def best_days(df: pd.DataFrame, top_k: int = 3):
    if "dow" not in df.columns:
        return None, None
    day_perf = (
        df.groupby("dow", dropna=False)["views"]
        .mean()
        .reset_index()
        .rename(columns={"views": "avg_views"})
        .dropna(subset=["avg_views"])
    )
    if day_perf.empty:
        return None, None
    day_perf_sorted = day_perf.sort_values("avg_views", ascending=False)
    return day_perf_sorted.head(top_k), day_perf_sorted


def build_day_model(df: pd.DataFrame):
    model = (
        df.groupby("dow", dropna=False)["views"]
        .mean()
        .reset_index()
        .rename(columns={"views": "expected_views"})
    )
    return model


def get_view_curve_template_days(n_days: int = 14) -> pd.DataFrame:
    x = np.arange(0, n_days)
    k = 0.6
    midpoint = 3
    base = 1 / (1 + np.exp(-k * (x - midpoint)))
    base = base / base.max()
    bump_center = 7
    bump_width = 2
    bump = 0.15 * np.exp(-((x - bump_center) ** 2) / (2 * bump_width**2))
    curve = base + bump
    curve = curve / curve.max()
    cum_curve = np.cumsum(curve)
    cum_curve = cum_curve / cum_curve.max()
    return pd.DataFrame({"day_since_publish": x, "cum_frac": cum_curve})


def predict_views_for_day(day_model, dow, total_curve, expected_factor=1.0):
    row = day_model[day_model["dow"] == dow]
    if row.empty:
        base_views = day_model["expected_views"].mean()
    else:
        base_views = row["expected_views"].iloc[0]
    total_views = base_views * expected_factor
    curve_df = total_curve.copy()
    curve_df["predicted_views"] = curve_df["cum_frac"] * total_views
    return total_views, curve_df


# ---------- APP ----------

st.title("YouTube Creator Day Planner (Thesis Prototype)")

uploaded = st.sidebar.file_uploader("Upload YouTube Studio CSV", type=["csv"])

if uploaded is None:
    st.info("Upload your CSV from the sidebar to begin.")
else:
    df_raw = pd.read_csv(uploaded)
    st.subheader("Raw data preview")
    st.dataframe(df_raw.head())

    df = load_and_clean(df_raw)

    if df.empty:
        st.error("Still no usable rows after cleaning. Check the debug info above (columns and sample dates).")
        st.stop()

    st.success(f"Loaded {len(df)} videos after cleaning.")

    day_model = build_day_model(df)
    template_curve_days = get_view_curve_template_days(n_days=14)

    tab_dash, tab_best, tab_planner, tab_scenarios = st.tabs(
        ["📊 Dashboard", "📅 Best Day", "🗓️ Post Planner", "🎯 Scenario Planner"]
    )

    # --- DASHBOARD ---
    with tab_dash:
        st.subheader("Channel Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Views", int(df["views"].sum()))
        col2.metric("Total Videos", df.shape[0])
        col3.metric("Avg Views / Video", f"{df['views'].mean():.0f}")

        st.markdown("### Views over time")
        ts = (
            df.groupby("publish_date")["views"]
            .sum()
            .reset_index()
            .sort_values("publish_date")
        )
        chart_ts = (
            alt.Chart(ts)
            .mark_line()
            .encode(
                x=alt.X("publish_date:T", title="Publish Date"),
                y=alt.Y("views:Q", title="Views"),
            )
        )
        st.altair_chart(chart_ts, use_container_width=True)

    # --- BEST DAY ---
    with tab_best:
        st.subheader("Best Day to Post")
        res_top3, full_day_perf = best_days(df, top_k=3)
        if res_top3 is None or res_top3.empty:
            st.warning("Not enough data to compute best days.")
        else:
            st.dataframe(res_top3)

    # --- POST PLANNER ---
    with tab_planner:
        st.subheader("Post Planner")
        dow_options = list(day_model["dow"].dropna().unique())
        if dow_options:
            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            dow_sorted = [d for d in day_order if d in dow_options]
            dow_choice = st.selectbox("Planned day of week", dow_sorted)
            factor = st.slider("Multiplier vs typical video on this day", 0.5, 1.5, 1.0, step=0.05)
            total_views, curve_df = predict_views_for_day(day_model, dow_choice, template_curve_days, factor)
            st.metric("Predicted 14-day views", f"{int(total_views):,}")
            chart_curve = (
                alt.Chart(curve_df)
                .mark_line()
                .encode(
                    x="day_since_publish:Q",
                    y="predicted_views:Q",
                )
            )
            st.altair_chart(chart_curve, use_container_width=True)


    # --- SCENARIO PLANNER (kept simple) ---
    with tab_scenarios:
        st.subheader("Scenario Planner")
        horizon_days = st.selectbox("Horizon (days)", [30, 60, 90], index=0)
        n_scenarios = st.slider("Number of scenarios", 1, 3, 2)
        results = []
        for i in range(n_scenarios):
            with st.expander(f"Scenario {i+1}", expanded=True):
                name = st.text_input(f"Name {i+1}", value=f"Scenario {i+1}")
                vids_per_week = st.slider(f"Videos per week {i+1}", 1, 14, 3)
                dow_multi = st.multiselect(
                    f"Posting days {i+1}",
                    options=sorted(day_model["dow"].dropna().unique().tolist()),
                    default=sorted(day_model["dow"].dropna().unique().tolist())[:3],
                )
                factor = st.slider(f"Strength multiplier {i+1}", 0.5, 1.5, 1.0, step=0.05)
            if dow_multi:
                weeks = horizon_days / 7
                n_vids = int(vids_per_week * weeks)
                views_list = []
                for v in range(n_vids):
                    d = dow_multi[v % len(dow_multi)]
                    row = day_model[day_model["dow"] == d]
                    base = row["expected_views"].iloc[0] if not row.empty else day_model["expected_views"].mean()
                    views_list.append(base * factor)
                total = np.sum(views_list)
                results.append({"Scenario": name, "Videos": n_vids, "Total Views": total})

        if results:
            res_df = pd.DataFrame(results)
            st.dataframe(res_df)
