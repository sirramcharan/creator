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
    Clean YouTube Studio export based on your structure:
    Content, Video title, Video publish time, Duration, Likes, Dislikes, Shares,
    Comments added, Views, Watch time (hours), Subscribers, Impressions,
    Impressions click-through rate (%)
    """
    df = df_raw.copy()

    # Debug info
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

    # Use pandas inference (works according to your debug: 127 non-null)
    date_series = df["Video publish time"].astype(str).str.strip()
    parsed = pd.to_datetime(date_series, errors="coerce")  # infer format
    st.write(f"Parsed publish dates (infer): {parsed.notna().sum()} non-null")

    if parsed.notna().sum() == 0:
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
        st.error("No usable rows after cleaning. Check the debug info above.")
        st.stop()

    st.success(f"Loaded {len(df)} videos after cleaning.")

    day_model = build_day_model(df)
    template_curve_days = get_view_curve_template_days(n_days=14)

    tab_dash, tab_best, tab_planner, tab_scenarios = st.tabs(
        ["📊 Dashboard", "📅 Best Day", "🗓️ Post Planner", "🎯 Scenario Planner"]
    )

    # ---------- DASHBOARD ----------
    with tab_dash:
        st.subheader("Channel Overview")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Views", int(df["views"].sum()))
        col2.metric("Total Videos", df.shape[0])
        col3.metric("Avg Views / Video", f"{df['views'].mean():.0f}")

        st.markdown("### Views over time (by publish date)")
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
                tooltip=["publish_date:T", "views:Q"],
            )
            .properties(height=300)
        )
        st.altair_chart(chart_ts, use_container_width=True)

        st.markdown("### Average views by day of week")
        day_perf = (
            df.groupby("dow")["views"].mean().reset_index().rename(columns={"views": "avg_views"})
        )
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day_perf["dow"] = pd.Categorical(day_perf["dow"], categories=day_order, ordered=True)
        day_perf = day_perf.sort_values("dow")

        chart_day = (
            alt.Chart(day_perf)
            .mark_bar()
            .encode(
                x=alt.X("dow:N", title="Day of Week"),
                y=alt.Y("avg_views:Q", title="Avg Views per Video"),
                tooltip=["dow", "avg_views"],
            )
            .properties(height=300)
        )
        st.altair_chart(chart_day, use_container_width=True)

    # ---------- BEST DAY ----------
    with tab_best:
        st.subheader("Best Day to Post")

        res_top3, full_day_perf = best_days(df, top_k=3)
        if res_top3 is None or res_top3.empty:
            st.warning("Not enough data to compute best days.")
        else:
            st.markdown("#### Top days by average views per video")
            st.dataframe(res_top3)

            st.markdown("#### Average views per video by day")
            perf = full_day_perf.copy()
            perf["dow"] = pd.Categorical(perf["dow"], categories=day_order, ordered=True)
            perf = perf.sort_values("dow")

            chart = (
                alt.Chart(perf)
                .mark_bar()
                .encode(
                    x=alt.X("dow:N", title="Day of Week"),
                    y=alt.Y("avg_views:Q", title="Avg Views per Video"),
                    tooltip=["dow", "avg_views"],
                )
                .properties(height=300)
            )
            st.altair_chart(chart, use_container_width=True)

    # ---------- POST PLANNER ----------
    with tab_planner:
        st.subheader("Post Planner: Predict Views for a Planned Day")

        dow_options = list(day_model["dow"].dropna().unique())
        if not dow_options:
            st.warning("No days found in data.")
        else:
            dow_sorted = [d for d in day_order if d in dow_options]

            dow_choice = st.selectbox("Planned day of week", dow_sorted)

            st.markdown("Content strength vs typical videos on that day.")
            factor = st.slider(
                "Multiplier vs typical video on this day",
                0.5,
                1.5,
                1.0,
                step=0.05,
            )

            total_views, curve_df = predict_views_for_day(
                day_model, dow_choice, template_curve_days, expected_factor=factor
            )

            col1, col2 = st.columns(2)
            col1.metric("Predicted 14-day views", f"{int(total_views):,}")
            col2.metric("Day baseline views", f"{int(total_views / factor):,}")

            st.markdown("#### Predicted view accumulation (first 14 days)")
            chart_curve = (
                alt.Chart(curve_df)
                .mark_line()
                .encode(
                    x=alt.X("day_since_publish:Q", title="Days since publish"),
                    y=alt.Y("predicted_views:Q", title="Predicted cumulative views"),
                    tooltip=["day_since_publish", "predicted_views"],
                )
                .properties(height=300)
            )
            st.altair_chart(chart_curve, use_container_width=True)

    # ---------- SCENARIO PLANNER ----------
    with tab_scenarios:
        st.subheader("Scenario Planner")

        horizon_days = st.selectbox("Horizon (days)", [30, 60, 90], index=0)
        n_scenarios = st.slider("Number of scenarios", 1, 3, 2)
        scenario_results = []

        for i in range(n_scenarios):
            st.markdown(f"### Scenario {i+1}")
            with st.expander(f"Configure Scenario {i+1}", expanded=True):
                name = st.text_input(f"Scenario {i+1} name", value=f"Scenario {i+1}")
                vids_per_week = st.slider(
                    f"Videos per week (Scenario {i+1})", 1, 14, 3
                )
                dow_multi = st.multiselect(
                    f"Posting days of week (Scenario {i+1})",
                    options=sorted(day_model["dow"].dropna().unique().tolist()),
                    default=sorted(day_model["dow"].dropna().unique().tolist())[:3],
                )
                content_factor = st.slider(
                    f"Avg content strength multiplier (Scenario {i+1})",
                    0.5,
                    1.5,
                    1.0,
                    step=0.05,
                )

            if dow_multi:
                weeks = horizon_days / 7
                n_videos = int(vids_per_week * weeks)

                simulated_views = []
                for v in range(n_videos):
                    d = dow_multi[v % len(dow_multi)]
                    row = day_model[day_model["dow"] == d]
                    if row.empty:
                        base = day_model["expected_views"].mean()
                    else:
                        base = row["expected_views"].iloc[0]
                    simulated_views.append(base * content_factor)

                total_views_scenario = float(np.sum(simulated_views))
                avg_views_per_video = total_views_scenario / max(n_videos, 1)

                scenario_results.append(
                    {
                        "Scenario": name,
                        "Videos": n_videos,
                        "Total Views": total_views_scenario,
                        "Avg Views/Video": avg_views_per_video,
                    }
                )

        if scenario_results:
            res_df = pd.DataFrame(scenario_results)
            st.markdown("### Scenario summary")
            st.dataframe(res_df)

            st.markdown("### Total views by scenario")
            chart_scen_total = (
                alt.Chart(res_df)
                .mark_bar()
                .encode(
                    x=alt.X("Scenario:N"),
                    y=alt.Y("Total Views:Q"),
                    tooltip=["Scenario", "Total Views", "Videos", "Avg Views/Video"],
                )
                .properties(height=300)
            )
            st.altair_chart(chart_scen_total, use_container_width=True)

            st.markdown("### Average views per video by scenario")
            chart_scen_avg = (
                alt.Chart(res_df)
                .mark_bar(color="#FF7F0E")
                .encode(
                    x=alt.X("Scenario:N"),
                    y=alt.Y("Avg Views/Video:Q"),
                    tooltip=["Scenario", "Total Views", "Videos", "Avg Views/Video"],
                )
                .properties(height=300)
            )
            st.altair_chart(chart_scen_avg, use_container_width=True)
