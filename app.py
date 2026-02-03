import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime, timedelta

# ------------- CONFIG -----------------
st.set_page_config(page_title="YouTube Creator Day Planner", layout="wide")


# ------------- UTILS ------------------
def parse_date_col(df):
    """
    Tries to create a 'publish_date' column as datetime.date.
    Adapt based on your actual export (e.g., 'Date' or 'Video publish time').
    """
    possible_date_cols = ["publish_date", "date", "Video publish time", "Video publish date"]
    date_col = None

    for c in possible_date_cols:
        if c in df.columns:
            date_col = c
            break

    if date_col is None:
        st.error("Could not find a publish date column. Please rename in the CSV or adjust code.")
        return df

    df["publish_date"] = pd.to_datetime(df[date_col], errors="coerce").dt.date
    df["dow"] = pd.to_datetime(df["publish_date"]).dt.day_name()
    return df


def ensure_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = np.nan
    return df


def best_days(df, top_k=3):
    """
    Compute best days of week based on average views per video.
    """
    if "dow" not in df.columns:
        return None, None

    day_perf = (
        df.groupby("dow", dropna=False)["views"]
        .mean()
        .reset_index()
        .rename(columns={"views": "avg_views"})
    ).dropna(subset=["avg_views"])

    if day_perf.empty:
        return None, None

    day_perf_sorted = day_perf.sort_values("avg_views", ascending=False)
    return day_perf_sorted.head(top_k), day_perf_sorted


def build_day_model(df):
    """
    Build a simple model:
    - For each day of week, compute average views per video.
    """
    model = (
        df.groupby("dow", dropna=False)["views"]
        .mean()
        .reset_index()
        .rename(columns={"views": "expected_views"})
    )
    return model


def get_view_curve_template_days(n_days=14):
    """
    Construct a generic non-linear accumulation curve for views over days.
    S-shaped early, then plateau, with a small later bump.
    """
    x = np.arange(0, n_days)  # days since publish

    # Logistic S-curve
    k = 0.6
    midpoint = 3
    base = 1 / (1 + np.exp(-k * (x - midpoint)))
    base = base / base.max()

    # Small bump around day 7
    bump_center = 7
    bump_width = 2
    bump = 0.15 * np.exp(-((x - bump_center) ** 2) / (2 * bump_width**2))

    curve = base + bump
    curve = curve / curve.max()

    cum_curve = np.cumsum(curve)
    cum_curve = cum_curve / cum_curve.max()

    return pd.DataFrame({"day_since_publish": x, "cum_frac": cum_curve})


def predict_views_for_day(day_model, dow, total_curve, expected_factor=1.0):
    """
    Given a day-of-week, get expected total views and produce a full curve.
    """
    row = day_model[day_model["dow"] == dow]

    if row.empty:
        base_views = day_model["expected_views"].mean()
    else:
        base_views = row["expected_views"].iloc[0]

    total_views = base_views * expected_factor

    curve_df = total_curve.copy()
    curve_df["predicted_views"] = curve_df["cum_frac"] * total_views
    return total_views, curve_df


# ------------- APP --------------------

st.title("YouTube Creator Day Planner (Thesis Prototype)")

st.markdown(
    "Upload your YouTube Studio export, explore your channel, "
    "discover your best posting **day**, and plan future uploads with scenario simulations."
)

# --- Sidebar: Upload ---
st.sidebar.header("1. Upload data")
uploaded_file = st.sidebar.file_uploader("Upload YouTube Studio CSV", type=["csv"])

if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
    st.success("File uploaded. Preview below:")
    st.dataframe(df_raw.head())

    df = df_raw.copy()

    # Parse publish date -> day of week
    df = parse_date_col(df)

    # Ensure numeric metrics
    numeric_cols = ["views", "watch_time_hours", "impressions", "ctr"]
    df = ensure_numeric(df, numeric_cols)

    # Drop rows without views
    df = df.dropna(subset=["views"])

    # Build model artifacts
    day_model = build_day_model(df)
    template_curve_days = get_view_curve_template_days(n_days=14)

    # Create tabs
    tab_dash, tab_best, tab_planner, tab_scenarios = st.tabs(
        ["📊 Dashboard", "📅 Best Day", "🗓️ Post Planner", "🎯 Scenario Planner"]
    )

    # ------ DASHBOARD TAB ------
    with tab_dash:
        st.subheader("Channel Overview")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Views", int(df["views"].sum()))
        col2.metric("Total Videos", df.shape[0])
        col3.metric(
            "Avg Views per Video",
            f"{df['views'].mean():.0f}" if not df["views"].empty else "N/A",
        )
        if "watch_time_hours" in df.columns:
            col4.metric("Total Watch Time (hrs)", f"{df['watch_time_hours'].sum():.1f}")
        else:
            col4.metric("Total Watch Time (hrs)", "N/A")

        # Time series by publish date
        st.markdown("### Views over time (by publish date)")
        if "publish_date" in df.columns:
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

        # Views by day of week
        st.markdown("### Average views by day of week")
        if "dow" in df.columns:
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

        # Top videos
        st.markdown("### Top Videos")
        top_n = st.slider("Show top N videos by views", 5, 50, 10)
        display_cols = [c for c in ["title", "views", "impressions", "ctr", "watch_time_hours", "publish_date"] if c in df.columns]
        top_videos = df.sort_values("views", ascending=False).head(top_n)[display_cols]
        st.dataframe(top_videos)

    # ------ BEST DAY TAB ------
    with tab_best:
        st.subheader("Best Day to Post")

        res_top3, full_day_perf = best_days(df, top_k=3)

        if res_top3 is None or res_top3.empty:
            st.warning("Not enough data to compute best days.")
        else:
            st.markdown("#### Top days by average views per video")
            st.dataframe(res_top3)

            if full_day_perf is not None and not full_day_perf.empty:
                st.markdown("#### Average views per video by day")
                day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
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

    # ------ POST PLANNER TAB ------
    with tab_planner:
        st.subheader("Post Planner: Predict Views for a Planned Day")

        st.markdown(
            "Pick a **day of week** you plan to upload. "
            "The app predicts expected 14-day views and shows a **non-linear** view accumulation curve."
        )

        dow_options = list(day_model["dow"].dropna().unique())
        if not dow_options:
            st.warning("No valid days found in data.")
        else:
            day_order_all = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            dow_options_sorted = [d for d in day_order_all if d in dow_options]

            dow_choice = st.selectbox("Planned day of week", dow_options_sorted)

            st.markdown("Content strength: how strong you expect this video to be vs typical videos on that day.")
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

            # Optional: compare different days with a slider
            st.markdown("#### Compare days quickly")
            compare_dow = st.select_slider("Compare with another day", options=dow_options_sorted, value=dow_choice)
            if compare_dow != dow_choice:
                total_views_cmp, curve_df_cmp = predict_views_for_day(
                    day_model, compare_dow, template_curve_days, expected_factor=1.0
                )
                comp_df = curve_df.copy()
                comp_df["Scenario"] = dow_choice
                curve_df_cmp["Scenario"] = compare_dow
                merged = pd.concat([comp_df, curve_df_cmp], ignore_index=True)

                chart_compare = (
                    alt.Chart(merged)
                    .mark_line()
                    .encode(
                        x=alt.X("day_since_publish:Q", title="Days since publish"),
                        y=alt.Y("predicted_views:Q", title="Predicted cumulative views"),
                        color="Scenario:N",
                        tooltip=["Scenario", "day_since_publish", "predicted_views"],
                    )
                    .properties(height=300)
                )
                st.altair_chart(chart_compare, use_container_width=True)

    # ------ SCENARIO PLANNER TAB ------
    with tab_scenarios:
        st.subheader("Scenario Planner")

        st.markdown(
            "Define different posting **schedules** (days + frequency) for the next 30–90 days "
            "and compare their total predicted views."
        )

        horizon_days = st.selectbox("Horizon (days)", [30, 60, 90], index=0)

        n_scenarios = st.slider("Number of scenarios", 1, 3, 2)
        scenario_results = []

        for i in range(n_scenarios):
            st.markdown(f"### Scenario {i+1}")
            with st.expander(f"Configure Scenario {i+1}", expanded=True):
                name = st.text_input(f"Scenario {i+1} name", value=f"Scenario {i+1}")
                videos_per_week = st.slider(
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
                n_videos = int(videos_per_week * weeks)

                combos = dow_multi
                simulated_views = []
                for v in range(n_videos):
                    d = combos[v % len(combos)]
                    row = day_model[day_model["dow"] == d]
                    if row.empty:
                        base = day_model["expected_views"].mean()
                    else:
                        base = row["expected_views"].iloc[0]
                    simulated_views.append(base * content_factor)

                total_views_scenario = np.sum(simulated_views)
                scenario_results.append(
                    {
                        "Scenario": name,
                        "Videos": n_videos,
                        "Total Views": total_views_scenario,
                        "Avg Views/Video": total_views_scenario / max(n_videos, 1),
                    }
                )

        if scenario_results:
            st.markdown("### Scenario summary")
            res_df = pd.DataFrame(scenario_results)
            st.dataframe(res_df)

            st.markdown("### Total views by scenario")
            chart_scen = (
                alt.Chart(res_df)
                .mark_bar()
                .encode(
                    x=alt.X("Scenario:N"),
                    y=alt.Y("Total Views:Q"),
                    tooltip=["Scenario", "Total Views", "Videos", "Avg Views/Video"],
                )
            )
            st.altair_chart(chart_scen, use_container_width=True)

else:
    st.info("Upload your YouTube Studio CSV from the sidebar to begin.")
