import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

st.set_page_config(page_title="YouTube Creator Day Planner", layout="wide")


# ---------- COLUMN HELPERS ----------

def find_col(df, candidates):
    """Return the first column name from candidates that exists in df, or None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def show_missing_cols_message(missing_required, df):
    st.error(
        "Your CSV is missing some required columns. "
        "Please export the **video-level report** from YouTube Studio (Advanced mode) "
        "and include at least the required columns listed below."
    )
    st.markdown("**Required columns (any of these names is okay):**")
    st.markdown("- Date: `Video publish time`, `Published at`, or `Date`")
    st.markdown("- Views: `Views`")

    st.markdown("**Missing in your file:**")
    for item in missing_required:
        st.markdown(f"- {item}")

    st.markdown("**Columns detected in your upload:**")
    st.write(list(df.columns))


# ---------- CLEANING & MODEL HELPERS ----------

def load_and_clean(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Clean YouTube Studio export:
    - Flexible column mapping
    - Drop total row
    - Parse dates
    - Standardize numeric columns
    """

    df = df_raw.copy()

    # Drop 'Total' summary row if present
    if "Content" in df.columns:
        df = df[df["Content"] != "Total"]

    # --- Detect required columns ---

    date_col = find_col(df, ["Video publish time", "Published at", "Date"])
    views_col = find_col(df, ["Views"])

    missing_required = []
    if date_col is None:
        missing_required.append("Date column (Video publish time / Published at / Date)")
    if views_col is None:
        missing_required.append("Views")

    if missing_required:
        show_missing_cols_message(missing_required, df_raw)
        return pd.DataFrame()

    # --- Parse date ---

    date_series = df[date_col].astype(str).str.strip()
    parsed = pd.to_datetime(date_series, errors="coerce")
    if parsed.notna().sum() == 0:
        st.error(
            "Could not parse any dates from the date column. "
            "Make sure the column contains valid dates (as exported from YouTube Studio)."
        )
        return pd.DataFrame()

    df["publish_date"] = parsed.dt.date
    df["dow"] = parsed.dt.day_name()
    df["year_month"] = parsed.dt.to_period("M").astype(str)  # e.g. '2025-12'

    # --- Standardize key numeric columns ---

    rename_map = {}

    # Always rename views column
    rename_map[views_col] = "views"

    # Optional columns
    col_likes = find_col(df, ["Likes"])
    col_comments = find_col(df, ["Comments added"])
    col_watch = find_col(df, ["Watch time (hours)"])
    col_subs = find_col(df, ["Subscribers"])
    col_impr = find_col(df, ["Impressions"])
    col_ctr = find_col(df, ["Impressions click-through rate (%)"])

    if col_likes:
        rename_map[col_likes] = "likes"
    if col_comments:
        rename_map[col_comments] = "comments"
    if col_watch:
        rename_map[col_watch] = "watch_time_hours"
    if col_subs:
        rename_map[col_subs] = "subscribers"
    if col_impr:
        rename_map[col_impr] = "impressions"
    if col_ctr:
        rename_map[col_ctr] = "ctr_percent"

    df = df.rename(columns=rename_map)

    # Convert numerics where present
    numeric_cols = [
        "views",
        "watch_time_hours",
        "impressions",
        "ctr_percent",
        "subscribers",
        "likes",
        "dislikes",  # if present
        "shares",    # if present
        "comments",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows missing key fields
    df = df.dropna(subset=["publish_date"])
    df = df.dropna(subset=["views"])

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

st.title("YouTube Creator Day Planner")

st.markdown(
    "Upload a **YouTube Studio video report CSV** (Advanced mode, video-level). "
    "Make sure it includes at least the publish date and views for each video."
)

uploaded = st.sidebar.file_uploader("Upload YouTube Studio CSV", type=["csv"])

if uploaded is None:
    st.info("Upload your CSV from the sidebar to begin.")
else:
    try:
        df_raw = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV file: {e}")
        st.stop()

    st.subheader("Raw data preview")
    st.dataframe(df_raw.head())

    df = load_and_clean(df_raw)

    if df.empty:
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

        # Best month metrics (if year_month exists)
        if "year_month" in df.columns:
            month_views = (
                df.groupby("year_month")["views"]
                .sum()
                .reset_index()
                .sort_values("views", ascending=False)
            )
            best_month_views = month_views.iloc[0]

            # For count, fall back to counting rows if Content not present
            if "Content" in df.columns:
                month_counts = (
                    df.groupby("year_month")["Content"]
                    .count()
                    .reset_index()
                    .rename(columns={"Content": "video_count"})
                    .sort_values("video_count", ascending=False)
                )
            else:
                month_counts = (
                    df.groupby("year_month")["views"]
                    .count()
                    .reset_index()
                    .rename(columns={"views": "video_count"})
                    .sort_values("video_count", ascending=False)
                )
            best_month_videos = month_counts.iloc[0]

            st.markdown("### Monthly performance")
            colm1, colm2 = st.columns(2)
            colm1.metric(
                "Best month by views",
                f"{best_month_views['year_month']} ({int(best_month_views['views']):,} views)",
            )
            colm2.metric(
                "Best month by number of videos",
                f"{best_month_videos['year_month']} ({int(best_month_videos['video_count'])} videos)",
            )

            st.markdown("#### Views by month")
            chart_month = (
                alt.Chart(month_views.sort_values("year_month"))
                .mark_bar()
                .encode(
                    x=alt.X("year_month:N", title="Year-Month"),
                    y=alt.Y("views:Q", title="Total Views"),
                    tooltip=["year_month", "views"],
                )
                .properties(height=300)
            )
            st.altair_chart(chart_month, use_container_width=True)

        # Engagement highlights (only if columns exist)
        st.markdown("### Engagement highlights")

        if "likes" in df.columns and df["likes"].notna().any():
            most_liked = df.sort_values("likes", ascending=False).iloc[0]
            st.write("**Most liked video:**")
            st.write(
                f"{most_liked.get('Video title', 'N/A')} "
                f"({int(most_liked['likes'])} likes, {int(most_liked['views'])} views)"
            )

        if "comments" in df.columns and df["comments"].notna().any():
            most_commented = df.sort_values("comments", ascending=False).iloc[0]
            st.write("**Most commented video:**")
            st.write(
                f"{most_commented.get('Video title', 'N/A')} "
                f"({int(most_commented['comments'])} comments, {int(most_commented['views'])} views)"
            )

        st.markdown("### Top videos (by views)")
        cols_to_show = ["Video title", "views", "likes", "comments", "publish_date"]
        existing_cols = [c for c in cols_to_show if c in df.columns]
        top_n = st.slider("Show top N videos", 5, 50, 10)
        top_videos = df.sort_values("views", ascending=False).head(top_n)[existing_cols]
        st.dataframe(top_videos)

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
            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            perf = full_day_perf.copy()
            perf["dow"] = pd.Categorical(perf["dow"], categories=day_order, ordered=True)
            perf = perf.sort_values("dow")

            chart = (
                alt.Chart(perf)
                .mark_bar()
                .encode(
                    x=alt.X("avg_views:Q", title="Avg Views per Video"),
                    y=alt.Y("dow:N", title="Day of Week"),
                    tooltip=["dow", "avg_views"],
                )
                .properties(height=300)
            )
            st.altair_chart(chart, use_container_width=True)

    # ---------- POST PLANNER ----------
    with tab_planner:
        st.subheader("Post Planner: Predict Views for a Planned Day")

        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
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
                name = st.textinput(f"Scenario {i+1} name", value=f"Scenario {i+1}")
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

            st.markdown("### Total views by scenario (horizontal)")
            chart_scen_total = (
                alt.Chart(res_df)
                .mark_bar()
                .encode(
                    y=alt.Y("Scenario:N", title="Scenario"),
                    x=alt.X("Total Views:Q", title="Total Views"),
                    tooltip=["Scenario", "Total Views", "Videos", "Avg Views/Video"],
                )
                .properties(height=300)
            )
            st.altair_chart(chart_scen_total, use_container_width=True)

            st.markdown("### Average views per video by scenario (horizontal)")
            chart_scen_avg = (
                alt.Chart(res_df)
                .mark_bar(color="#FF7F0E")
                .encode(
                    y=alt.Y("Scenario:N", title="Scenario"),
                    x=alt.X("Avg Views/Video:Q", title="Avg Views per Video"),
                    tooltip=["Scenario", "Total Views", "Videos", "Avg Views/Video"],
                )
                .properties(height=300)
            )
            st.altair_chart(chart_scen_avg, use_container_width=True)
