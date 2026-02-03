import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime, timedelta

# ------------- CONFIG -----------------
st.set_page_config(page_title="Creator Post Planner", layout="wide")


# ------------- UTILS ------------------
def parse_datetime_col(df):
    """
    Tries to create a unified 'published_at' column as datetime.
    Adapt this function based on your actual export schema.
    """
    possible_datetime_cols = ["publishedAt", "publish_time", "publish_date", "date"]
    dt_col = None

    for c in possible_datetime_cols:
        if c in df.columns:
            dt_col = c
            break

    if dt_col is None:
        # If no single datetime col, try combining date + time
        if "date" in df.columns and "time" in df.columns:
            df["published_at"] = pd.to_datetime(
                df["date"].astype(str) + " " + df["time"].astype(str),
                errors="coerce",
            )
        else:
            st.error("Could not find a publish datetime column. Please adjust code.")
            return df
    else:
        df["published_at"] = pd.to_datetime(df[dt_col], errors="coerce")

    df["publish_date"] = df["published_at"].dt.date
    df["publish_hour"] = df["published_at"].dt.hour
    df["dow"] = df["published_at"].dt.day_name()
    return df


def ensure_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = np.nan
    return df


def compute_slot(df, slot_minutes=60):
    df["slot"] = (df["publish_hour"] * 60 // slot_minutes).astype(int)
    df["slot_label"] = df["slot"].apply(lambda s: f"{(s*slot_minutes)//60:02d}:00")
    return df


def best_slots(df, horizon_days=7, top_k=3):
    """
    Compute best posting windows (day_of_week, slot) based on normalized views.

    Assumes df has:
      - published_at
      - views
      - dow
      - slot
    """
    if "published_at" not in df.columns:
        return None

    # For simplicity, approximate "7-day views" as total views (for each video)
    # In a more detailed implementation, we'd need daily video-level views.
    slot_perf = (
        df.groupby(["dow", "slot", "slot_label"], dropna=False)["views"]
        .mean()
        .reset_index()
    )
    slot_perf = slot_perf.dropna(subset=["views"])
    slot_perf = slot_perf.sort_values("views", ascending=False)

    return slot_perf.head(top_k), slot_perf


def build_slot_model(df):
    """
    Build a simple slot-based model:
    - For each (dow, slot), compute average views per video.
    - This will be used as the base for predicting total views for a planned post.
    """
    model = (
        df.groupby(["dow", "slot", "slot_label"], dropna=False)["views"]
        .mean()
        .reset_index()
        .rename(columns={"views": "expected_views"})
    )
    return model


def get_view_curve_template(df, n_points=72):
    """
    Construct a generic non-linear accumulation curve for views over hours.
    In reality, you would derive this from time-series data.
    Here we build a simple S-shaped curve + small later bump.
    """
    # x = hours since publish
    x = np.linspace(0, n_points - 1, n_points)

    # Base S-shaped curve with logistic function
    k = 0.2  # steepness
    midpoint = 12
    base = 1 / (1 + np.exp(-k * (x - midpoint)))

    # Normalize to max 1
    base = base / base.max()

    # Add a small bump later (e.g., re-surfacing by algorithm)
    bump_center = 48
    bump_width = 8
    bump = 0.15 * np.exp(-((x - bump_center) ** 2) / (2 * bump_width**2))

    curve = base + bump
    curve = curve / curve.max()

    # Convert to cumulative proportion over time
    cum_curve = np.cumsum(curve)
    cum_curve = cum_curve / cum_curve.max()

    return pd.DataFrame({"hour_since_publish": x, "cum_frac": cum_curve})


def predict_views_for_slot(slot_model, dow, slot, total_curve, expected_factor=1.0):
    """
    Given a day-of-week and slot, get expected total views and produce a full curve.
    expected_factor allows scenario adjustments (e.g. +10%).
    """
    row = slot_model[
        (slot_model["dow"] == dow) & (slot_model["slot"] == slot)
    ]

    if row.empty:
        # Fallback: global mean
        base_views = slot_model["expected_views"].mean()
    else:
        base_views = row["expected_views"].iloc[0]

    total_views = base_views * expected_factor

    curve_df = total_curve.copy()
    curve_df["predicted_views"] = curve_df["cum_frac"] * total_views
    return total_views, curve_df


# ------------- APP --------------------

st.title("YouTube Creator Post Planner (Thesis Prototype)")

st.markdown(
    "Upload your YouTube Studio export, explore your channel, "
    "see best posting times, and plan future posts with scenario simulations."
)

# --- Sidebar: Upload ---
st.sidebar.header("1. Upload data")
uploaded_file = st.sidebar.file_uploader("Upload YouTube Studio CSV", type=["csv"])

if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
    st.success("File uploaded. Preview below:")
    st.dataframe(df_raw.head())

    # Assumptions: these columns exist or will be renamed accordingly.
    df = df_raw.copy()

    # Parse datetime
    df = parse_datetime_col(df)

    # Ensure numeric metrics exist
    numeric_cols = ["views", "impressions", "ctr", "watch_time_hours"]
    df = ensure_numeric(df, numeric_cols)

    # Add slot & day-of-week
    df = compute_slot(df, slot_minutes=60)

    # Handle missing views
    df = df.dropna(subset=["views"])

    # Build model artifacts
    slot_model = build_slot_model(df)
    template_curve = get_view_curve_template(df, n_points=72)

    # Create tabs
    tab_dash, tab_best, tab_planner, tab_scenarios = st.tabs(
        ["📊 Dashboard", "⏰ Best Time to Post", "📅 Post Planner", "🎯 Scenario Planner"]
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

        # Time series
        st.markdown("### Views over time")
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
                    x="publish_date:T",
                    y="views:Q",
                    tooltip=["publish_date:T", "views:Q"],
                )
                .properties(height=300)
            )
            st.altair_chart(chart_ts, use_container_width=True)

        # Top videos
        st.markdown("### Top Videos")
        top_n = st.slider("Show top N videos by views", 5, 50, 10)
        top_videos = df.sort_values("views", ascending=False).head(top_n)[
            ["title", "views", "impressions", "ctr", "watch_time_hours", "published_at"]
            if "watch_time_hours" in df.columns
            else ["title", "views", "impressions", "ctr", "published_at"]
        ]
        st.dataframe(top_videos)

    # ------ BEST TIME TAB ------
    with tab_best:
        st.subheader("Best Days & Times to Post")

        res_top3, full_slot_perf = best_slots(df, top_k=3)

        if res_top3 is None or res_top3.empty:
            st.warning("Not enough data to compute best posting windows.")
        else:
            st.markdown("#### Top 3 posting windows (by average views per video)")
            st.dataframe(res_top3[["dow", "slot_label", "views"]])

            st.markdown("#### Heatmap: Average views by day and hour")
            if not full_slot_perf.empty:
                heat = full_slot_perf.copy()
                heat_chart = (
                    alt.Chart(heat)
                    .mark_rect()
                    .encode(
                        x=alt.X("slot_label:N", title="Hour Slot"),
                        y=alt.Y("dow:N", title="Day of Week"),
                        color=alt.Color("views:Q", title="Avg Views", scale=alt.Scale(scheme="blues")),
                        tooltip=["dow", "slot_label", "views"],
                    )
                    .properties(height=300)
                )
                st.altair_chart(heat_chart, use_container_width=True)

    # ------ POST PLANNER TAB ------
    with tab_planner:
        st.subheader("Post Planner: Predict Views for a Planned Upload")

        st.markdown(
            "Select a **day** and **time** you plan to upload. "
            "The app predicts expected views and shows a **non-linear** view accumulation curve."
        )

        # Inputs for planned post
        dow_options = list(slot_model["dow"].dropna().unique())
        dow_choice = st.selectbox("Day of Week", sorted(dow_options))

        slot_labels = (
            slot_model[slot_model["dow"] == dow_choice]["slot_label"]
            .dropna()
            .unique()
        )
        if len(slot_labels) == 0:
            st.warning("No slot data for this day of week. Choose another.")
        else:
            # Slider over hour slots
            slot_labels_sorted = sorted(slot_labels)
            slot_label_choice = st.select_slider(
                "Hour Slot", options=slot_labels_sorted, value=slot_labels_sorted[0]
            )

            # Map back to numeric slot
            slot_row = slot_model[
                (slot_model["dow"] == dow_choice)
                & (slot_model["slot_label"] == slot_label_choice)
            ]
            if slot_row.empty:
                st.warning("Could not map slot label to slot. Please adjust data.")
            else:
                slot_value = int(slot_row["slot"].iloc[0])

                # Factor slider to simulate content strength (+/- 50% views)
                st.markdown("Content strength adjustment (how strong you expect the video to be):")
                factor = st.slider(
                    "Multiplier vs typical video in this slot",
                    0.5,
                    1.5,
                    1.0,
                    step=0.05,
                )

                total_views, curve_df = predict_views_for_slot(
                    slot_model, dow_choice, slot_value, template_curve, expected_factor=factor
                )

                col1, col2 = st.columns(2)
                col1.metric("Predicted 7-day views", f"{int(total_views):,}")
                col2.metric("Slot baseline views", f"{int(total_views / factor):,}")

                st.markdown("#### Predicted view accumulation (first 72 hours)")
                chart_curve = (
                    alt.Chart(curve_df)
                    .mark_line()
                    .encode(
                        x=alt.X("hour_since_publish:Q", title="Hours since publish"),
                        y=alt.Y("predicted_views:Q", title="Predicted cumulative views"),
                        tooltip=["hour_since_publish", "predicted_views"],
                    )
                    .properties(height=300)
                )
                st.altair_chart(chart_curve, use_container_width=True)

    # ------ SCENARIO PLANNER TAB ------
    with tab_scenarios:
        st.subheader("Scenario Planner")

        st.markdown(
            "Compare different posting strategies for the next 30 or 90 days. "
            "Each scenario defines posting days, slots, and content strength."
        )

        horizon_days = st.selectbox("Horizon", [30, 60, 90], index=0)

        n_scenarios = st.slider("Number of scenarios", 1, 3, 2)
        scenario_results = []

        for i in range(n_scenarios):
            st.markdown(f"### Scenario {i+1}")
            with st.expander(f"Configure Scenario {i+1}", expanded=True):
                name = st.text_input(f"Scenario {i+1} name", value=f"Scenario {i+1}")
                videos_per_week = st.slider(
                    f"Videos per week (Scenario {i+1})", 1, 14, 3
                )
                # Day-of-week selection
                dow_multi = st.multiselect(
                    f"Posting days of week (Scenario {i+1})",
                    options=sorted(slot_model["dow"].dropna().unique()),
                    default=sorted(slot_model["dow"].dropna().unique())[:3],
                )
                # Slot selection
                slot_multi = st.multiselect(
                    f"Hour slots (Scenario {i+1})",
                    options=sorted(slot_model["slot_label"].dropna().unique()),
                    default=sorted(slot_model["slot_label"].dropna().unique())[:2],
                )
                content_factor = st.slider(
                    f"Avg content strength multiplier (Scenario {i+1})",
                    0.5,
                    1.5,
                    1.0,
                    step=0.05,
                )

            if dow_multi and slot_multi:
                # Approx number of videos in horizon
                weeks = horizon_days / 7
                n_videos = int(videos_per_week * weeks)

                # Cycle through combos of (dow, slot) for simplicity
                combos = [(d, s) for d in dow_multi for s in slot_multi]
                if not combos:
                    continue

                # Simulate videos
                simulated_views = []
                for v in range(n_videos):
                    d, s_label = combos[v % len(combos)]
                    row = slot_model[
                        (slot_model["dow"] == d) & (slot_model["slot_label"] == s_label)
                    ]
                    if row.empty:
                        base = slot_model["expected_views"].mean()
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
