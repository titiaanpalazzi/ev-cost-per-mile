import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json

from ev_model import (
    SimConfig, ALL_COMPONENTS,
    haversine_vec, nearest_depot, ROAD_FACTOR,
    load_boundary, generate_sample_data,
    validate_csv, prepare_rides, MAX_RIDES,
    run_simulation, get_cost_components, render_stacked_bar,
    CATEGORY_COLORS,
)

st.set_page_config(page_title="EV Cost Per Mile", layout="wide")

DATA_DIR = Path(__file__).parent / "data"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

SF_BOUNDARY = load_boundary(DATA_DIR)


def detect_data_centroid(df):
    """Return (lat, lon) centroid of ride pickup locations."""
    return float(df["pickup_latitude"].mean()), float(df["pickup_longitude"].mean())


@st.cache_data
def load_default_data(n_rides=60000, _version=5):
    csv_path = DATA_DIR / "sf_waymo_estimates.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        df = generate_sample_data(n=n_rides, boundary=SF_BOUNDARY)
        st.info(
            "Using **simulated rides** based on "
            "[this Kaggle dataset](https://www.kaggle.com/datasets/npurav/waymo-rides-estimates) "
            "of actual Waymo rides."
        )
    return prepare_rides(df)


def load_template(filename):
    """Load a city template CSV from the data directory."""
    path = DATA_DIR / filename
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df, errors, warnings = validate_csv(df)
    if errors:
        return None
    return prepare_rides(df)


# ---------------------------------------------------------------------------
# Data source selector
# ---------------------------------------------------------------------------

st.sidebar.header("Data Source")

# Discover available templates
template_files = sorted(DATA_DIR.glob("*_template.csv"))
_CITY_ABBREVS = {"La": "LA", "Nyc": "NYC", "Sf": "SF"}
def _city_label(stem):
    label = stem.replace("_template", "").replace("_", " ").title()
    for k, v in _CITY_ABBREVS.items():
        label = label.replace(k, v)
    return label
template_choices = {_city_label(f.stem): f.name for f in template_files}

source_options = ["SF (Waymo)"] + list(template_choices.keys()) + ["Custom (Upload)"]
data_source = st.sidebar.selectbox("City", source_options)

df = None
data_source_key = "sf_waymo"
use_sf_boundary = True

if data_source == "SF (Waymo)":
    df = load_default_data()
    data_source_key = "sf_waymo"
    use_sf_boundary = True

elif data_source == "Custom (Upload)":
    uploaded = st.sidebar.file_uploader("Upload ride data CSV", type="csv")
    if uploaded is not None:
        try:
            raw = pd.read_csv(uploaded)
        except Exception as e:
            st.sidebar.error(f"Could not read CSV: {e}")
            raw = None
        if raw is not None:
            raw, errors, warnings = validate_csv(raw)
            for w in warnings:
                st.sidebar.warning(f"⚠️ {w}")
            if errors:
                for e in errors:
                    st.sidebar.error(e)
                df = None
            else:
                df = prepare_rides(raw)
                data_source_key = "custom_upload"
                use_sf_boundary = False
    else:
        st.sidebar.info("Upload a CSV with pickup/dropoff lat/lon columns, or select a city template above.")
        _template_csv = "pickup_latitude,pickup_longitude,dropoff_latitude,dropoff_longitude,request_timestamp\n37.7749,-122.4194,37.7849,-122.4094,2024-06-01 08:30:00\n37.7649,-122.4294,37.7749,-122.4194,2024-06-01 09:15:00\n"
        st.sidebar.download_button(
            "Download CSV template",
            data=_template_csv,
            file_name="ride_data_template.csv",
            mime="text/csv",
            use_container_width=True,
        )
        df = load_default_data()
        data_source_key = "sf_waymo"

elif data_source in template_choices:
    template_file = template_choices[data_source]
    df = load_template(template_file)
    if df is None:
        st.sidebar.error(f"Could not load template: {template_file}")
        df = load_default_data()
    else:
        st.sidebar.warning("⚠️ Estimated ride distribution. Upload real data for accuracy.")
        data_source_key = template_file.replace("_template.csv", "_template")
        use_sf_boundary = False

if df is None:
    df = load_default_data()
    data_source_key = "sf_waymo"

# Compute data-driven map center and depot bounds
map_center_lat, map_center_lon = detect_data_centroid(df)
lat_min = float(df["pickup_latitude"].min()) - 0.05
lat_max = float(df["pickup_latitude"].max()) + 0.05
lon_min = float(df["pickup_longitude"].min()) - 0.05
lon_max = float(df["pickup_longitude"].max()) + 0.05

# ---------------------------------------------------------------------------
# Sidebar — vehicle & cost parameters
# ---------------------------------------------------------------------------

st.sidebar.header("Vehicle Parameters")
battery_kwh = st.sidebar.slider("Battery capacity (kWh)", 30, 150, 75)
efficiency = st.sidebar.slider("Efficiency (miles / kWh)", 2.0, 5.0, 3.5, 0.1)
charge_threshold = st.sidebar.slider("Charge when battery below (%)", 5, 40, 20)
charge_speed_kw = st.sidebar.slider("Charging speed (kW)", 7, 250, 150)

st.sidebar.header("Cost Parameters", help="Costs incurred during non-revenue miles are displayed as deadhead.")
lease_per_stall = st.sidebar.number_input("Lease price ($/stall/mo)", 0, 50000, 800, 100, format="%d")
purchase_price = st.sidebar.number_input("Vehicle purchase price ($)", 20000, 150000, 85000, 1000)
lifetime_miles = st.sidebar.number_input("Expected lifetime miles", 50000, 500000, 200000, 10000)
electricity_offpeak = st.sidebar.slider("Off-peak electricity ($/kWh)", 0.05, 0.50, 0.20, 0.01)
electricity_peak = st.sidebar.slider("Peak electricity ($/kWh)", 0.10, 0.80, 0.40, 0.01)
peak_start, peak_end = 16, 21
toll_per_mile = st.sidebar.slider("Tolls ($/mile)", 0.10, 0.40, 0.25, 0.01)
insurance_per_mile = st.sidebar.slider("Insurance ($/mile)", 0.10, 0.50, 0.37, 0.01)
tire_per_mile = st.sidebar.slider("Tire wear ($/mile)", 0.00, 0.06, 0.02, 0.005)

cleaning_per_mile = 0.07
maintenance_per_mile = 0.16

st.sidebar.header("Deadhead Opportunity Cost")
include_opp_cost = st.sidebar.checkbox("Include deadhead mile opportunity cost", True)
opp_cost_per_mile = st.sidebar.slider("Opportunity cost ($/mile)", 0.50, 3.00, 2.00, 0.10) if include_opp_cost else 0.0

st.sidebar.header("Cost Components")
include_electricity = st.sidebar.checkbox("Electricity", True)
include_depreciation = st.sidebar.checkbox("Vehicle Depreciation", True)
include_maintenance = st.sidebar.checkbox("Maintenance", True)
include_insurance = st.sidebar.checkbox("Insurance", True)
include_tires = st.sidebar.checkbox("Tire wear", True)
include_cleaning = st.sidebar.checkbox("Cleaning & plug-ins", True)
include_tolls = st.sidebar.checkbox("Tolls", True)
include_deadhead = st.sidebar.checkbox("Deadhead (to depot)", True)
include_depot_lease = st.sidebar.checkbox("Depot lease", True)

# Build enabled_components set from checkboxes
_toggle_map = {
    "electricity": include_electricity,
    "depreciation": include_depreciation,
    "maintenance": include_maintenance,
    "insurance": include_insurance,
    "tires": include_tires,
    "cleaning": include_cleaning,
    "tolls": include_tolls,
    "deadhead": include_deadhead,
    "depot_lease": include_depot_lease,
    "opp_cost": include_opp_cost,
}
enabled_components = {k for k, v in _toggle_map.items() if v}

# Build SimConfig
config = SimConfig(
    battery_kwh=battery_kwh,
    efficiency=efficiency,
    charge_threshold=charge_threshold,
    charge_speed_kw=charge_speed_kw,
    lease_per_stall=lease_per_stall,
    purchase_price=purchase_price,
    lifetime_miles=lifetime_miles,
    electricity_offpeak=electricity_offpeak,
    electricity_peak=electricity_peak,
    peak_start=peak_start,
    peak_end=peak_end,
    toll_per_mile=toll_per_mile,
    insurance_per_mile=insurance_per_mile,
    tire_per_mile=tire_per_mile,
    cleaning_per_mile=cleaning_per_mile,
    maintenance_per_mile=maintenance_per_mile,
    opp_cost_per_mile=opp_cost_per_mile,
    enabled_components=enabled_components,
)

# ---------------------------------------------------------------------------
# Scenario Save / Load
# ---------------------------------------------------------------------------

st.sidebar.header("Scenarios")

col_save, col_load = st.sidebar.columns(2)

with col_save:
    scenario_name = st.text_input("Scenario name", "My Scenario", label_visibility="collapsed")
    scenario_dict = config.to_dict()
    scenario_dict["name"] = scenario_name
    scenario_dict["data_source"] = data_source_key
    if "depots" in st.session_state:
        scenario_dict["depots"] = st.session_state.depots
    if "compare_depots_a" in st.session_state:
        scenario_dict["compare_depots_a"] = st.session_state.compare_depots_a
    if "compare_depots_b" in st.session_state:
        scenario_dict["compare_depots_b"] = st.session_state.compare_depots_b

    st.download_button(
        "Save Scenario",
        data=json.dumps(scenario_dict, indent=2),
        file_name=f"{scenario_name.replace(' ', '_').lower()}.json",
        mime="application/json",
        use_container_width=True,
    )

with col_load:
    scenario_file = st.file_uploader("Load Scenario", type="json", label_visibility="collapsed")
    if scenario_file is not None:
        try:
            scenario_data = json.loads(scenario_file.read())
            loaded_config = SimConfig.from_dict(scenario_data)
            # Restore depots
            if "depots" in scenario_data:
                st.session_state.depots = scenario_data["depots"]
            if "compare_depots_a" in scenario_data:
                st.session_state.compare_depots_a = scenario_data["compare_depots_a"]
            if "compare_depots_b" in scenario_data:
                st.session_state.compare_depots_b = scenario_data["compare_depots_b"]
            st.sidebar.success(f"Loaded scenario: {scenario_data.get('name', 'unnamed')}")
            st.sidebar.info("Note: Sidebar sliders show defaults. Re-run the simulation to apply loaded values.")
        except (json.JSONDecodeError, Exception) as e:
            st.sidebar.error(f"Invalid scenario file: {e}")


# ---------------------------------------------------------------------------
# Page navigation
# ---------------------------------------------------------------------------

st.title("EV Cost Per Mile Simulator")

page = st.radio(
    "Navigation",
    ["Simulator", "Compare Depot Locations"],
    horizontal=True,
    label_visibility="collapsed",
)

# =========================================================================
# PAGE: Simulator
# =========================================================================

if page == "Simulator":

    # --- Depot placement ---
    st.subheader("Charging Depot Locations")
    st.caption(
        "Click on the map to place a depot, or edit the table. "
        "Press **Recalculate** to re-run the simulation with updated depots."
    )

    DEFAULT_DEPOTS = [
        {"lat": round(map_center_lat, 4), "lon": round(map_center_lon, 4), "name": "Central", "stalls": 10},
    ]

    if "depots" not in st.session_state:
        st.session_state.depots = list(DEFAULT_DEPOTS)

    col_map, col_edit = st.columns([2, 1])

    with col_map:
        grid_lat = np.linspace(lat_min, lat_max, 40)
        grid_lon = np.linspace(lon_min, lon_max, 40)
        glat, glon = np.meshgrid(grid_lat, grid_lon)
        glat, glon = glat.ravel(), glon.ravel()

        fig_map = go.Figure()
        fig_map.add_trace(go.Scattermapbox(
            lat=glat,
            lon=glon,
            mode="markers",
            marker=dict(size=12, color="rgba(0,0,0,0)"),
            name="Click to place depot",
            hovertemplate="Click to add depot<br>%{lat:.4f}, %{lon:.4f}<extra></extra>",
        ))
        fig_map.add_trace(go.Scattermapbox(
            lat=df["pickup_latitude"],
            lon=df["pickup_longitude"],
            mode="markers",
            marker=dict(size=3, color="#636EFA", opacity=0.3),
            name="Ride pickups",
        ))
        if st.session_state.depots:
            fig_map.add_trace(go.Scattermapbox(
                lat=[d["lat"] for d in st.session_state.depots],
                lon=[d["lon"] for d in st.session_state.depots],
                mode="markers+text",
                marker=dict(size=18, color="#EF553B", symbol="circle"),
                text=[d.get("name", f"Depot {i+1}") for i, d in enumerate(st.session_state.depots)],
                textposition="top center",
                name="Depots",
            ))
        fig_map.update_layout(
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=map_center_lat, lon=map_center_lon),
                zoom=11,
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            height=450,
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )
        map_selection = st.plotly_chart(
            fig_map, use_container_width=True, on_select="rerun", key="depot_map",
            selection_mode="points",
        )

        if map_selection and map_selection.selection and map_selection.selection.points:
            for pt in map_selection.selection.points:
                if pt.get("curve_number", -1) == 0:
                    new_lat = round(pt["lat"], 4)
                    new_lon = round(pt["lon"], 4)
                    already = any(
                        abs(d["lat"] - new_lat) < 0.001 and abs(d["lon"] - new_lon) < 0.001
                        for d in st.session_state.depots
                    )
                    if not already:
                        n = len(st.session_state.depots) + 1
                        st.session_state.depots.append(
                            {"lat": new_lat, "lon": new_lon, "name": f"Depot {n}",
                             "stalls": 10}
                        )
                        st.rerun()

    with col_edit:
        st.markdown("**Depots**")
        depot_df = pd.DataFrame(st.session_state.depots) if st.session_state.depots else pd.DataFrame(columns=["lat", "lon", "name", "stalls"])
        edited = st.data_editor(
            depot_df,
            num_rows="dynamic",
            column_config={
                "lat": st.column_config.NumberColumn("Latitude", min_value=lat_min, max_value=lat_max, step=0.001, format="%.4f"),
                "lon": st.column_config.NumberColumn("Longitude", min_value=lon_min, max_value=lon_max, step=0.001, format="%.4f"),
                "name": st.column_config.TextColumn("Name"),
                "stalls": st.column_config.NumberColumn("Stalls", min_value=1, max_value=500, step=1),
            },
            key="depot_editor",
        )
        st.session_state.depots = edited.dropna(subset=["lat", "lon"]).to_dict("records")

        if st.button("Recalculate costs", type="primary", use_container_width=True):
            st.rerun()

    depots = st.session_state.depots
    if not depots:
        st.warning("Add at least one depot to run the simulation.")
        st.stop()

    # --- Run simulation ---
    sim_df, charge_events = run_simulation(df, depots, config)

    # --- Metrics ---
    st.divider()
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total rides", f"{len(sim_df):,}", help="Simulated over one month period.")
    col2.metric("Revenue miles", f"{sim_df['distance_miles'].sum():,.0f}")
    _dh = sim_df["deadhead_miles"].sum()
    _total = sim_df["distance_miles"].sum() + _dh
    col3.metric("Deadhead miles", f"{_dh:,.0f} ({_dh/_total*100:.1f}%)")
    col4.metric("Avg cost/mile", f"${sim_df['total_cost'].sum() / sim_df['distance_miles'].sum():.3f}")
    col5.metric("Charge events", f"{len(charge_events):,}")

    # --- Charts ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Cost Breakdown", "Cost Per Mile Over Rides", "Trip Distances",
        "Battery SOC", "Charge Events",
    ])

    with tab1:
        st.plotly_chart(render_stacked_bar(sim_df), use_container_width=True)
        components = get_cost_components(sim_df)
        total_rev_miles = sim_df["distance_miles"].sum()
        st.caption(
            f"Total cost: **${sum(components.values()):,.2f}** over "
            f"**{total_rev_miles:,.0f} revenue miles** + "
            f"**{sim_df['deadhead_miles'].sum():,.0f} deadhead miles**"
        )

    with tab2:
        window = max(1, len(sim_df) // 50)
        sim_df["cpm_rolling"] = sim_df["cost_per_mile"].rolling(window, min_periods=1).mean()
        fig2 = px.line(sim_df.reset_index(), x="index", y="cpm_rolling",
                       labels={"index": "Ride #", "cpm_rolling": "Cost per mile ($)"},
                       title=f"Cost Per Mile (rolling avg, window={window})")
        st.plotly_chart(fig2, use_container_width=True)
        st.caption(
            "Cost per mile varies across rides due to peak vs. off-peak electricity rates "
            "and deadhead trips to the depot — a short ride that triggers a charge event "
            "absorbs the deadhead cost, spiking its per-mile cost."
        )

    with tab3:
        fig3 = px.histogram(sim_df, x="distance_miles", nbins=50,
                            title="Trip Distance Distribution",
                            labels={"distance_miles": "Distance (miles)"})
        st.plotly_chart(fig3, use_container_width=True)

    with tab4:
        fig4 = px.line(sim_df.reset_index(), x="index", y="soc_after",
                       labels={"index": "Ride #", "soc_after": "Battery SOC (kWh)"},
                       title="Battery State of Charge Over Rides")
        fig4.add_hline(y=battery_kwh * charge_threshold / 100,
                       line_dash="dash", line_color="red",
                       annotation_text="Charge threshold")
        st.plotly_chart(fig4, use_container_width=True)

    with tab5:
        if charge_events:
            ce_df = pd.DataFrame(charge_events)
            st.dataframe(ce_df, use_container_width=True)
            st.caption(f"Total deadhead to depots: **{ce_df['deadhead_miles'].sum():,.1f} miles**")
        else:
            st.info("No charge events occurred during this simulation.")

    # --- Export ---
    st.divider()
    st.subheader("Export")
    export_col1, export_col2 = st.columns(2)

    with export_col1:
        csv_data = sim_df.to_csv(index=False)
        st.download_button(
            "Download simulation CSV",
            data=csv_data,
            file_name="ev_simulation_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with export_col2:
        try:
            from export import generate_pdf
            pdf_bytes = generate_pdf(sim_df, config, depots)
            if pdf_bytes:
                st.download_button(
                    "Download PDF report",
                    data=pdf_bytes,
                    file_name="ev_cost_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
        except ImportError:
            st.info("PDF export requires `fpdf2` and `kaleido`. Install with: `pip install fpdf2 kaleido`")
        except Exception as e:
            st.error(f"PDF generation failed: {e}")


# =========================================================================
# PAGE: Compare Depot Locations
# =========================================================================

elif page == "Compare Depot Locations":

    st.subheader("Compare Depot Configurations")
    st.caption("This assumes the same ride data as the Simulator page. Edit the depot tables below, then press **Run Comparison** to simulate both configurations side by side.")

    if "compare_depots_a" not in st.session_state:
        st.session_state.compare_depots_a = [
            {"name": "Central A", "lat": round(map_center_lat, 4), "lon": round(map_center_lon, 4), "stalls": 20, "lease_per_stall": 800},
        ]
    if "compare_depots_b" not in st.session_state:
        st.session_state.compare_depots_b = [
            {"name": "Alt Site B", "lat": round(map_center_lat + 0.05, 4), "lon": round(map_center_lon + 0.02, 4), "stalls": 20, "lease_per_stall": 800},
        ]

    col_a, col_b = st.columns(2)

    depot_col_config = {
        "name": st.column_config.TextColumn("Name"),
        "lat": st.column_config.NumberColumn("Latitude", min_value=lat_min, max_value=lat_max, step=0.001, format="%.4f"),
        "lon": st.column_config.NumberColumn("Longitude", min_value=lon_min, max_value=lon_max, step=0.001, format="%.4f"),
        "stalls": st.column_config.NumberColumn("Stalls", min_value=1, max_value=500, step=1),
        "lease_per_stall": st.column_config.NumberColumn("Lease ($/stall/mo)", min_value=0, max_value=50000, step=100, format="%d"),
    }

    with col_a:
        st.markdown("#### Configuration A")
        df_a = pd.DataFrame(st.session_state.compare_depots_a)
        edited_a = st.data_editor(
            df_a, num_rows="dynamic", column_config=depot_col_config,
            key="compare_editor_a",
        )
        st.session_state.compare_depots_a = edited_a.dropna(subset=["lat", "lon"]).to_dict("records")

    with col_b:
        st.markdown("#### Configuration B")
        df_b = pd.DataFrame(st.session_state.compare_depots_b)
        edited_b = st.data_editor(
            df_b, num_rows="dynamic", column_config=depot_col_config,
            key="compare_editor_b",
        )
        st.session_state.compare_depots_b = edited_b.dropna(subset=["lat", "lon"]).to_dict("records")

    if st.button("Run Comparison", type="primary", use_container_width=True):
        st.session_state.compare_run = True

    if st.session_state.get("compare_run"):
        depots_a = st.session_state.compare_depots_a
        depots_b = st.session_state.compare_depots_b

        if not depots_a or not depots_b:
            st.warning("Both configurations need at least one depot.")
            st.stop()

        with st.spinner("Running simulations..."):
            sim_a, ce_a = run_simulation(df, depots_a, config, override_lease=True)
            sim_b, ce_b = run_simulation(df, depots_b, config, override_lease=True)

        # --- Metrics side by side ---
        st.divider()

        def fmt_delta(val_b, val_a, fmt=",.0f", prefix="", suffix="", pct=False):
            diff = val_b - val_a
            sign = "+" if diff >= 0 else ""
            pct_str = ""
            if pct and val_a != 0:
                pct_str = f" ({sign}{diff/val_a*100:.1f}%)"
            return f"{sign}{prefix}{diff:{fmt}}{suffix}{pct_str}"

        col_a, col_b = st.columns(2)

        dh_a = sim_a["deadhead_miles"].sum()
        total_a = sim_a["distance_miles"].sum() + dh_a
        dh_pct_a = dh_a / total_a * 100

        dh_b = sim_b["deadhead_miles"].sum()
        total_b = sim_b["distance_miles"].sum() + dh_b
        dh_pct_b = dh_b / total_b * 100

        cpm_a = sim_a["total_cost"].sum() / sim_a["distance_miles"].sum()
        cpm_b = sim_b["total_cost"].sum() / sim_b["distance_miles"].sum()

        with col_a:
            st.markdown("#### Config A")
            m1, m2, m3 = st.columns(3)
            m1.metric("Avg cost/mile", f"${cpm_a:.2f}")
            m2.metric("Deadhead %", f"{dh_pct_a:.1f}%")
            m3.metric("Charge events", f"{len(ce_a):,}")

        with col_b:
            st.markdown("#### Config B")
            m1, m2, m3 = st.columns(3)
            m1.metric("Avg cost/mile", f"${cpm_b:.2f}", delta=fmt_delta(cpm_b, cpm_a, ".2f", "$"), delta_color="inverse")
            m2.metric("Deadhead %", f"{dh_pct_b:.1f}%", delta=f"{dh_pct_b - dh_pct_a:+.1f}pp", delta_color="inverse")
            m3.metric("Charge events", f"{len(ce_b):,}", delta=fmt_delta(len(ce_b), len(ce_a), ","), delta_color="inverse")

        # --- Stacked bars side by side ---
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(render_stacked_bar(sim_a, height=300), use_container_width=True)
        with col_b:
            st.plotly_chart(render_stacked_bar(sim_b, height=300), use_container_width=True)

        # --- Impact summary ---
        st.divider()
        st.markdown("#### Impact Summary: Config B vs A")

        comp_a = get_cost_components(sim_a)
        comp_b = get_cost_components(sim_b)
        rev_miles_a = sim_a["distance_miles"].sum()
        rev_miles_b = sim_b["distance_miles"].sum()

        all_keys = list(dict.fromkeys(list(comp_a.keys()) + list(comp_b.keys())))
        summary_rows = []
        for k in all_keys:
            cpm_val_a = comp_a.get(k, 0) / rev_miles_a
            cpm_val_b = comp_b.get(k, 0) / rev_miles_b
            diff = cpm_val_b - cpm_val_a
            summary_rows.append({
                "Component": k,
                "Config A ($/mi)": f"${cpm_val_a:.3f}",
                "Config B ($/mi)": f"${cpm_val_b:.3f}",
                "Difference": f"${diff:+.3f}",
            })
        total_a_cpm = sum(comp_a.values()) / rev_miles_a
        total_b_cpm = sum(comp_b.values()) / rev_miles_b
        summary_rows.append({
            "Component": "TOTAL",
            "Config A ($/mi)": f"${total_a_cpm:.3f}",
            "Config B ($/mi)": f"${total_b_cpm:.3f}",
            "Difference": f"${total_b_cpm - total_a_cpm:+.3f}",
        })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

        # --- Comparison Export ---
        st.divider()
        st.subheader("Export")
        export_col1, export_col2 = st.columns(2)

        with export_col1:
            summary_csv = pd.DataFrame(summary_rows).to_csv(index=False)
            st.download_button(
                "Download comparison CSV",
                data=summary_csv,
                file_name="ev_comparison_results.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with export_col2:
            try:
                from export import generate_comparison_pdf
                pdf_bytes = generate_comparison_pdf(sim_a, sim_b, config, depots_a, depots_b)
                if pdf_bytes:
                    st.download_button(
                        "Download comparison PDF",
                        data=pdf_bytes,
                        file_name="ev_comparison_report.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
            except ImportError:
                st.info("PDF export requires `fpdf2` and `kaleido`. Install with: `pip install fpdf2 kaleido`")
            except Exception as e:
                st.error(f"PDF generation failed: {e}")
