import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from math import radians, cos, sin, asin, sqrt
from pathlib import Path
import json
from shapely.geometry import shape, Point
from shapely.prepared import prep

st.set_page_config(page_title="EV Cost Per Mile", layout="wide")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def haversine(lat1, lon1, lat2, lon2):
    """Distance in miles between two lat/lon points."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * 3956 * asin(sqrt(a))


def haversine_vec(lat1, lon1, lat2, lon2):
    """Vectorized haversine for arrays."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * 3956 * np.arcsin(np.sqrt(a))


ROAD_FACTOR = 1.3


def nearest_depot(lat, lon, depots):
    """Return (index, distance_miles) to the nearest depot."""
    if not depots:
        return None, 0.0
    dists = [haversine(lat, lon, d["lat"], d["lon"]) * ROAD_FACTOR for d in depots]
    idx = int(np.argmin(dists))
    return idx, dists[idx]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_sf_boundary():
    """Load SF land boundary polygon for filtering points."""
    from shapely.ops import unary_union

    # Prefer land-clipped census tracts (excludes water)
    land_path = Path(__file__).parent / "data" / "sf_land.geojson"
    if land_path.exists():
        with open(land_path) as f:
            geojson = json.load(f)
        # Union all tract polygons to get a single land boundary
        polygons = [shape(feature["geometry"]) for feature in geojson["features"]]
        land = unary_union(polygons)
        return prep(land)

    # Fallback to old boundary file (includes water)
    boundary_path = Path(__file__).parent / "data" / "sf_boundary.geojson"
    if boundary_path.exists():
        with open(boundary_path) as f:
            geojson = json.load(f)
        polygon = shape(geojson)
        return prep(polygon)

    return None


SF_BOUNDARY = load_sf_boundary()


def point_on_land(lat, lon):
    """Check if a lat/lon point is within SF land boundary."""
    if SF_BOUNDARY is None:
        return True  # No boundary file, accept all points
    return SF_BOUNDARY.contains(Point(lon, lat))  # Note: Point takes (x, y) = (lon, lat)


def generate_sample_data(n=20000, seed=42):
    """Generate synthetic ride data calibrated to real Waymo ride distributions."""
    rng = np.random.default_rng(seed)

    # Calibrated from Waymo SF data: center, spread, and dropoff offsets
    lat_center, lon_center = 37.7611, -122.4383
    lat_std, lon_std = 0.0269, 0.0324

    # Generate more candidates than needed, then filter to land
    oversample = 2.5
    n_candidates = int(n * oversample)

    pickup_lat = rng.normal(lat_center, lat_std, n_candidates)
    pickup_lon = rng.normal(lon_center, lon_std, n_candidates)

    # Dropoff offset calibrated to match Waymo ~4.2 mi mean, ~4.1 mi median distance
    dropoff_lat = pickup_lat + rng.normal(0, 0.043, n_candidates)
    dropoff_lon = pickup_lon + rng.normal(0, 0.043, n_candidates)

    # Filter to points on land
    if SF_BOUNDARY is not None:
        valid = []
        for i in range(n_candidates):
            if (point_on_land(pickup_lat[i], pickup_lon[i]) and
                point_on_land(dropoff_lat[i], dropoff_lon[i])):
                valid.append(i)
            if len(valid) >= n:
                break
        valid = valid[:n]
        pickup_lat = pickup_lat[valid]
        pickup_lon = pickup_lon[valid]
        dropoff_lat = dropoff_lat[valid]
        dropoff_lon = dropoff_lon[valid]
        actual_n = len(valid)
    else:
        actual_n = n
        pickup_lat = pickup_lat[:n]
        pickup_lon = pickup_lon[:n]
        dropoff_lat = dropoff_lat[:n]
        dropoff_lon = dropoff_lon[:n]

    base = pd.Timestamp("2024-06-01")
    timestamps = [base + pd.Timedelta(minutes=int(m)) for m in rng.integers(0, 30 * 24 * 60, actual_n)]
    # Duration and price calibrated from Waymo: mean ~22 min, median ~21; mean ~$19.7, median ~$18.6
    durations = rng.exponential(18, actual_n).clip(3, 90)
    prices = rng.exponential(16, actual_n).clip(5, 80)
    return pd.DataFrame({
        "request_timestamp": timestamps,
        "pickup_latitude": pickup_lat,
        "pickup_longitude": pickup_lon,
        "dropoff_latitude": dropoff_lat,
        "dropoff_longitude": dropoff_lon,
        "trip_duration_mins": np.round(durations, 1),
        "price_usd": np.round(prices, 2),
    })


@st.cache_data
def load_data(n_rides=60000, _version=5):  # bump version to invalidate cache
    data_dir = Path(__file__).parent / "data"
    csv_path = data_dir / "sf_waymo_estimates.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        df = generate_sample_data(n=n_rides)
        st.info(
            "Using **simulated rides** based on "
            "[this Kaggle dataset](https://www.kaggle.com/datasets/npurav/waymo-rides-estimates) "
            "of actual Waymo rides."
        )
    df["distance_miles"] = (
        haversine_vec(
            df["pickup_latitude"].values, df["pickup_longitude"].values,
            df["dropoff_latitude"].values, df["dropoff_longitude"].values,
        ) * ROAD_FACTOR
    )
    df["distance_miles"] = np.clip(df["distance_miles"], a_min=0.1, a_max=None)
    if "request_timestamp" in df.columns:
        df["request_timestamp"] = pd.to_datetime(df["request_timestamp"], errors="coerce")
        df["hour"] = df["request_timestamp"].dt.hour
    return df


df = load_data()

# ---------------------------------------------------------------------------
# Sidebar — vehicle & cost parameters
# ---------------------------------------------------------------------------

st.sidebar.header("Vehicle Parameters")
battery_kwh = st.sidebar.slider("Battery capacity (kWh)", 30, 150, 75)
efficiency = st.sidebar.slider("Efficiency (miles / kWh)", 2.0, 5.0, 3.5, 0.1)
charge_threshold = st.sidebar.slider("Charge when battery below (%)", 5, 40, 20)
charge_speed_kw = st.sidebar.slider("Charging speed (kW)", 7, 250, 150)

st.sidebar.header("Cost Parameters")
lease_per_stall = st.sidebar.number_input("Lease price ($/stall/mo)", 0, 50000, 800, 100, format="%d")
purchase_price = st.sidebar.number_input("Vehicle purchase price ($)", 20000, 150000, 85000, 1000)
lifetime_miles = st.sidebar.number_input("Expected lifetime miles", 50000, 500000, 200000, 10000)
electricity_offpeak = st.sidebar.slider("Off-peak electricity ($/kWh)", 0.05, 0.50, 0.20, 0.01)
electricity_peak = st.sidebar.slider("Peak electricity ($/kWh)", 0.10, 0.80, 0.40, 0.01)
peak_start, peak_end = 16, 21
toll_per_mile = st.sidebar.slider("Tolls ($/mile)", 0.10, 0.40, 0.25, 0.01)
insurance_per_mile = st.sidebar.slider("Insurance ($/mile)", 0.10, 0.50, 0.37, 0.01)
tire_per_mile = st.sidebar.slider("Tire wear ($/mile)", 0.00, 0.06, 0.02, 0.005)

# Fixed per-mile costs (no UI controls)
cleaning_per_mile = 0.07
maintenance_per_mile = 0.16

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

# ---------------------------------------------------------------------------
# Simulation function (reusable)
# ---------------------------------------------------------------------------

depreciation_per_mile = purchase_price / lifetime_miles


def run_simulation(df, depots):
    """Run the cost simulation for a given set of depots. Returns (sim_df, charge_events)."""
    sim_df = df.sort_values("request_timestamp").reset_index(drop=True) if "request_timestamp" in df.columns else df.reset_index(drop=True)

    soc = battery_kwh
    soc_history = []
    charge_events = []
    electricity_cost_per_ride = []
    deadhead_miles_per_ride = []

    for i, row in sim_df.iterrows():
        deadhead = 0.0

        # Check if we need to charge
        if soc / battery_kwh < charge_threshold / 100:
            if i > 0:
                prev = sim_df.iloc[i - 1]
                depot_idx, dist_to_depot = nearest_depot(
                    prev["dropoff_latitude"], prev["dropoff_longitude"], depots
                )
            else:
                depot_idx, dist_to_depot = nearest_depot(
                    row["pickup_latitude"], row["pickup_longitude"], depots
                )

            deadhead = dist_to_depot
            energy_to_depot = dist_to_depot / efficiency
            soc = max(0, soc - energy_to_depot)

            kwh_to_charge = battery_kwh - soc
            hour = row.get("hour", 12)
            rate = electricity_peak if peak_start <= hour < peak_end else electricity_offpeak
            cost = kwh_to_charge * rate
            charge_events.append({
                "ride_index": i,
                "depot": depots[depot_idx].get("name", f"Depot {depot_idx+1}"),
                "deadhead_miles": dist_to_depot,
                "kwh_charged": kwh_to_charge,
                "cost": cost,
                "rate": rate,
            })
            soc = battery_kwh

        energy_needed = row["distance_miles"] / efficiency
        energy_used = min(energy_needed, soc)
        soc -= energy_used

        hour = row.get("hour", 12)
        rate = electricity_peak if peak_start <= hour < peak_end else electricity_offpeak
        electricity_cost_per_ride.append(energy_used * rate)
        deadhead_miles_per_ride.append(deadhead)
        soc_history.append(soc)

    sim_df["soc_after"] = soc_history
    sim_df["electricity_cost"] = electricity_cost_per_ride
    sim_df["deadhead_miles"] = deadhead_miles_per_ride

    # Cost breakdown per ride
    sim_df["cost_electricity"] = sim_df["electricity_cost"] if include_electricity else 0
    sim_df["cost_depreciation"] = sim_df["distance_miles"] * depreciation_per_mile if include_depreciation else 0
    sim_df["cost_maintenance"] = sim_df["distance_miles"] * maintenance_per_mile if include_maintenance else 0
    sim_df["cost_insurance"] = sim_df["distance_miles"] * insurance_per_mile if include_insurance else 0
    sim_df["cost_tires"] = sim_df["distance_miles"] * tire_per_mile if include_tires else 0
    sim_df["cost_cleaning"] = sim_df["distance_miles"] * cleaning_per_mile if include_cleaning else 0
    sim_df["cost_tolls"] = sim_df["distance_miles"] * toll_per_mile if include_tolls else 0
    sim_df["cost_deadhead"] = sim_df["deadhead_miles"] * (
        (depreciation_per_mile if include_depreciation else 0)
        + (maintenance_per_mile if include_maintenance else 0)
        + (insurance_per_mile if include_insurance else 0)
        + (tire_per_mile if include_tires else 0)
    ) if include_deadhead else 0
    if include_deadhead and include_electricity:
        sim_df["cost_deadhead"] = sim_df["cost_deadhead"] + sim_df["deadhead_miles"] / efficiency * electricity_offpeak

    total_monthly_lease = sum(
        lease_per_stall * d.get("stalls", 0) for d in depots
    )
    sim_df["cost_depot_lease"] = total_monthly_lease / len(sim_df) if include_depot_lease else 0

    sim_df["total_cost"] = (
        sim_df["cost_electricity"]
        + sim_df["cost_depreciation"]
        + sim_df["cost_maintenance"]
        + sim_df["cost_insurance"]
        + sim_df["cost_tires"]
        + sim_df["cost_cleaning"]
        + sim_df["cost_tolls"]
        + sim_df["cost_deadhead"]
        + sim_df["cost_depot_lease"]
    )
    sim_df["cost_per_mile"] = sim_df["total_cost"] / sim_df["distance_miles"]

    return sim_df, charge_events


def get_cost_components(sim_df):
    """Return dict of cost component totals."""
    components = {
        "Electricity": sim_df["cost_electricity"].sum(),
        "Vehicle Depreciation": sim_df["cost_depreciation"].sum(),
        "Maintenance": sim_df["cost_maintenance"].sum(),
        "Insurance": sim_df["cost_insurance"].sum(),
        "Tires": sim_df["cost_tires"].sum(),
        "Cleaning & Plug-ins": sim_df["cost_cleaning"].sum(),
        "Tolls": sim_df["cost_tolls"].sum(),
        "Deadhead": sim_df["cost_deadhead"].sum(),
        "Depot Lease": sim_df["cost_depot_lease"].sum(),
    }
    return {k: v for k, v in components.items() if v > 0}


COLORS = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#B6E880", "#FECB52", "#FF6692", "#19D3F3"]


def render_stacked_bar(sim_df, height=350):
    """Render the stacked horizontal cost-per-mile bar chart."""
    components = get_cost_components(sim_df)
    total_rev_miles = sim_df["distance_miles"].sum()
    cpm = {k: v / total_rev_miles for k, v in components.items()}

    fig = go.Figure()
    cumulative = 0
    total_cpm = sum(cpm.values())
    items = sorted(cpm.items(), key=lambda x: x[1], reverse=True)
    small_offset_idx = 0
    for i, (name, val) in enumerate(items):
        fig.add_trace(go.Bar(
            y=["Cost per mile"],
            x=[val],
            name=name,
            orientation="h",
            marker_color=COLORS[i % len(COLORS)],
            textposition="none",
        ))
        mid_x = cumulative + val / 2
        is_small = val / total_cpm < 0.10
        if is_small:
            y_offset = 45 + 25 * small_offset_idx if small_offset_idx % 2 == 0 else -(45 + 25 * (small_offset_idx - 1))
            small_offset_idx += 1
            fig.add_annotation(
                x=mid_x, y=0,
                text=f"<b>{name}</b>: ${val:.2f}",
                showarrow=True,
                arrowhead=2,
                arrowwidth=1,
                arrowcolor="#888",
                ax=0, ay=y_offset,
                font=dict(size=12, color="black"),
                xanchor="center",
            )
        else:
            fig.add_annotation(
                x=mid_x, y=0,
                text=f"<b>{name}</b><br>${val:.2f}",
                showarrow=False,
                font=dict(size=12, color="black"),
                xanchor="center",
            )
        cumulative += val
    fig.add_annotation(
        x=cumulative, y=0,
        text=f"  <b>Total: ${cumulative:.2f}/mi</b>",
        showarrow=False,
        font=dict(size=13),
        xanchor="left",
    )
    fig.update_layout(
        barmode="stack",
        title="Cost Per Revenue Mile by Component",
        xaxis_title="$/mile",
        yaxis=dict(visible=False),
        height=height,
        margin=dict(l=20, r=150, t=40, b=80),
    )
    return fig

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
        {"lat": 37.7749, "lon": -122.4194, "name": "Downtown", "stalls": 10},
    ]

    if "depots" not in st.session_state:
        st.session_state.depots = list(DEFAULT_DEPOTS)

    col_map, col_edit = st.columns([2, 1])

    with col_map:
        grid_lat = np.linspace(37.70, 37.84, 40)
        grid_lon = np.linspace(-122.52, -122.34, 40)
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
            mapbox=dict(style="open-street-map", center=dict(lat=37.77, lon=-122.42), zoom=11),
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
                "lat": st.column_config.NumberColumn("Latitude", min_value=37.5, max_value=38.0, step=0.001, format="%.4f"),
                "lon": st.column_config.NumberColumn("Longitude", min_value=-122.6, max_value=-122.2, step=0.001, format="%.4f"),
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
    sim_df, charge_events = run_simulation(df, depots)

    # --- Metrics ---
    st.divider()
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total rides", f"{len(sim_df):,}", help="Simulated over one month period.")
    col2.metric("Revenue miles", f"{sim_df['distance_miles'].sum():,.0f}")
    _dh = sim_df["deadhead_miles"].sum()
    _total = sim_df["distance_miles"].sum() + _dh
    col3.metric("Deadhead miles", f"{_dh:,.0f} ({_dh/_total*100:.1f}%)")
    col4.metric("Avg cost/mile", f"${sim_df['cost_per_mile'].mean():.3f}")
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

# =========================================================================
# PAGE: Compare Depot Locations
# =========================================================================

elif page == "Compare Depot Locations":

    st.subheader("Compare Depot Configurations")
    st.caption("This assumes the same simulated rides as the Simulator page. Edit the depot tables below, then press **Run Comparison** to simulate both configurations side by side.")

    # Defaults for the two configs
    if "compare_depots_a" not in st.session_state:
        st.session_state.compare_depots_a = [
            {"name": "Downtown", "lat": 37.7749, "lon": -122.4194, "stalls": 20},
        ]
    if "compare_depots_b" not in st.session_state:
        st.session_state.compare_depots_b = [
            {"name": "San Bruno", "lat": 37.6305, "lon": -122.4111, "stalls": 20},
        ]

    col_a, col_b = st.columns(2)

    depot_col_config = {
        "name": st.column_config.TextColumn("Name"),
        "lat": st.column_config.NumberColumn("Latitude", min_value=36.0, max_value=39.0, step=0.001, format="%.4f"),
        "lon": st.column_config.NumberColumn("Longitude", min_value=-123.5, max_value=-121.0, step=0.001, format="%.4f"),
        "stalls": st.column_config.NumberColumn("Stalls", min_value=1, max_value=500, step=1),
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
            sim_a, ce_a = run_simulation(df, depots_a)
            sim_b, ce_b = run_simulation(df, depots_b)

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

        cpm_a = sim_a["cost_per_mile"].mean()
        cpm_b = sim_b["cost_per_mile"].mean()

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
