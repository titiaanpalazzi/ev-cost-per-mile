"""EV Cost-Per-Mile model — pure Python logic, no Streamlit dependency."""

from dataclasses import dataclass, field, asdict
from math import radians, cos, sin, asin, sqrt
from pathlib import Path
from typing import Optional
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# SimConfig — all cost/vehicle parameters in one place
# ---------------------------------------------------------------------------

ALL_COMPONENTS = frozenset({
    "electricity", "depreciation", "maintenance", "insurance",
    "tires", "cleaning", "tolls", "deadhead", "depot_lease", "opp_cost",
})


@dataclass
class SimConfig:
    # Vehicle
    battery_kwh: float = 75
    efficiency: float = 3.5  # miles per kWh
    charge_threshold: int = 20  # percent
    charge_speed_kw: float = 150

    # Cost — per-unit rates
    lease_per_stall: int = 800
    purchase_price: int = 85000
    lifetime_miles: int = 200000
    electricity_offpeak: float = 0.20
    electricity_peak: float = 0.40
    peak_start: int = 16
    peak_end: int = 21
    toll_per_mile: float = 0.25
    insurance_per_mile: float = 0.37
    tire_per_mile: float = 0.02
    cleaning_per_mile: float = 0.07
    maintenance_per_mile: float = 0.16
    opp_cost_per_mile: float = 2.00

    # Toggles
    enabled_components: set = field(default_factory=lambda: set(ALL_COMPONENTS))

    @property
    def depreciation_per_mile(self) -> float:
        return self.purchase_price / self.lifetime_miles if self.lifetime_miles else 0

    # -- Serialization --

    def to_dict(self) -> dict:
        d = asdict(self)
        d["enabled_components"] = sorted(d["enabled_components"])
        d["version"] = 1
        return d

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, d: dict) -> "SimConfig":
        d = dict(d)  # shallow copy
        d.pop("version", None)
        if "enabled_components" in d:
            d["enabled_components"] = set(d["enabled_components"])
        # Forward compatibility: drop unknown keys
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        d = {k: v for k, v in d.items() if k in valid}
        return cls(**d)

    @classmethod
    def from_json(cls, s: str) -> "SimConfig":
        return cls.from_dict(json.loads(s))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ROAD_FACTOR = 1.3


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


def nearest_depot(lat, lon, depots):
    """Return (index, distance_miles) to the nearest depot."""
    if not depots:
        return None, 0.0
    dists = [haversine(lat, lon, d["lat"], d["lon"]) * ROAD_FACTOR for d in depots]
    idx = int(np.argmin(dists))
    return idx, dists[idx]


# ---------------------------------------------------------------------------
# Geo boundary
# ---------------------------------------------------------------------------

def load_boundary(data_dir: Path):
    """Load land boundary polygon for filtering points. Returns a prepared geometry or None."""
    from shapely.geometry import shape
    from shapely.ops import unary_union
    from shapely.prepared import prep

    land_path = data_dir / "sf_land.geojson"
    if land_path.exists():
        with open(land_path) as f:
            geojson = json.load(f)
        polygons = [shape(feature["geometry"]) for feature in geojson["features"]]
        land = unary_union(polygons)
        return prep(land)

    boundary_path = data_dir / "sf_boundary.geojson"
    if boundary_path.exists():
        with open(boundary_path) as f:
            geojson = json.load(f)
        polygon = shape(geojson)
        return prep(polygon)

    return None


def point_on_land(lat, lon, boundary):
    """Check if a lat/lon point is within a prepared boundary."""
    if boundary is None:
        return True
    from shapely.geometry import Point
    return boundary.contains(Point(lon, lat))  # Point takes (x, y) = (lon, lat)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_sample_data(n=20000, seed=42, boundary=None):
    """Generate synthetic ride data calibrated to real Waymo ride distributions."""
    rng = np.random.default_rng(seed)

    lat_center, lon_center = 37.7611, -122.4383
    lat_std, lon_std = 0.0269, 0.0324

    oversample = 2.5
    n_candidates = int(n * oversample)

    pickup_lat = rng.normal(lat_center, lat_std, n_candidates)
    pickup_lon = rng.normal(lon_center, lon_std, n_candidates)

    dropoff_lat = pickup_lat + rng.normal(0, 0.043, n_candidates)
    dropoff_lon = pickup_lon + rng.normal(0, 0.043, n_candidates)

    if boundary is not None:
        valid = []
        for i in range(n_candidates):
            if (point_on_land(pickup_lat[i], pickup_lon[i], boundary) and
                    point_on_land(dropoff_lat[i], dropoff_lon[i], boundary)):
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


# ---------------------------------------------------------------------------
# CSV validation
# ---------------------------------------------------------------------------

COLUMN_MAP = {
    "pickup_lat": "pickup_latitude",
    "origin_lat": "pickup_latitude",
    "start_lat": "pickup_latitude",
    "lat": "pickup_latitude",
    "pickup_lon": "pickup_longitude",
    "pickup_lng": "pickup_longitude",
    "origin_lon": "pickup_longitude",
    "origin_lng": "pickup_longitude",
    "start_lon": "pickup_longitude",
    "start_lng": "pickup_longitude",
    "lng": "pickup_longitude",
    "dropoff_lat": "dropoff_latitude",
    "dest_lat": "dropoff_latitude",
    "end_lat": "dropoff_latitude",
    "dropoff_lon": "dropoff_longitude",
    "dropoff_lng": "dropoff_longitude",
    "dest_lon": "dropoff_longitude",
    "dest_lng": "dropoff_longitude",
    "end_lon": "dropoff_longitude",
    "end_lng": "dropoff_longitude",
}

REQUIRED_COLUMNS = ["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"]

MAX_RIDES = 50_000


def validate_csv(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Validate and normalize a ride data CSV.

    Returns (normalized_df, errors, warnings).
    If errors is non-empty, normalized_df should not be used.
    """
    errors = []
    warnings = []

    # Normalize column names: lowercase, strip whitespace
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    # Auto-detect column mapping
    for src, dst in COLUMN_MAP.items():
        if src in df.columns and dst not in df.columns:
            df = df.rename(columns={src: dst})

    # Check required columns
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {', '.join(missing)}")
        return df, errors, warnings

    # Check for non-numeric values in coordinate columns
    for col in REQUIRED_COLUMNS:
        non_numeric = pd.to_numeric(df[col], errors="coerce").isna() & df[col].notna()
        count = non_numeric.sum()
        if count > 0:
            errors.append(f"Column '{col}' contains {count} non-numeric values")

    # Check coordinate ranges
    if not errors:
        for col, lo, hi in [
            ("pickup_latitude", -90, 90), ("pickup_longitude", -180, 180),
            ("dropoff_latitude", -90, 90), ("dropoff_longitude", -180, 180),
        ]:
            vals = pd.to_numeric(df[col], errors="coerce")
            out_of_range = ((vals < lo) | (vals > hi)).sum()
            if out_of_range > 0:
                errors.append(f"Column '{col}' has {out_of_range} values outside [{lo}, {hi}]")

    # Minimum ride count
    if len(df) < 100:
        errors.append(f"Need at least 100 rides, got {len(df)}")

    if errors:
        return df, errors, warnings

    # Convert coordinate columns to numeric
    for col in REQUIRED_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Handle optional columns
    if "request_timestamp" not in df.columns:
        rng = np.random.default_rng(42)
        base = pd.Timestamp("2024-06-01")
        df["request_timestamp"] = [
            base + pd.Timedelta(minutes=int(m))
            for m in rng.integers(0, 30 * 24 * 60, len(df))
        ]
        warnings.append("No timestamps in data — electricity costs assume uniform time distribution (no peak/off-peak distinction).")

    # Sampling cap
    if len(df) > MAX_RIDES:
        warnings.append(f"Sampled {MAX_RIDES:,} of {len(df):,} rides for performance. Full dataset support coming soon.")
        df = df.sample(n=MAX_RIDES, random_state=42).reset_index(drop=True)

    return df, errors, warnings


def prepare_rides(df: pd.DataFrame) -> pd.DataFrame:
    """Add computed columns (distance_miles, hour) to a ride dataframe."""
    df = df.copy()
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


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def run_simulation(df, depots, config: SimConfig, override_lease=False):
    """Run the cost simulation for a given set of depots.

    Returns (sim_df, charge_events).
    If override_lease is True, use per-depot lease_per_stall from the depot dicts.
    """
    sim_df = (df.sort_values("request_timestamp").reset_index(drop=True)
              if "request_timestamp" in df.columns
              else df.reset_index(drop=True))

    enabled = config.enabled_components
    soc = config.battery_kwh
    soc_history = []
    charge_events = []
    electricity_cost_per_ride = []
    deadhead_miles_per_ride = []

    for i, row in sim_df.iterrows():
        deadhead = 0.0

        if soc / config.battery_kwh < config.charge_threshold / 100:
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
            energy_to_depot = dist_to_depot / config.efficiency
            soc = max(0, soc - energy_to_depot)

            kwh_to_charge = config.battery_kwh - soc
            hour = row.get("hour", 12)
            rate = (config.electricity_peak
                    if config.peak_start <= hour < config.peak_end
                    else config.electricity_offpeak)
            cost = kwh_to_charge * rate
            charge_events.append({
                "ride_index": i,
                "depot": depots[depot_idx].get("name", f"Depot {depot_idx + 1}"),
                "deadhead_miles": dist_to_depot,
                "kwh_charged": kwh_to_charge,
                "cost": cost,
                "rate": rate,
            })
            soc = config.battery_kwh

        energy_needed = row["distance_miles"] / config.efficiency
        energy_used = min(energy_needed, soc)
        soc -= energy_used

        hour = row.get("hour", 12)
        rate = (config.electricity_peak
                if config.peak_start <= hour < config.peak_end
                else config.electricity_offpeak)
        electricity_cost_per_ride.append(energy_used * rate)
        deadhead_miles_per_ride.append(deadhead)
        soc_history.append(soc)

    sim_df["soc_after"] = soc_history
    sim_df["electricity_cost"] = electricity_cost_per_ride
    sim_df["deadhead_miles"] = deadhead_miles_per_ride

    dep = config.depreciation_per_mile

    # Cost breakdown per ride
    sim_df["cost_electricity"] = sim_df["electricity_cost"] if "electricity" in enabled else 0
    sim_df["cost_depreciation"] = sim_df["distance_miles"] * dep if "depreciation" in enabled else 0
    sim_df["cost_maintenance"] = sim_df["distance_miles"] * config.maintenance_per_mile if "maintenance" in enabled else 0
    sim_df["cost_insurance"] = sim_df["distance_miles"] * config.insurance_per_mile if "insurance" in enabled else 0
    sim_df["cost_tires"] = sim_df["distance_miles"] * config.tire_per_mile if "tires" in enabled else 0
    sim_df["cost_cleaning"] = sim_df["distance_miles"] * config.cleaning_per_mile if "cleaning" in enabled else 0
    sim_df["cost_tolls"] = sim_df["distance_miles"] * config.toll_per_mile if "tolls" in enabled else 0

    sim_df["cost_deadhead"] = sim_df["deadhead_miles"] * (
        (dep if "depreciation" in enabled else 0)
        + (config.maintenance_per_mile if "maintenance" in enabled else 0)
        + (config.insurance_per_mile if "insurance" in enabled else 0)
        + (config.tire_per_mile if "tires" in enabled else 0)
        + (config.toll_per_mile if "tolls" in enabled else 0)
    ) if "deadhead" in enabled else 0
    if "deadhead" in enabled and "electricity" in enabled:
        sim_df["cost_deadhead"] = (
            sim_df["cost_deadhead"]
            + sim_df["deadhead_miles"] / config.efficiency * config.electricity_offpeak
        )

    sim_df["cost_opp_cost"] = (
        sim_df["deadhead_miles"] * config.opp_cost_per_mile
        if "opp_cost" in enabled else 0
    )

    # Depot lease
    if override_lease:
        total_monthly_lease = sum(
            d.get("lease_per_stall", config.lease_per_stall) * d.get("stalls", 0)
            for d in depots
        )
    else:
        total_monthly_lease = sum(
            config.lease_per_stall * d.get("stalls", 0) for d in depots
        )
    sim_df["cost_depot_lease"] = total_monthly_lease / len(sim_df) if "depot_lease" in enabled else 0

    sim_df["total_cost"] = (
        sim_df["cost_electricity"]
        + sim_df["cost_depreciation"]
        + sim_df["cost_maintenance"]
        + sim_df["cost_insurance"]
        + sim_df["cost_tires"]
        + sim_df["cost_cleaning"]
        + sim_df["cost_tolls"]
        + sim_df["cost_deadhead"]
        + sim_df["cost_opp_cost"]
        + sim_df["cost_depot_lease"]
    )
    sim_df["cost_per_mile"] = sim_df["total_cost"] / sim_df["distance_miles"]

    return sim_df, charge_events


# ---------------------------------------------------------------------------
# Cost components & chart
# ---------------------------------------------------------------------------

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
        "Deadhead": sim_df["cost_deadhead"].sum() + sim_df["cost_opp_cost"].sum(),
        "Depot Lease": sim_df["cost_depot_lease"].sum(),
    }
    return {k: v for k, v in components.items() if v > 0}


CATEGORY_COLORS = {
    "Electricity": "#636EFA",
    "Vehicle Depreciation": "#EF553B",
    "Maintenance": "#00CC96",
    "Insurance": "#AB63FA",
    "Tires": "#FFA15A",
    "Cleaning & Plug-ins": "#B6E880",
    "Tolls": "#FECB52",
    "Deadhead": "#FF6692",
    "Depot Lease": "#19D3F3",
}


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
            marker_color=CATEGORY_COLORS.get(name, "#888888"),
            textposition="none",
            legendrank=i,
        ))
        mid_x = cumulative + val / 2
        is_small = val / total_cpm < 0.10 if total_cpm > 0 else False
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
        legend=dict(traceorder="normal"),
    )
    return fig
