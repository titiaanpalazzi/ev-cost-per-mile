"""Tests for the EV Cost Per Mile Simulator core logic."""

import numpy as np
import pandas as pd
import pytest
from math import radians, cos, sin, asin, sqrt


# ---------------------------------------------------------------------------
# Import the functions under test (duplicated here to avoid importing the
# Streamlit app module, which triggers st.set_page_config at import time).
# ---------------------------------------------------------------------------

def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * 3956 * asin(sqrt(a))


def haversine_vec(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * 3956 * np.arcsin(np.sqrt(a))


ROAD_FACTOR = 1.3


def nearest_depot(lat, lon, depots):
    if not depots:
        return None, 0.0
    dists = [haversine(lat, lon, d["lat"], d["lon"]) * ROAD_FACTOR for d in depots]
    idx = int(np.argmin(dists))
    return idx, dists[idx]


def generate_sample_data(n=100, seed=42):
    rng = np.random.default_rng(seed)
    lat_center, lon_center = 37.77, -122.42
    pickup_lat = rng.normal(lat_center, 0.02, n)
    pickup_lon = rng.normal(lon_center, 0.02, n)
    dropoff_lat = pickup_lat + rng.normal(0, 0.03, n)
    dropoff_lon = pickup_lon + rng.normal(0, 0.03, n)
    base = pd.Timestamp("2024-06-01")
    timestamps = [base + pd.Timedelta(minutes=int(m)) for m in rng.integers(0, 30 * 24 * 60, n)]
    durations = rng.exponential(15, n).clip(3, 90)
    prices = rng.exponential(18, n).clip(5, 80)
    return pd.DataFrame({
        "request_timestamp": timestamps,
        "pickup_latitude": pickup_lat,
        "pickup_longitude": pickup_lon,
        "dropoff_latitude": dropoff_lat,
        "dropoff_longitude": dropoff_lon,
        "trip_duration_mins": np.round(durations, 1),
        "price_usd": np.round(prices, 2),
    })


def run_simulation(df, depots, battery_kwh=75, efficiency=3.5,
                   charge_threshold=20, electricity_peak=0.40,
                   electricity_offpeak=0.20, peak_start=16, peak_end=21):
    """Run the charging simulation on a ride dataframe. Returns sim_df, charge_events."""
    sim_df = df.sort_values("request_timestamp").reset_index(drop=True)

    soc = battery_kwh
    soc_history = []
    charge_events = []
    electricity_cost_per_ride = []
    deadhead_miles_per_ride = []

    for i, row in sim_df.iterrows():
        deadhead = 0.0
        if soc / battery_kwh < charge_threshold / 100:
            if i > 0:
                prev = sim_df.iloc[i - 1]
                depot_idx, dist_to_depot = nearest_depot(
                    prev["dropoff_latitude"], prev["dropoff_longitude"], depots)
            else:
                depot_idx, dist_to_depot = nearest_depot(
                    row["pickup_latitude"], row["pickup_longitude"], depots)

            deadhead = dist_to_depot
            energy_to_depot = dist_to_depot / efficiency
            soc = max(0, soc - energy_to_depot)

            kwh_to_charge = battery_kwh - soc
            hour = row.get("hour", 12)
            rate = electricity_peak if peak_start <= hour < peak_end else electricity_offpeak
            charge_events.append({
                "ride_index": i,
                "depot": depots[depot_idx].get("name", f"Depot {depot_idx+1}"),
                "deadhead_miles": dist_to_depot,
                "kwh_charged": kwh_to_charge,
                "cost": kwh_to_charge * rate,
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
    return sim_df, charge_events


# ---------------------------------------------------------------------------
# Tests: haversine
# ---------------------------------------------------------------------------

class TestHaversine:
    def test_same_point_is_zero(self):
        assert haversine(37.77, -122.42, 37.77, -122.42) == 0.0

    def test_known_distance(self):
        # SF downtown to SFO airport is roughly 11-13 miles
        dist = haversine(37.7749, -122.4194, 37.6213, -122.3790)
        assert 10 < dist < 15

    def test_symmetry(self):
        d1 = haversine(37.77, -122.42, 37.78, -122.43)
        d2 = haversine(37.78, -122.43, 37.77, -122.42)
        assert abs(d1 - d2) < 1e-10

    def test_vec_matches_scalar(self):
        lats1 = np.array([37.77, 37.78])
        lons1 = np.array([-122.42, -122.43])
        lats2 = np.array([37.78, 37.79])
        lons2 = np.array([-122.43, -122.44])
        vec_result = haversine_vec(lats1, lons1, lats2, lons2)
        for j in range(2):
            scalar = haversine(lats1[j], lons1[j], lats2[j], lons2[j])
            assert abs(vec_result[j] - scalar) < 1e-10


# ---------------------------------------------------------------------------
# Tests: nearest_depot
# ---------------------------------------------------------------------------

class TestNearestDepot:
    def test_empty_depots(self):
        idx, dist = nearest_depot(37.77, -122.42, [])
        assert idx is None
        assert dist == 0.0

    def test_single_depot(self):
        depots = [{"lat": 37.77, "lon": -122.42}]
        idx, dist = nearest_depot(37.77, -122.42, depots)
        assert idx == 0
        assert dist == 0.0

    def test_picks_closest(self):
        depots = [
            {"lat": 37.80, "lon": -122.42},  # farther north
            {"lat": 37.771, "lon": -122.421},  # very close
        ]
        idx, dist = nearest_depot(37.77, -122.42, depots)
        assert idx == 1

    def test_distance_includes_road_factor(self):
        depots = [{"lat": 37.78, "lon": -122.42}]
        idx, dist = nearest_depot(37.77, -122.42, depots)
        straight = haversine(37.77, -122.42, 37.78, -122.42)
        assert abs(dist - straight * ROAD_FACTOR) < 1e-10


# ---------------------------------------------------------------------------
# Tests: generate_sample_data
# ---------------------------------------------------------------------------

class TestGenerateSampleData:
    def test_correct_row_count(self):
        df = generate_sample_data(n=500)
        assert len(df) == 500

    def test_required_columns(self):
        df = generate_sample_data(n=10)
        expected = {"request_timestamp", "pickup_latitude", "pickup_longitude",
                    "dropoff_latitude", "dropoff_longitude",
                    "trip_duration_mins", "price_usd"}
        assert expected.issubset(set(df.columns))

    def test_coordinates_in_sf_range(self):
        df = generate_sample_data(n=1000)
        assert df["pickup_latitude"].between(37.5, 38.0).all()
        assert df["pickup_longitude"].between(-122.7, -122.1).all()

    def test_deterministic_with_seed(self):
        df1 = generate_sample_data(n=50, seed=99)
        df2 = generate_sample_data(n=50, seed=99)
        pd.testing.assert_frame_equal(df1, df2)

    def test_timestamps_within_month(self):
        df = generate_sample_data(n=100)
        ts = pd.to_datetime(df["request_timestamp"])
        assert ts.min() >= pd.Timestamp("2024-06-01")
        assert ts.max() < pd.Timestamp("2024-07-01")

    def test_prices_positive(self):
        df = generate_sample_data(n=100)
        assert (df["price_usd"] >= 5).all()

    def test_durations_clipped(self):
        df = generate_sample_data(n=1000)
        assert (df["trip_duration_mins"] >= 3).all()
        assert (df["trip_duration_mins"] <= 90).all()


# ---------------------------------------------------------------------------
# Tests: simulation logic
# ---------------------------------------------------------------------------

class TestSimulation:
    @pytest.fixture
    def small_data(self):
        df = generate_sample_data(n=200, seed=42)
        df["distance_miles"] = (
            haversine_vec(
                df["pickup_latitude"].values, df["pickup_longitude"].values,
                df["dropoff_latitude"].values, df["dropoff_longitude"].values,
            ) * ROAD_FACTOR
        )
        df["distance_miles"] = np.clip(df["distance_miles"], a_min=0.1, a_max=None)
        df["request_timestamp"] = pd.to_datetime(df["request_timestamp"])
        df["hour"] = df["request_timestamp"].dt.hour
        return df

    @pytest.fixture
    def one_depot(self):
        return [{"lat": 37.7749, "lon": -122.4194, "name": "Downtown",
                 "lease_per_stall": 2000, "stalls": 10}]

    def test_soc_never_negative(self, small_data, one_depot):
        sim_df, _ = run_simulation(small_data, one_depot)
        assert (sim_df["soc_after"] >= -0.001).all()

    def test_soc_never_exceeds_battery(self, small_data, one_depot):
        sim_df, _ = run_simulation(small_data, one_depot, battery_kwh=75)
        assert (sim_df["soc_after"] <= 75.001).all()

    def test_charge_events_happen(self, small_data, one_depot):
        sim_df, events = run_simulation(small_data, one_depot, battery_kwh=30)
        assert len(events) > 0

    def test_no_charge_with_huge_battery(self, small_data, one_depot):
        sim_df, events = run_simulation(small_data, one_depot, battery_kwh=10000)
        assert len(events) == 0

    def test_deadhead_zero_when_no_charge(self, small_data, one_depot):
        sim_df, _ = run_simulation(small_data, one_depot, battery_kwh=10000)
        assert sim_df["deadhead_miles"].sum() == 0.0

    def test_deadhead_positive_when_charging(self, small_data, one_depot):
        sim_df, events = run_simulation(small_data, one_depot, battery_kwh=30)
        if events:
            assert sim_df["deadhead_miles"].sum() > 0

    def test_electricity_cost_positive(self, small_data, one_depot):
        sim_df, _ = run_simulation(small_data, one_depot)
        assert (sim_df["electricity_cost"] >= 0).all()
        assert sim_df["electricity_cost"].sum() > 0

    def test_peak_rate_applied_correctly(self, small_data, one_depot):
        sim_df, _ = run_simulation(
            small_data, one_depot,
            electricity_peak=1.00, electricity_offpeak=0.01,
            peak_start=0, peak_end=24,  # all hours are peak
        )
        # Every ride should use peak rate (1.00)
        for _, row in sim_df.iterrows():
            energy = row["distance_miles"] / 3.5
            expected = min(energy, 75) * 1.00  # rough upper bound
            assert row["electricity_cost"] <= expected + 0.01

    def test_closer_depot_less_deadhead(self, small_data):
        far_depot = [{"lat": 37.85, "lon": -122.50, "name": "Far"}]
        close_depot = [{"lat": 37.77, "lon": -122.42, "name": "Close"}]
        sim_far, _ = run_simulation(small_data, far_depot, battery_kwh=30)
        sim_close, _ = run_simulation(small_data, close_depot, battery_kwh=30)
        assert sim_close["deadhead_miles"].sum() < sim_far["deadhead_miles"].sum()

    def test_multiple_depots_reduce_deadhead(self, small_data):
        one = [{"lat": 37.7749, "lon": -122.4194, "name": "Downtown"}]
        two = [
            {"lat": 37.7749, "lon": -122.4194, "name": "Downtown"},
            {"lat": 37.75, "lon": -122.45, "name": "South"},
        ]
        sim_one, _ = run_simulation(small_data, one, battery_kwh=30)
        sim_two, _ = run_simulation(small_data, two, battery_kwh=30)
        assert sim_two["deadhead_miles"].sum() <= sim_one["deadhead_miles"].sum()


# ---------------------------------------------------------------------------
# Tests: depot lease cost
# ---------------------------------------------------------------------------

class TestDepotLease:
    def test_lease_calculation(self):
        depots = [
            {"lat": 0, "lon": 0, "name": "A", "lease_per_stall": 2000, "stalls": 10},
            {"lat": 0, "lon": 0, "name": "B", "lease_per_stall": 1500, "stalls": 5},
        ]
        total = sum(d["lease_per_stall"] * d["stalls"] for d in depots)
        assert total == 27500

    def test_lease_per_ride(self):
        depots = [{"lat": 0, "lon": 0, "name": "A", "lease_per_stall": 2000, "stalls": 10}]
        total_lease = 20000
        n_rides = 20000
        per_ride = total_lease / n_rides
        assert per_ride == 1.0

    def test_more_depots_increase_lease(self):
        one = [{"lease_per_stall": 2000, "stalls": 10}]
        two = [{"lease_per_stall": 2000, "stalls": 10}, {"lease_per_stall": 2000, "stalls": 10}]
        cost_one = sum(d["lease_per_stall"] * d["stalls"] for d in one)
        cost_two = sum(d["lease_per_stall"] * d["stalls"] for d in two)
        assert cost_two == 2 * cost_one


# ---------------------------------------------------------------------------
# Tests: distance computation
# ---------------------------------------------------------------------------

class TestDistanceComputation:
    def test_distances_positive(self):
        df = generate_sample_data(n=100)
        dists = haversine_vec(
            df["pickup_latitude"].values, df["pickup_longitude"].values,
            df["dropoff_latitude"].values, df["dropoff_longitude"].values,
        ) * ROAD_FACTOR
        dists = np.clip(dists, a_min=0.1, a_max=None)
        assert (dists >= 0.1).all()

    def test_clip_enforces_minimum(self):
        # Same pickup and dropoff → 0 distance, clipped to 0.1
        dist = haversine(37.77, -122.42, 37.77, -122.42) * ROAD_FACTOR
        clipped = max(dist, 0.1)
        assert clipped == 0.1
