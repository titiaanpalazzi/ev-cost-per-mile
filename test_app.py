"""Tests for the EV Cost Per Mile Simulator core logic."""

import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from ev_model import (
    SimConfig, ALL_COMPONENTS,
    haversine, haversine_vec, nearest_depot, ROAD_FACTOR,
    generate_sample_data, prepare_rides,
    validate_csv, COLUMN_MAP, REQUIRED_COLUMNS, MAX_RIDES,
    run_simulation, get_cost_components, render_stacked_bar,
)


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
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_rides():
    df = generate_sample_data(n=200, seed=42)
    return prepare_rides(df)


@pytest.fixture
def one_depot():
    return [{"lat": 37.7749, "lon": -122.4194, "name": "Downtown",
             "lease_per_stall": 2000, "stalls": 10}]


@pytest.fixture
def default_config():
    return SimConfig()


# ---------------------------------------------------------------------------
# Tests: simulation logic
# ---------------------------------------------------------------------------

class TestSimulation:
    def test_soc_never_negative(self, small_rides, one_depot, default_config):
        sim_df, _ = run_simulation(small_rides, one_depot, default_config)
        assert (sim_df["soc_after"] >= -0.001).all()

    def test_soc_never_exceeds_battery(self, small_rides, one_depot, default_config):
        sim_df, _ = run_simulation(small_rides, one_depot, default_config)
        assert (sim_df["soc_after"] <= default_config.battery_kwh + 0.001).all()

    def test_charge_events_happen(self, small_rides, one_depot):
        config = SimConfig(battery_kwh=30)
        sim_df, events = run_simulation(small_rides, one_depot, config)
        assert len(events) > 0

    def test_no_charge_with_huge_battery(self, small_rides, one_depot):
        config = SimConfig(battery_kwh=10000)
        sim_df, events = run_simulation(small_rides, one_depot, config)
        assert len(events) == 0

    def test_deadhead_zero_when_no_charge(self, small_rides, one_depot):
        config = SimConfig(battery_kwh=10000)
        sim_df, _ = run_simulation(small_rides, one_depot, config)
        assert sim_df["deadhead_miles"].sum() == 0.0

    def test_deadhead_positive_when_charging(self, small_rides, one_depot):
        config = SimConfig(battery_kwh=30)
        sim_df, events = run_simulation(small_rides, one_depot, config)
        if events:
            assert sim_df["deadhead_miles"].sum() > 0

    def test_electricity_cost_positive(self, small_rides, one_depot, default_config):
        sim_df, _ = run_simulation(small_rides, one_depot, default_config)
        assert (sim_df["electricity_cost"] >= 0).all()
        assert sim_df["electricity_cost"].sum() > 0

    def test_peak_rate_applied_correctly(self, small_rides, one_depot):
        config = SimConfig(electricity_peak=1.00, electricity_offpeak=0.01,
                           peak_start=0, peak_end=24)
        sim_df, _ = run_simulation(small_rides, one_depot, config)
        for _, row in sim_df.iterrows():
            energy = row["distance_miles"] / 3.5
            expected = min(energy, 75) * 1.00
            assert row["electricity_cost"] <= expected + 0.01

    def test_closer_depot_less_deadhead(self, small_rides):
        config = SimConfig(battery_kwh=30)
        far_depot = [{"lat": 37.85, "lon": -122.50, "name": "Far"}]
        close_depot = [{"lat": 37.77, "lon": -122.42, "name": "Close"}]
        sim_far, _ = run_simulation(small_rides, far_depot, config)
        sim_close, _ = run_simulation(small_rides, close_depot, config)
        assert sim_close["deadhead_miles"].sum() < sim_far["deadhead_miles"].sum()

    def test_multiple_depots_reduce_deadhead(self, small_rides):
        config = SimConfig(battery_kwh=30)
        one = [{"lat": 37.7749, "lon": -122.4194, "name": "Downtown"}]
        two = [
            {"lat": 37.7749, "lon": -122.4194, "name": "Downtown"},
            {"lat": 37.75, "lon": -122.45, "name": "South"},
        ]
        sim_one, _ = run_simulation(small_rides, one, config)
        sim_two, _ = run_simulation(small_rides, two, config)
        assert sim_two["deadhead_miles"].sum() <= sim_one["deadhead_miles"].sum()


# ---------------------------------------------------------------------------
# Tests: cost column correctness (NEW)
# ---------------------------------------------------------------------------

class TestCostColumns:
    def test_cost_per_mile_formula(self, small_rides, one_depot, default_config):
        sim_df, _ = run_simulation(small_rides, one_depot, default_config)
        expected_cpm = sim_df["total_cost"] / sim_df["distance_miles"]
        pd.testing.assert_series_equal(sim_df["cost_per_mile"], expected_cpm, check_names=False)

    def test_total_cost_is_sum_of_components(self, small_rides, one_depot, default_config):
        sim_df, _ = run_simulation(small_rides, one_depot, default_config)
        component_sum = (
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
        pd.testing.assert_series_equal(sim_df["total_cost"], component_sum, check_names=False)

    def test_depreciation_uses_purchase_and_lifetime(self, small_rides, one_depot):
        config = SimConfig(purchase_price=100000, lifetime_miles=100000)
        sim_df, _ = run_simulation(small_rides, one_depot, config)
        # depreciation_per_mile = 100000/100000 = 1.0
        expected = sim_df["distance_miles"] * 1.0
        pd.testing.assert_series_equal(sim_df["cost_depreciation"], expected, check_names=False)

    def test_disabled_component_is_zero(self, small_rides, one_depot):
        config = SimConfig(enabled_components={"electricity"})
        sim_df, _ = run_simulation(small_rides, one_depot, config)
        assert sim_df["cost_depreciation"].sum() == 0
        assert sim_df["cost_maintenance"].sum() == 0
        assert sim_df["cost_insurance"].sum() == 0
        assert sim_df["cost_tolls"].sum() == 0
        assert sim_df["cost_depot_lease"].sum() == 0
        # Electricity should still be positive
        assert sim_df["cost_electricity"].sum() > 0

    def test_deadhead_includes_electricity(self, small_rides, one_depot):
        config = SimConfig(battery_kwh=30)
        sim_df, events = run_simulation(small_rides, one_depot, config)
        if events:
            # Deadhead cost should include electricity component
            dh_rows = sim_df[sim_df["deadhead_miles"] > 0]
            assert (dh_rows["cost_deadhead"] > 0).all()

    def test_override_lease_per_depot(self, small_rides):
        depots = [
            {"lat": 37.77, "lon": -122.42, "name": "A", "stalls": 10, "lease_per_stall": 1000},
            {"lat": 37.78, "lon": -122.43, "name": "B", "stalls": 5, "lease_per_stall": 2000},
        ]
        config = SimConfig(lease_per_stall=500)  # sidebar default, should be overridden
        sim_df, _ = run_simulation(small_rides, depots, config, override_lease=True)
        # Total lease = 1000*10 + 2000*5 = 20000
        expected_per_ride = 20000 / len(sim_df)
        assert abs(sim_df["cost_depot_lease"].iloc[0] - expected_per_ride) < 0.01

    def test_all_components_disabled_zero_cost(self, small_rides, one_depot):
        config = SimConfig(enabled_components=set())
        sim_df, _ = run_simulation(small_rides, one_depot, config)
        assert sim_df["total_cost"].sum() == 0


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
        dist = haversine(37.77, -122.42, 37.77, -122.42) * ROAD_FACTOR
        clipped = max(dist, 0.1)
        assert clipped == 0.1


# ---------------------------------------------------------------------------
# Tests: SimConfig (NEW)
# ---------------------------------------------------------------------------

class TestSimConfig:
    def test_default_values(self):
        config = SimConfig()
        assert config.battery_kwh == 75
        assert config.efficiency == 3.5
        assert config.enabled_components == ALL_COMPONENTS

    def test_depreciation_per_mile(self):
        config = SimConfig(purchase_price=100000, lifetime_miles=200000)
        assert config.depreciation_per_mile == 0.5

    def test_json_round_trip(self):
        config = SimConfig(battery_kwh=50, efficiency=4.0, toll_per_mile=0.15,
                           enabled_components={"electricity", "depreciation"})
        json_str = config.to_json()
        restored = SimConfig.from_json(json_str)
        assert restored.battery_kwh == 50
        assert restored.efficiency == 4.0
        assert restored.toll_per_mile == 0.15
        assert restored.enabled_components == {"electricity", "depreciation"}

    def test_forward_compatibility_ignores_unknown_keys(self):
        data = SimConfig().to_dict()
        data["unknown_future_field"] = "something"
        data["another_new_thing"] = 42
        config = SimConfig.from_dict(data)
        assert config.battery_kwh == 75  # still works

    def test_version_field_in_json(self):
        config = SimConfig()
        d = config.to_dict()
        assert "version" in d
        assert d["version"] == 1

    def test_missing_keys_use_defaults(self):
        config = SimConfig.from_dict({"battery_kwh": 50})
        assert config.battery_kwh == 50
        assert config.efficiency == 3.5  # default


# ---------------------------------------------------------------------------
# Tests: validate_csv (NEW)
# ---------------------------------------------------------------------------

class TestValidateCsv:
    def _make_df(self, n=200, **overrides):
        rng = np.random.default_rng(42)
        data = {
            "pickup_latitude": rng.uniform(37.5, 38.0, n),
            "pickup_longitude": rng.uniform(-122.6, -122.2, n),
            "dropoff_latitude": rng.uniform(37.5, 38.0, n),
            "dropoff_longitude": rng.uniform(-122.6, -122.2, n),
            "request_timestamp": pd.date_range("2024-06-01", periods=n, freq="h"),
        }
        data.update(overrides)
        return pd.DataFrame(data)

    def test_happy_path(self):
        df = self._make_df()
        result, errors, warnings = validate_csv(df)
        assert len(errors) == 0
        assert "pickup_latitude" in result.columns

    def test_missing_required_columns(self):
        df = self._make_df().drop(columns=["pickup_latitude", "dropoff_longitude"])
        _, errors, _ = validate_csv(df)
        assert len(errors) == 1
        assert "pickup_latitude" in errors[0]
        assert "dropoff_longitude" in errors[0]

    def test_auto_detect_column_names(self):
        df = self._make_df()
        df = df.rename(columns={
            "pickup_latitude": "pickup_lat",
            "pickup_longitude": "pickup_lon",
            "dropoff_latitude": "dropoff_lat",
            "dropoff_longitude": "dropoff_lon",
        })
        result, errors, _ = validate_csv(df)
        assert len(errors) == 0
        assert "pickup_latitude" in result.columns

    def test_non_numeric_values(self):
        df = self._make_df()
        df.loc[0, "pickup_latitude"] = "not_a_number"
        df.loc[1, "pickup_latitude"] = "bad"
        _, errors, _ = validate_csv(df)
        assert any("non-numeric" in e and "pickup_latitude" in e for e in errors)

    def test_empty_csv(self):
        df = pd.DataFrame(columns=["pickup_latitude", "pickup_longitude",
                                    "dropoff_latitude", "dropoff_longitude"])
        _, errors, _ = validate_csv(df)
        assert any("at least 100" in e for e in errors)

    def test_too_few_rides(self):
        df = self._make_df(n=50)
        _, errors, _ = validate_csv(df)
        assert any("at least 100" in e for e in errors)

    def test_sampling_cap(self):
        df = self._make_df(n=60000)
        result, errors, warnings = validate_csv(df)
        assert len(errors) == 0
        assert len(result) == MAX_RIDES
        assert any("Sampled" in w for w in warnings)

    def test_no_timestamp_warning(self):
        df = self._make_df()
        df = df.drop(columns=["request_timestamp"])
        result, errors, warnings = validate_csv(df)
        assert len(errors) == 0
        assert any("No timestamps" in w for w in warnings)
        assert "request_timestamp" in result.columns

    def test_batch_errors_all_at_once(self):
        df = self._make_df(n=50)  # too few
        df.loc[0, "pickup_latitude"] = "bad"  # non-numeric
        _, errors, _ = validate_csv(df)
        assert len(errors) >= 2  # both errors reported

    def test_coordinate_out_of_range(self):
        df = self._make_df()
        df.loc[0, "pickup_latitude"] = 999.0
        _, errors, _ = validate_csv(df)
        assert any("outside" in e for e in errors)


# ---------------------------------------------------------------------------
# Tests: get_cost_components (NEW)
# ---------------------------------------------------------------------------

class TestGetCostComponents:
    def test_filters_zero_components(self, small_rides, one_depot):
        config = SimConfig(enabled_components={"electricity"})
        sim_df, _ = run_simulation(small_rides, one_depot, config)
        components = get_cost_components(sim_df)
        assert "Electricity" in components
        assert "Vehicle Depreciation" not in components

    def test_all_components_present(self, small_rides, one_depot, default_config):
        sim_df, _ = run_simulation(small_rides, one_depot, default_config)
        components = get_cost_components(sim_df)
        assert "Electricity" in components
        assert "Vehicle Depreciation" in components


# ---------------------------------------------------------------------------
# Tests: render_stacked_bar (NEW)
# ---------------------------------------------------------------------------

class TestRenderStackedBar:
    def test_returns_plotly_figure(self, small_rides, one_depot, default_config):
        sim_df, _ = run_simulation(small_rides, one_depot, default_config)
        fig = render_stacked_bar(sim_df)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
