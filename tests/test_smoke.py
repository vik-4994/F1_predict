from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.feature_audit import column_health_report, group_health_report
from src.frame_utils import filter_feature_cols, sanitize_frame_columns


def _load_build_features_module():
    spec = importlib.util.spec_from_file_location("build_features_script", ROOT / "scripts" / "build_features.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_training_config_module():
    spec = importlib.util.spec_from_file_location("training_config_module", ROOT / "src" / "training" / "config.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_export_module():
    spec = importlib.util.spec_from_file_location("export_fastf1_script", ROOT / "scripts" / "export_last_two_years.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_predict_module():
    spec = importlib.util.spec_from_file_location("predict_script", ROOT / "scripts" / "predict.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_simulate_season_module():
    spec = importlib.util.spec_from_file_location("simulate_season_script", ROOT / "scripts" / "simulate_season.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class SmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.build_features = _load_build_features_module()
        cls.training_config = _load_training_config_module()
        cls.export_fastf1 = _load_export_module()
        cls.predict_script = _load_predict_module()
        cls.simulate_season = _load_simulate_season_module()

    def test_sanitize_frame_columns_strips_and_coalesces(self) -> None:
        df = pd.DataFrame(
            {
                "Driver": ["VER", "NOR"],
                "score": [1.0, np.nan],
                "score\n": [np.nan, 2.0],
            }
        )
        out = sanitize_frame_columns(df)
        self.assertEqual(out.columns.tolist(), ["Driver", "score"])
        self.assertEqual(out["score"].tolist(), [1.0, 2.0])

    def test_run_modules_coalesces_compatible_overlap(self) -> None:
        def mod_a(_ctx):
            return pd.DataFrame({"Driver": ["VER", "NOR"], "shared": [1.0, np.nan]})

        def mod_b(_ctx):
            return pd.DataFrame({"Driver": ["VER", "NOR"], "shared ": [1.0, 2.0], "extra": [3.0, 4.0]})

        out = self.build_features.run_modules({}, [("a", mod_a), ("b", mod_b)], merge_how="outer")
        self.assertEqual(sorted(out.columns.tolist()), ["Driver", "extra", "shared"])
        self.assertEqual(out.sort_values("Driver")["shared"].tolist(), [2.0, 1.0])

    def test_run_modules_rejects_conflicting_overlap(self) -> None:
        def mod_a(_ctx):
            return pd.DataFrame({"Driver": ["VER"], "shared": [1.0]})

        def mod_b(_ctx):
            return pd.DataFrame({"Driver": ["VER"], "shared": [9.0]})

        with self.assertRaises(RuntimeError):
            self.build_features.run_modules({}, [("a", mod_a), ("b", mod_b)], merge_how="outer", strict_empty=True)

    def test_run_modules_progress_smoke(self) -> None:
        def mod_a(_ctx):
            return pd.DataFrame({"Driver": ["VER", "NOR"], "score": [1.0, 2.0]})

        out = self.build_features.run_modules(
            {},
            [("a", mod_a)],
            merge_how="outer",
            progress=True,
            progress_desc="test build",
        )
        self.assertEqual(out.shape, (2, 2))

    def test_collect_saved_race_frames_recombines_existing_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            pd.DataFrame({"Driver": ["VER"], "x": [1.0]}).to_parquet(out_dir / "features_2025_1.parquet", index=False)
            pd.DataFrame({"Driver": ["NOR"], "x": [2.0]}).to_parquet(out_dir / "features_2025_2.parquet", index=False)

            frames = self.build_features.collect_saved_race_frames(out_dir, "features", skip_tags={"2025_2"})
            self.assertEqual(len(frames), 1)
            self.assertEqual(frames[0]["Driver"].tolist(), ["VER"])
            self.assertEqual(frames[0]["year"].tolist(), [2025])
            self.assertEqual(frames[0]["round"].tolist(), [1])

    def test_sanitize_frame_columns_conflict_raises(self) -> None:
        df = pd.DataFrame({"Driver": ["VER"], "score": [1.0], "score ": [9.0]})
        with self.assertRaises(ValueError):
            sanitize_frame_columns(df)

    def test_filter_feature_cols_supports_ablation_filters(self) -> None:
        cols = [
            "track_is_bahrain_grand_prix",
            "tele_pre_hist_n",
            "double_stack_risk",
            "hist_pre_hist_n",
        ]
        kept, dropped = filter_feature_cols(
            cols,
            drop_prefixes=["tele_pre_"],
            drop_contains=["double_stack"],
            keep_prefixes=[],
        )
        self.assertEqual(kept, ["track_is_bahrain_grand_prix", "hist_pre_hist_n"])
        self.assertEqual(dropped, ["tele_pre_hist_n", "double_stack_risk"])

    def test_track_profile_import_smoke(self) -> None:
        from src.features.track_profile import track_profile

        self.assertTrue(callable(track_profile.featurize))

    def test_default_feature_registry_excludes_experimental_strategy_modules(self) -> None:
        from src import features

        default_names = {name for name, _ in features.FEATURIZERS}
        all_names = {name for name, _ in features.ALL_FEATURIZERS}
        self.assertIn("event_chaos_priors_pre", default_names)
        self.assertIn("practice_longrun_pre", default_names)
        self.assertIn("practice_compound_pre", default_names)
        self.assertIn("weekend_team_delta_pre", default_names)
        self.assertIn("sprint_weekend_pre", default_names)
        self.assertIn("telemetry_efficiency_pre", default_names)
        self.assertIn("quali_execution_pre", default_names)
        self.assertNotIn("strategy_priors_pre", default_names)
        self.assertNotIn("pit_ops_pre", default_names)
        self.assertIn("strategy_priors_pre", all_names)
        self.assertIn("pit_ops_pre", all_names)

    def test_train_config_defaults_match_baseline(self) -> None:
        cfg = self.training_config.TrainConfig()
        self.assertEqual(cfg.hidden, [128, 64])
        self.assertEqual(cfg.epochs, 16)
        self.assertAlmostEqual(cfg.lr, 0.0007)
        self.assertAlmostEqual(cfg.weight_decay, 0.0003)
        self.assertAlmostEqual(cfg.dropout, 0.25)
        self.assertEqual(cfg.drop_prefixes, ["pitcrew_", "slowstop_"])
        self.assertEqual(cfg.drop_contains, ["double_stack", "undercut", "overcut"])
        self.assertEqual(cfg.drop_cols, ["expected_stop_count", "first_stint_len_exp"])

    def test_predict_parse_args_allows_full_grid_in_raw_mode(self) -> None:
        args = self.predict_script.parse_args(
            [
                "--artifacts",
                "models/baseline_v4",
                "--raw-dir",
                "data/raw_csv",
                "--sim-year",
                "2025",
                "--sim-round",
                "1",
            ]
        )
        self.assertIsNone(args.drivers)
        self.assertIsNone(args.track)

    def test_scenario_builder_resolves_track_and_roster(self) -> None:
        from src.scenario_builder import load_roster_drivers, resolve_official_track_name

        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir)
            pd.DataFrame({"Abbreviation": ["NOR", "PIA", "VER"]}).to_csv(raw_dir / "entrylist_2025_1_Q.csv", index=False)
            pd.DataFrame([{"Year": 2025, "Round": 1, "EventName": "Australian Grand Prix"}]).to_csv(
                raw_dir / "meta_2025_1.csv",
                index=False,
            )
            self.assertEqual(load_roster_drivers(raw_dir, 2025, 1), ["NOR", "PIA", "VER"])
            self.assertEqual(resolve_official_track_name(raw_dir, 2025, 1), "Australian Grand Prix")

    def test_export_schedule_round_resolution_uses_completed_sessions(self) -> None:
        sched = pd.DataFrame(
            {
                "RoundNumber": [1, 2, 3],
                "EventDate": ["2025-03-16", "2025-03-23", "2025-04-06"],
                "Session4": ["Qualifying", "Qualifying", "Qualifying"],
                "Session4DateUtc": ["2025-03-15T05:00:00Z", "2025-03-22T05:00:00Z", "2025-04-05T05:00:00Z"],
                "Session5": ["Race", "Race", "Race"],
                "Session5DateUtc": ["2025-03-16T04:00:00Z", "2025-03-23T04:00:00Z", "2025-04-06T05:00:00Z"],
            }
        )
        rounds = self.export_fastf1.resolve_rounds_from_schedule(
            sched,
            ["R", "Q"],
            completed_only=True,
            lookback_rounds=2,
            now_utc=pd.Timestamp("2025-03-24T00:00:00Z"),
        )
        self.assertEqual(rounds, [1, 2])

        latest = self.export_fastf1.resolve_rounds_from_schedule(
            sched,
            ["R", "Q"],
            completed_only=True,
            latest_only=True,
            now_utc=pd.Timestamp("2025-03-24T00:00:00Z"),
        )
        self.assertEqual(latest, [2])

    def test_feature_audit_reports_dead_group(self) -> None:
        df = pd.DataFrame(
            {
                "Driver": ["VER", "NOR", "VER", "NOR"],
                "year": [2025, 2025, 2025, 2025],
                "round": [1, 1, 2, 2],
                "hist_pre_best10_pace_p50_s": [90.1, 91.3, 89.8, 90.7],
                "pitcrew_iqr_team": [np.nan, np.nan, np.nan, np.nan],
            }
        )
        col_report = column_health_report(df)
        grp_report = group_health_report(col_report)
        pit = grp_report.loc[grp_report["group"] == "pit_ops"].iloc[0]
        self.assertTrue(bool(pit["dead_group"]))
        self.assertEqual(int(pit["all_nan_cols"]), 1)

    def test_weather_basic_relative_times_use_start_window_without_warning(self) -> None:
        import importlib

        weather_basic = importlib.import_module("src.features.weather_basic")

        weather = pd.DataFrame(
            {
                "Time": ["0 days 00:01:00", "0 days 00:10:00", "0 days 00:20:00"],
                "AirTemp": [20.0, 22.0, 40.0],
                "Humidity": [60.0, 62.0, 90.0],
                "WindSpeed": [3.0, 5.0, 30.0],
                "Rainfall": [False, True, True],
            }
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("error")
            agg = weather_basic._aggregate_window(weather, None, None, start_window_min=15)
        self.assertEqual(len(caught), 0)
        self.assertEqual(int(agg["weather_pre_records_n"]), 2)
        self.assertAlmostEqual(float(agg["weather_pre_air_temp_mean"]), 21.0)
        self.assertAlmostEqual(float(agg["weather_pre_rain_prob_p75"]), 0.75)

    def test_simulate_season_points_assignment(self) -> None:
        df = pd.DataFrame(
            {
                "Driver": ["NOR", "VER", "LEC", "PIA"],
                "rank": [1, 2, 3, 11],
            }
        )
        out = self.simulate_season.assign_race_points(df)
        self.assertEqual(out["points"].tolist(), [25, 18, 15, 0])
        self.assertEqual(out["win"].tolist(), [1, 0, 0, 0])
        self.assertEqual(out["podium"].tolist(), [1, 1, 1, 0])

    def test_event_chaos_module_extracts_start_and_incident_priors(self) -> None:
        from src.features.event_chaos_priors_pre import featurize

        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir)
            pd.DataFrame(
                {
                    "year": [2024, 2024, 2025],
                    "round": [1, 2, 3],
                }
            ).to_csv(raw_dir / "races.csv", index=False)

            for y, r, name in (
                (2024, 1, "Australian Grand Prix"),
                (2024, 2, "Bahrain Grand Prix"),
                (2025, 3, "Australian Grand Prix"),
            ):
                pd.DataFrame(
                    [{"Year": y, "Round": r, "Session": "R", "EventName": name, "Location": name.split()[0]}]
                ).to_csv(raw_dir / f"meta_{y}_{r}.csv", index=False)

            pd.DataFrame({"Abbreviation": ["VER", "NOR"]}).to_csv(raw_dir / "results_2025_3.csv", index=False)

            pd.DataFrame(
                {
                    "Time": ["0 days 00:01:00", "0 days 00:03:00", "0 days 00:20:00"],
                    "Status": ["2", "1", "4"],
                    "Message": ["Yellow", "AllClear", "SCDeployed"],
                }
            ).to_csv(raw_dir / "track_status_2024_1.csv", index=False)
            pd.DataFrame(
                {
                    "Time": ["2024-03-17 05:00:00Z", "2024-03-17 05:01:00Z", "2024-03-17 06:00:00Z"],
                    "Category": ["Other", "Flag", "SafetyCar"],
                    "Message": ["STARTING PROCEDURE SUSPENDED", "RED FLAG", "SAFETY CAR DEPLOYED"],
                    "Status": [np.nan, np.nan, "DEPLOYED"],
                    "Flag": [np.nan, "RED", np.nan],
                }
            ).to_csv(raw_dir / "race_ctrl_2024_1.csv", index=False)
            pd.DataFrame(
                {
                    "Time": ["0 days 00:00:10", "0 days 00:01:10"],
                    "Rainfall": [True, True],
                    "AirTemp": [18.0, 18.0],
                    "TrackTemp": [21.0, 21.0],
                    "WindSpeed": [2.0, 2.0],
                    "Humidity": [70.0, 70.0],
                }
            ).to_csv(raw_dir / "weather_2024_1.csv", index=False)

            pd.DataFrame(
                {
                    "Time": ["0 days 00:01:00", "0 days 00:02:00"],
                    "Status": ["1", "1"],
                    "Message": ["AllClear", "AllClear"],
                }
            ).to_csv(raw_dir / "track_status_2024_2.csv", index=False)
            pd.DataFrame(
                {
                    "Time": ["2024-03-24 05:00:00Z"],
                    "Category": ["Flag"],
                    "Message": ["GREEN LIGHT - PIT EXIT OPEN"],
                    "Status": [np.nan],
                    "Flag": ["GREEN"],
                }
            ).to_csv(raw_dir / "race_ctrl_2024_2.csv", index=False)
            pd.DataFrame(
                {
                    "Time": ["0 days 00:00:10", "0 days 00:01:10"],
                    "Rainfall": [False, False],
                    "AirTemp": [21.0, 21.0],
                    "TrackTemp": [31.0, 31.0],
                    "WindSpeed": [1.0, 1.0],
                    "Humidity": [40.0, 40.0],
                }
            ).to_csv(raw_dir / "weather_2024_2.csv", index=False)

            out = featurize({"raw_dir": raw_dir, "year": 2025, "round": 3})
            self.assertFalse(out.empty)
            self.assertEqual(out["Driver"].tolist(), ["VER", "NOR"])
            self.assertTrue((out["chaos_pre_hist_n"] == 2.0).all())
            self.assertTrue((out["chaos_pre_same_track_n"] == 1.0).all())
            self.assertTrue((out["chaos_pre_sc_rate"] > 0.6).all())
            self.assertTrue((out["chaos_pre_red_flag_rate"] > 0.6).all())
            self.assertTrue((out["chaos_pre_wet_start_rate"] > 0.6).all())
            self.assertTrue((out["chaos_pre_delayed_start_rate"] > 0.6).all())
            self.assertTrue((out["chaos_pre_index"] > 0.0).all())

    def test_practice_longrun_module_extracts_current_weekend_dry_runs(self) -> None:
        from src.features.practice_longrun_pre import featurize

        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir)
            pd.DataFrame({"Abbreviation": ["VER", "NOR"]}).to_csv(raw_dir / "results_2025_3.csv", index=False)

            fp2 = pd.DataFrame(
                {
                    "Driver": ["VER"] * 6 + ["NOR"] * 5,
                    "LapTime": [
                        "0 days 00:01:35.0",
                        "0 days 00:01:35.4",
                        "0 days 00:01:35.7",
                        "0 days 00:01:36.0",
                        "0 days 00:01:36.2",
                        "0 days 00:01:36.5",
                        "0 days 00:01:35.8",
                        "0 days 00:01:36.0",
                        "0 days 00:01:36.1",
                        "0 days 00:01:36.4",
                        "0 days 00:01:36.6",
                    ],
                    "Stint": [1] * 11,
                    "Compound": ["MEDIUM"] * 11,
                    "TyreLife": [2, 3, 4, 5, 6, 7, 2, 3, 4, 5, 6],
                    "TrackStatus": ["1"] * 11,
                    "PitInTime": [np.nan] * 11,
                    "PitOutTime": [np.nan] * 11,
                }
            )
            fp2.to_csv(raw_dir / "laps_2025_3_FP2.csv", index=False)

            fp3 = pd.DataFrame(
                {
                    "Driver": ["VER"] * 7,
                    "LapTime": [
                        "0 days 00:01:34.7",
                        "0 days 00:01:34.9",
                        "0 days 00:01:35.1",
                        "0 days 00:01:35.4",
                        "0 days 00:01:35.7",
                        "0 days 00:01:35.9",
                        "0 days 00:01:36.1",
                    ],
                    "Stint": [2] * 7,
                    "Compound": ["HARD"] * 7,
                    "TyreLife": [2, 3, 4, 5, 6, 7, 8],
                    "TrackStatus": ["1"] * 7,
                    "PitInTime": [np.nan] * 7,
                    "PitOutTime": [np.nan] * 7,
                }
            )
            fp3.to_csv(raw_dir / "laps_2025_3_FP3.csv", index=False)

            out = featurize({"raw_dir": raw_dir, "year": 2025, "round": 3})
            self.assertFalse(out.empty)
            self.assertEqual(out["Driver"].tolist(), ["VER", "NOR"])
            self.assertTrue((out["prac_pre_dry_runs_n"] >= 1).all())
            ver = out.loc[out["Driver"] == "VER"].iloc[0]
            nor = out.loc[out["Driver"] == "NOR"].iloc[0]
            self.assertEqual(int(ver["prac_pre_longrun_laps"]), 7)
            self.assertEqual(int(ver["prac_pre_longrun_session_ord"]), 3)
            self.assertEqual(int(nor["prac_pre_longrun_laps"]), 5)
            self.assertTrue(pd.notna(ver["prac_pre_shortrun_best3_s"]))
            self.assertTrue(pd.notna(nor["prac_pre_shortrun_best3_s"]))

    def test_practice_compound_module_extracts_compound_specific_form(self) -> None:
        from src.features.practice_compound_pre import featurize

        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir)
            pd.DataFrame({"Abbreviation": ["VER", "NOR"]}).to_csv(raw_dir / "results_2025_3.csv", index=False)

            fp2 = pd.DataFrame(
                {
                    "Driver": ["VER"] * 10 + ["NOR"] * 10,
                    "LapTime": [
                        "0 days 00:01:34.000000",
                        "0 days 00:01:34.100000",
                        "0 days 00:01:34.200000",
                        "0 days 00:01:34.300000",
                        "0 days 00:01:34.400000",
                        "0 days 00:01:35.000000",
                        "0 days 00:01:35.200000",
                        "0 days 00:01:35.400000",
                        "0 days 00:01:35.700000",
                        "0 days 00:01:36.000000",
                        "0 days 00:01:34.500000",
                        "0 days 00:01:34.600000",
                        "0 days 00:01:34.700000",
                        "0 days 00:01:34.900000",
                        "0 days 00:01:35.000000",
                        "0 days 00:01:35.500000",
                        "0 days 00:01:35.700000",
                        "0 days 00:01:35.800000",
                        "0 days 00:01:36.000000",
                        "0 days 00:01:36.100000",
                    ],
                    "Stint": [1] * 5 + [2] * 5 + [1] * 5 + [2] * 5,
                    "Compound": ["SOFT"] * 5 + ["MEDIUM"] * 5 + ["SOFT"] * 5 + ["MEDIUM"] * 5,
                    "TyreLife": [2, 3, 4, 5, 6] * 4,
                    "TrackStatus": ["1"] * 20,
                }
            )
            fp2.to_csv(raw_dir / "laps_2025_3_FP2.csv", index=False)

            fp3 = pd.DataFrame(
                {
                    "Driver": ["VER"] * 5,
                    "LapTime": [
                        "0 days 00:01:36.500000",
                        "0 days 00:01:36.700000",
                        "0 days 00:01:36.900000",
                        "0 days 00:01:37.100000",
                        "0 days 00:01:37.300000",
                    ],
                    "Stint": [3] * 5,
                    "Compound": ["HARD"] * 5,
                    "TyreLife": [2, 3, 4, 5, 6],
                    "TrackStatus": ["1"] * 5,
                }
            )
            fp3.to_csv(raw_dir / "laps_2025_3_FP3.csv", index=False)

            out = featurize({"raw_dir": raw_dir, "year": 2025, "round": 3})
            self.assertFalse(out.empty)
            ver = out.loc[out["Driver"] == "VER"].iloc[0]
            nor = out.loc[out["Driver"] == "NOR"].iloc[0]
            self.assertEqual(float(ver["prac_cmp_pre_dry_compounds_n"]), 3.0)
            self.assertEqual(float(nor["prac_cmp_pre_dry_compounds_n"]), 2.0)
            self.assertAlmostEqual(float(ver["prac_cmp_pre_s_best_s"]), 94.0, places=6)
            self.assertAlmostEqual(float(ver["prac_cmp_pre_m_longrun_pace_s"]), 95.4, places=6)
            self.assertAlmostEqual(float(ver["prac_cmp_pre_crossover_sm_s"]), 1.0, places=6)
            self.assertTrue(pd.isna(nor["prac_cmp_pre_h_best_s"]))

    def test_sprint_weekend_module_extracts_current_sprint_signals(self) -> None:
        from src.features.sprint_weekend_pre import featurize

        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir)
            pd.DataFrame(
                {
                    "Abbreviation": ["PIA", "NOR", "VER", "PER"],
                    "TeamName": ["McLaren", "McLaren", "Red Bull Racing", "Red Bull Racing"],
                }
            ).to_csv(raw_dir / "entrylist_2025_2_Q.csv", index=False)
            pd.DataFrame(
                [{"Year": 2025, "Round": 2, "EventName": "Chinese Grand Prix", "EventFormat": "sprint_qualifying"}]
            ).to_csv(raw_dir / "meta_2025_2.csv", index=False)
            pd.DataFrame(
                {
                    "Abbreviation": ["PIA", "NOR", "VER", "PER"],
                    "TeamName": ["McLaren", "McLaren", "Red Bull Racing", "Red Bull Racing"],
                    "Position": [1, 2, 3, 8],
                    "Q1": ["0 days 00:01:31.000000"] * 4,
                    "Q2": ["0 days 00:01:30.500000", "0 days 00:01:30.600000", "0 days 00:01:30.900000", "0 days 00:01:31.300000"],
                    "Q3": ["0 days 00:01:30.000000", "0 days 00:01:30.200000", "0 days 00:01:30.500000", "0 days 00:01:31.000000"],
                }
            ).to_csv(raw_dir / "results_2025_2_SQ.csv", index=False)
            pd.DataFrame(
                {
                    "Abbreviation": ["NOR", "PIA", "VER", "PER"],
                    "TeamName": ["McLaren", "McLaren", "Red Bull Racing", "Red Bull Racing"],
                    "Position": [1, 2, 3, 6],
                    "GridPosition": [2, 1, 4, 8],
                    "Points": [8, 7, 6, 3],
                }
            ).to_csv(raw_dir / "results_2025_2_S.csv", index=False)

            out = featurize({"raw_dir": raw_dir, "year": 2025, "round": 2})
            self.assertFalse(out.empty)
            pia = out.loc[out["Driver"] == "PIA"].iloc[0]
            nor = out.loc[out["Driver"] == "NOR"].iloc[0]
            self.assertEqual(float(pia["sprint_pre_format_flag"]), 1.0)
            self.assertEqual(float(pia["sprint_pre_has_data"]), 1.0)
            self.assertAlmostEqual(float(pia["sprint_pre_sq_gap_to_best_s"]), 0.0, places=6)
            self.assertAlmostEqual(float(nor["sprint_pre_gain_vs_grid"]), 1.0, places=6)
            self.assertAlmostEqual(float(nor["sprint_pre_tm_finish_delta"]), -1.0, places=6)

    def test_quali_execution_module_extracts_execution_signals(self) -> None:
        from src.features.quali_execution_pre import featurize

        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir)
            pd.DataFrame(
                {
                    "Abbreviation": ["VER", "PER"],
                    "TeamName": ["Red Bull Racing", "Red Bull Racing"],
                    "Position": [1, 4],
                    "Q1": ["0 days 00:01:16.000000", "0 days 00:01:16.900000"],
                    "Q2": ["0 days 00:01:15.800000", "0 days 00:01:16.500000"],
                    "Q3": ["0 days 00:01:15.500000", "0 days 00:01:16.200000"],
                }
            ).to_csv(raw_dir / "results_2025_3_Q.csv", index=False)
            pd.DataFrame(
                {
                    "Driver": ["VER", "VER", "VER", "PER", "PER"],
                    "LapTime": [
                        "0 days 00:01:16.000000",
                        "0 days 00:01:15.700000",
                        "0 days 00:01:15.500000",
                        "0 days 00:01:16.400000",
                        "0 days 00:01:16.200000",
                    ],
                    "Sector1Time": [
                        "0 days 00:00:25.100000",
                        "0 days 00:00:25.000000",
                        "0 days 00:00:24.900000",
                        "0 days 00:00:25.500000",
                        "0 days 00:00:25.300000",
                    ],
                    "Sector2Time": [
                        "0 days 00:00:17.300000",
                        "0 days 00:00:17.200000",
                        "0 days 00:00:17.100000",
                        "0 days 00:00:17.700000",
                        "0 days 00:00:17.500000",
                    ],
                    "Sector3Time": [
                        "0 days 00:00:33.600000",
                        "0 days 00:00:33.500000",
                        "0 days 00:00:33.500000",
                        "0 days 00:00:33.200000",
                        "0 days 00:00:33.400000",
                    ],
                    "TrackStatus": ["1"] * 5,
                    "IsAccurate": [True, True, True, True, True],
                    "Deleted": [False, True, False, False, False],
                    "Stint": [1, 1, 2, 1, 2],
                }
            ).to_csv(raw_dir / "laps_2025_3_Q.csv", index=False)

            out = featurize({"raw_dir": raw_dir, "year": 2025, "round": 3})
            self.assertFalse(out.empty)
            ver = out.loc[out["Driver"] == "VER"].iloc[0]
            per = out.loc[out["Driver"] == "PER"].iloc[0]
            self.assertAlmostEqual(float(ver["qexec_pre_gap_to_pole_s"]), 0.0, places=6)
            self.assertAlmostEqual(float(ver["qexec_pre_final_run_improve_s"]), 0.5, places=6)
            self.assertAlmostEqual(float(ver["qexec_pre_deleted_lap_share"]), 1 / 3, places=6)
            self.assertAlmostEqual(float(ver["qexec_pre_gap_to_tm_s"]), -0.7, places=6)
            self.assertEqual(float(per["qexec_pre_stage_reached"]), 3.0)
            self.assertTrue(float(ver["qexec_pre_ideal_lap_gap_s"]) >= 0.0)

    def test_telemetry_efficiency_module_extracts_weekend_metrics(self) -> None:
        from src.features.telemetry_efficiency_pre import featurize

        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir)
            pd.DataFrame(
                {
                    "Abbreviation": ["VER", "PER", "NOR", "PIA"],
                    "TeamName": ["Red Bull Racing", "Red Bull Racing", "McLaren", "McLaren"],
                }
            ).to_csv(raw_dir / "results_2025_3_Q.csv", index=False)

            def save_tel(name: str, speeds: list[float], throttle: list[float], drs: list[int], brake: list[bool]) -> None:
                pd.DataFrame(
                    {
                        "Speed": speeds,
                        "Throttle": throttle,
                        "DRS": drs,
                        "Brake": brake,
                    }
                ).to_csv(raw_dir / name, index=False)

            save_tel(
                "telemetry_2025_3_Q_VER.csv",
                [180.0, 220.0, 300.0, 312.0, 90.0],
                [100.0, 100.0, 100.0, 100.0, 20.0],
                [8, 10, 14, 14, 8],
                [False, False, False, False, True],
            )
            save_tel(
                "telemetry_2025_3_Q_PER.csv",
                [170.0, 210.0, 290.0, 300.0, 85.0],
                [100.0, 100.0, 100.0, 100.0, 15.0],
                [8, 10, 14, 14, 8],
                [False, False, False, False, True],
            )
            save_tel(
                "telemetry_2025_3_FP3_NOR.csv",
                [185.0, 230.0, 315.0, 330.0, 95.0],
                [100.0, 100.0, 100.0, 100.0, 25.0],
                [8, 10, 14, 14, 8],
                [False, False, False, False, True],
            )
            save_tel(
                "telemetry_2025_3_FP3_PIA.csv",
                [175.0, 220.0, 305.0, 320.0, 92.0],
                [100.0, 100.0, 100.0, 100.0, 25.0],
                [8, 10, 14, 14, 8],
                [False, False, False, False, True],
            )

            out = featurize({"raw_dir": raw_dir, "year": 2025, "round": 3})
            self.assertFalse(out.empty)
            ver = out.loc[out["Driver"] == "VER"].iloc[0]
            nor = out.loc[out["Driver"] == "NOR"].iloc[0]
            self.assertEqual(float(ver["tele_eff_pre_session_ord"]), 4.0)
            self.assertEqual(float(nor["tele_eff_pre_session_ord"]), 3.0)
            self.assertTrue(float(ver["tele_eff_pre_speed_p95_kph"]) > float(out.loc[out["Driver"] == "PER", "tele_eff_pre_speed_p95_kph"].iloc[0]))
            self.assertTrue(float(ver["tele_eff_pre_speed_tm_delta_kph"]) > 0.0)
            self.assertTrue(float(nor["tele_eff_pre_drs_speed_gain_kph"]) > 0.0)

    def test_weekend_team_delta_module_extracts_tm_deltas(self) -> None:
        from src.features.weekend_team_delta_pre import featurize

        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir)
            pd.DataFrame(
                {
                    "Abbreviation": ["VER", "PER", "NOR", "PIA"],
                    "TeamName": ["Red Bull Racing", "Red Bull Racing", "McLaren", "McLaren"],
                    "Position": [1, 4, 2, 3],
                    "Q1": ["0 days 00:01:16.000000", "0 days 00:01:16.400000", "0 days 00:01:15.900000", "0 days 00:01:16.000000"],
                    "Q2": ["0 days 00:01:15.600000", "0 days 00:01:16.100000", "0 days 00:01:15.500000", "0 days 00:01:15.700000"],
                    "Q3": ["0 days 00:01:15.300000", "0 days 00:01:15.900000", "0 days 00:01:15.100000", "0 days 00:01:15.300000"],
                }
            ).to_csv(raw_dir / "results_2025_3_Q.csv", index=False)

            pd.DataFrame(
                {
                    "Driver": ["VER", "VER", "PER", "PER", "NOR", "NOR", "PIA", "PIA"],
                    "LapTime": [
                        "0 days 00:01:15.300000",
                        "0 days 00:01:15.400000",
                        "0 days 00:01:15.900000",
                        "0 days 00:01:16.000000",
                        "0 days 00:01:15.100000",
                        "0 days 00:01:15.200000",
                        "0 days 00:01:15.300000",
                        "0 days 00:01:15.400000",
                    ],
                    "Sector1Time": [
                        "0 days 00:00:25.000000",
                        "0 days 00:00:25.100000",
                        "0 days 00:00:25.200000",
                        "0 days 00:00:25.300000",
                        "0 days 00:00:24.900000",
                        "0 days 00:00:25.000000",
                        "0 days 00:00:25.000000",
                        "0 days 00:00:25.100000",
                    ],
                    "Sector2Time": [
                        "0 days 00:00:17.100000",
                        "0 days 00:00:17.200000",
                        "0 days 00:00:17.400000",
                        "0 days 00:00:17.500000",
                        "0 days 00:00:17.000000",
                        "0 days 00:00:17.100000",
                        "0 days 00:00:17.200000",
                        "0 days 00:00:17.200000",
                    ],
                    "Sector3Time": [
                        "0 days 00:00:33.200000",
                        "0 days 00:00:33.300000",
                        "0 days 00:00:33.400000",
                        "0 days 00:00:33.500000",
                        "0 days 00:00:33.100000",
                        "0 days 00:00:33.200000",
                        "0 days 00:00:33.100000",
                        "0 days 00:00:33.200000",
                    ],
                    "TrackStatus": ["1"] * 8,
                    "IsAccurate": [True] * 8,
                }
            ).to_csv(raw_dir / "laps_2025_3_Q.csv", index=False)

            pd.DataFrame(
                {
                    "Driver": ["VER"] * 5 + ["PER"] * 5 + ["NOR"] * 5 + ["PIA"] * 5,
                    "LapTime": [
                        "0 days 00:01:34.8",
                        "0 days 00:01:35.0",
                        "0 days 00:01:35.2",
                        "0 days 00:01:35.4",
                        "0 days 00:01:35.6",
                        "0 days 00:01:35.3",
                        "0 days 00:01:35.5",
                        "0 days 00:01:35.8",
                        "0 days 00:01:36.0",
                        "0 days 00:01:36.3",
                        "0 days 00:01:34.7",
                        "0 days 00:01:34.9",
                        "0 days 00:01:35.0",
                        "0 days 00:01:35.2",
                        "0 days 00:01:35.3",
                        "0 days 00:01:34.9",
                        "0 days 00:01:35.0",
                        "0 days 00:01:35.2",
                        "0 days 00:01:35.3",
                        "0 days 00:01:35.4",
                    ],
                    "Stint": [1] * 20,
                    "Compound": ["MEDIUM"] * 20,
                    "TyreLife": [2, 3, 4, 5, 6] * 4,
                    "TrackStatus": ["1"] * 20,
                }
            ).to_csv(raw_dir / "laps_2025_3_FP2.csv", index=False)

            out = featurize({"raw_dir": raw_dir, "year": 2025, "round": 3})
            self.assertFalse(out.empty)
            self.assertTrue((out["wknd_pre_tm_has_pair"] == 1.0).all())
            ver = out.loc[out["Driver"] == "VER"].iloc[0]
            per = out.loc[out["Driver"] == "PER"].iloc[0]
            self.assertAlmostEqual(float(ver["wknd_pre_q_tm_delta_s"]), -0.6, places=6)
            self.assertAlmostEqual(float(per["wknd_pre_q_tm_delta_s"]), 0.6, places=6)
            self.assertTrue(pd.notna(ver["wknd_pre_q_sector_tm_delta_std_s"]))
            self.assertTrue(pd.notna(ver["wknd_pre_prac_longrun_tm_delta_s"]))

    def test_driver_track_cluster_module_has_signal_on_repo_data(self) -> None:
        from src.features.driver_track_cluster_pre import featurize

        raw_dir = ROOT / "data" / "raw_csv"
        if not raw_dir.exists():
            self.skipTest("raw_csv data not available")
        df = featurize({"raw_dir": raw_dir, "year": 2025, "round": 1})
        self.assertFalse(df.empty)
        self.assertGreater(int(df["driver_trackc_pre_hist_n"].notna().sum()), 0)

    def test_tyre_module_has_non_empty_mix_on_repo_data(self) -> None:
        from src.features.tyre_priors_pre import featurize

        raw_dir = ROOT / "data" / "raw_csv"
        if not raw_dir.exists():
            self.skipTest("raw_csv data not available")
        df = featurize({"raw_dir": raw_dir, "year": 2025, "round": 1})
        self.assertFalse(df.empty)
        self.assertIn("compound_mix_priors_S", df.columns)
        self.assertIn("tyre_delta_priors_s_SM", df.columns)

    def test_event_chaos_module_has_signal_on_repo_data(self) -> None:
        from src.features.event_chaos_priors_pre import featurize

        raw_dir = ROOT / "data" / "raw_csv"
        if not raw_dir.exists():
            self.skipTest("raw_csv data not available")
        df = featurize({"raw_dir": raw_dir, "year": 2025, "round": 1})
        self.assertFalse(df.empty)
        self.assertGreater(int(df["chaos_pre_hist_n"].notna().sum()), 0)
        non_na = df[
            [
                "chaos_pre_sc_rate",
                "chaos_pre_vsc_rate",
                "chaos_pre_red_flag_rate",
                "chaos_pre_index",
            ]
        ].notna().sum().sum()
        self.assertGreater(int(non_na), 0)

    def test_weekend_team_delta_module_has_signal_on_repo_data(self) -> None:
        from src.features.weekend_team_delta_pre import featurize

        raw_dir = ROOT / "data" / "raw_csv"
        if not raw_dir.exists():
            self.skipTest("raw_csv data not available")
        df = featurize({"raw_dir": raw_dir, "year": 2025, "round": 1})
        self.assertFalse(df.empty)
        self.assertGreater(int(df["wknd_pre_tm_has_pair"].notna().sum()), 0)
        non_na = df[
            [
                "wknd_pre_q_tm_delta_s",
                "wknd_pre_q_s1_tm_delta_s",
                "wknd_pre_q_sector_tm_delta_std_s",
            ]
        ].notna().sum().sum()
        self.assertGreater(int(non_na), 0)

    def test_traffic_module_uses_driver_codes_on_repo_data(self) -> None:
        from src.features.traffic_overtake_pre import featurize

        raw_dir = ROOT / "data" / "raw_csv"
        if not raw_dir.exists():
            self.skipTest("raw_csv data not available")
        df = featurize({"raw_dir": raw_dir, "year": 2025, "round": 1})
        self.assertFalse(df.empty)
        self.assertTrue(df["Driver"].astype(str).str.fullmatch(r"[A-Z]{3}").all())
        self.assertTrue({"lap1_gain_avg_prev10", "lap1_incident_rate_prev10", "net_pass_index_prev10"}.issubset(df.columns))


if __name__ == "__main__":
    unittest.main()
