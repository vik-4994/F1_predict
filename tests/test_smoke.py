from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


class SmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.build_features = _load_build_features_module()
        cls.training_config = _load_training_config_module()

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


if __name__ == "__main__":
    unittest.main()
