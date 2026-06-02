import unittest

import numpy as np
import pandas as pd

from spectrum_fundamentals.annotation import annotation
from spectrum_fundamentals.annotation.matchers import resolve_matches
from spectrum_fundamentals.annotation.matchers.global_ransac import (
    _resolve_residual_threshold,
    global_ransac_resolver,
)
from spectrum_fundamentals.annotation.matchers.nearest import nearest_resolver


def _ppm(theoretical_mass: float, error_ppm: float) -> float:
    """Return an experimental mass that is ``error_ppm`` away from theoretical."""
    return theoretical_mass * (1 + error_ppm * 1e-6)


def _line_candidates(error_ppm: float = 3.0) -> list[dict]:
    """Four fragment slots whose true peaks all sit on a constant ppm drift line."""
    return [
        {"ion_type": "b", "no": 1, "charge": 1, "theoretical_mass": 200.0,
         "exp_mass": _ppm(200.0, error_ppm), "intensity": 0.5},
        {"ion_type": "y", "no": 1, "charge": 1, "theoretical_mass": 400.0,
         "exp_mass": _ppm(400.0, error_ppm), "intensity": 0.7},
        {"ion_type": "b", "no": 2, "charge": 1, "theoretical_mass": 600.0,
         "exp_mass": _ppm(600.0, error_ppm), "intensity": 0.4},
        {"ion_type": "y", "no": 2, "charge": 1, "theoretical_mass": 800.0,
         "exp_mass": _ppm(800.0, error_ppm), "intensity": 0.6},
    ]


class TestNearestResolver(unittest.TestCase):
    """The legacy closest-m/z resolver."""

    def test_equivalent_to_handle_multiple_matches(self):
        """nearest_resolver must reproduce handle_multiple_matches(sort_by='mass_diff') byte-for-byte."""
        matched_peaks = [
            {"ion_type": "b", "no": 2, "charge": 1, "exp_mass": 200, "theoretical_mass": 198, "intensity": 0.05},
            {"ion_type": "b", "no": 2, "charge": 1, "exp_mass": 205, "theoretical_mass": 198, "intensity": 0.01},
            {"ion_type": "y", "no": 3, "charge": 1, "exp_mass": 300, "theoretical_mass": 303, "intensity": 0.1},
            {"ion_type": "y", "no": 3, "charge": 1, "exp_mass": 303, "theoretical_mass": 303, "intensity": 0.05},
        ]
        legacy_df, legacy_dropped = annotation.handle_multiple_matches(matched_peaks, sort_by="mass_diff")
        resolver_df, resolver_dropped = nearest_resolver(matched_peaks)

        pd.testing.assert_frame_equal(legacy_df, resolver_df)
        self.assertEqual(legacy_dropped, resolver_dropped)

    def test_empty_input(self):
        """Empty candidate list returns an empty frame and zero dropped."""
        df, dropped = nearest_resolver([])
        self.assertTrue(df.empty)
        self.assertEqual(dropped, 0)


class TestResolveMatchesDispatch(unittest.TestCase):
    """The registry dispatch function."""

    def test_unknown_method_raises(self):
        """An unregistered method name raises a helpful ValueError."""
        with self.assertRaises(ValueError):
            resolve_matches("does_not_exist", _line_candidates(), None, None, "PEPTIDE")

    def test_dispatch_matches_direct_call(self):
        """Dispatching forwards kwargs identically to calling the resolver directly."""
        cands = _line_candidates()
        via_registry, _ = resolve_matches(
            "global_ransac", cands, None, None, "PEPTIDE", residual_threshold_ppm=5.0, random_state=0
        )
        direct, _ = global_ransac_resolver(cands, residual_threshold_ppm=5.0, random_state=0)
        pd.testing.assert_frame_equal(
            via_registry.reset_index(drop=True), direct.reset_index(drop=True)
        )


class TestGlobalRansacResolver(unittest.TestCase):
    """Per-spectrum RANSAC mass-calibration matcher."""

    def test_empty_and_none_candidates(self):
        """None or empty candidates return an empty frame without fitting."""
        for arg in (None, []):
            df, dropped = global_ransac_resolver(arg)
            self.assertTrue(df.empty)
            self.assertEqual(dropped, 0)

    def test_drift_recovery_and_outlier_rejection(self):
        """The fit keeps the on-line matches and rejects peaks off the drift line."""
        cands = _line_candidates(error_ppm=3.0)
        noise = [
            {"ion_type": "b", "no": 1, "charge": 1, "theoretical_mass": 200.0,
             "exp_mass": _ppm(200.0, -18.0), "intensity": 0.9},  # same slot as a real match, far off
            {"ion_type": "y", "no": 2, "charge": 1, "theoretical_mass": 800.0,
             "exp_mass": _ppm(800.0, 19.0), "intensity": 0.2},   # same slot, far off
        ]
        df, dropped = global_ransac_resolver(cands + noise, residual_threshold_ppm=5.0)

        self.assertEqual(len(df), 4)  # one per slot
        self.assertEqual(dropped, 2)  # both noise rows rejected
        # the rejected (off-line) experimental masses must not appear
        self.assertNotIn(round(_ppm(200.0, -18.0), 6), df["exp_mass"].round(6).tolist())
        self.assertNotIn(round(_ppm(800.0, 19.0), 6), df["exp_mass"].round(6).tolist())
        # diagnostic columns are exposed
        self.assertIn("ppm_residual", df.columns)
        self.assertIn("abs_dev_from_fit", df.columns)

    def test_unique_peak_toggle(self):
        """unique_peak controls whether one observed peak may fill several slots."""
        shared = _ppm(800.0, 2.0)
        cands = [
            {"ion_type": "b", "no": 1, "charge": 1, "theoretical_mass": 200.0,
             "exp_mass": _ppm(200.0, 2.0), "intensity": 0.5},
            {"ion_type": "y", "no": 1, "charge": 1, "theoretical_mass": 400.0,
             "exp_mass": _ppm(400.0, 2.0), "intensity": 0.5},
            {"ion_type": "b", "no": 2, "charge": 1, "theoretical_mass": 600.0,
             "exp_mass": _ppm(600.0, 2.0), "intensity": 0.5},
            # two distinct slots that matched the same observed peak
            {"ion_type": "b", "no": 4, "charge": 1, "theoretical_mass": 800.0,
             "exp_mass": shared, "intensity": 0.5},
            {"ion_type": "y", "no": 7, "charge": 2, "theoretical_mass": 800.0008,
             "exp_mass": shared, "intensity": 0.5},
        ]
        unique_df, _ = global_ransac_resolver(cands, residual_threshold_ppm=5.0, unique_peak=True)
        reuse_df, _ = global_ransac_resolver(cands, residual_threshold_ppm=5.0, unique_peak=False)

        # with uniqueness the shared peak is claimed by exactly one slot
        self.assertEqual(len(unique_df), 4)
        self.assertFalse(unique_df["exp_mass"].duplicated().any())
        # without it, both slots keep the same peak
        self.assertEqual(len(reuse_df), 5)
        self.assertTrue(reuse_df["exp_mass"].duplicated().any())

    def test_fallback_below_min_samples(self):
        """Fewer valid candidates than min_samples defers to nearest."""
        single = [{"ion_type": "b", "no": 1, "charge": 1, "theoretical_mass": 200.0,
                   "exp_mass": _ppm(200.0, 3.0), "intensity": 0.5}]
        df, dropped = global_ransac_resolver(single, min_samples=2)
        self.assertEqual(len(df), 1)
        self.assertEqual(dropped, 0)

    def test_nonfinite_rows_dropped(self):
        """NaN/inf/non-positive masses are dropped and counted in n_dropped."""
        cands = _line_candidates(error_ppm=3.0) + [
            {"ion_type": "b", "no": 3, "charge": 1, "theoretical_mass": np.nan,
             "exp_mass": 300.0, "intensity": 0.1},
            {"ion_type": "b", "no": 4, "charge": 1, "theoretical_mass": np.inf,
             "exp_mass": 400.0, "intensity": 0.1},
            {"ion_type": "y", "no": 5, "charge": 1, "theoretical_mass": -10.0,
             "exp_mass": 500.0, "intensity": 0.1},
        ]
        df, dropped = global_ransac_resolver(cands, residual_threshold_ppm=5.0)
        self.assertEqual(len(df), 4)  # only the four valid on-line slots survive
        self.assertEqual(dropped, 3)
        self.assertTrue(np.isfinite(df["theoretical_mass"]).all())

    def test_inlier_floor_triggers_fallback(self):
        """A converged fit that explains too few slots defers to nearest."""
        cands = _line_candidates(error_ppm=2.0)
        # replace the last slot's peak with a gross outlier so it can't be an inlier
        cands[-1]["exp_mass"] = _ppm(800.0, 50.0)

        # 3/4 slots are inliers -> 0.75; below a 0.9 floor we fall back to nearest,
        # which fills every slot (including the outlier one).
        strict_df, _ = global_ransac_resolver(cands, residual_threshold_ppm=5.0, min_inlier_fraction=0.9)
        self.assertEqual(len(strict_df), 4)
        self.assertTrue(((strict_df["ion_type"] == "y") & (strict_df["no"] == 2)).any())

        # 0.75 >= 0.5 floor -> keep the RANSAC result, the outlier slot is dropped.
        lenient_df, _ = global_ransac_resolver(cands, residual_threshold_ppm=5.0, min_inlier_fraction=0.5)
        self.assertEqual(len(lenient_df), 3)
        self.assertFalse(((lenient_df["ion_type"] == "y") & (lenient_df["no"] == 2)).any())

    def test_full_name_preserved(self):
        """The multifrag full_name column is carried through to the output."""
        cands = _line_candidates(error_ppm=3.0)
        for i, c in enumerate(cands):
            c["full_name"] = f"frag_{i}"
        df, _ = global_ransac_resolver(cands, residual_threshold_ppm=5.0)
        self.assertIn("full_name", df.columns)

    def test_missing_required_column_raises(self):
        """Candidate rows missing a contract column raise ValueError."""
        bad = [{"ion_type": "b", "no": 1, "charge": 1, "exp_mass": 200.0, "intensity": 0.5}]  # no theoretical_mass
        with self.assertRaises(ValueError):
            global_ransac_resolver(bad)

    def test_hyperparameter_validation(self):
        """Out-of-range hyperparameters raise ValueError before fitting."""
        cands = _line_candidates()
        for kwargs in (
            {"min_samples": 1},
            {"max_trials": 0},
            {"min_inlier_fraction": 1.5},
            {"min_inlier_fraction": -0.1},
            {"residual_threshold_ppm": 0},
            {"residual_threshold_ppm": -3.0},
            {"residual_threshold_ppm": float("inf")},
        ):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    global_ransac_resolver(cands, **kwargs)


class TestResidualThresholdDerivation(unittest.TestCase):
    """Tolerance-aware default for the RANSAC inlier band."""

    def test_explicit_value_wins(self):
        self.assertEqual(_resolve_residual_threshold(8.0, 20.0, "ppm"), 8.0)

    def test_derived_from_ppm_tolerance(self):
        self.assertEqual(_resolve_residual_threshold(None, 20.0, "ppm"), 10.0)

    def test_da_tolerance_uses_fixed_default(self):
        self.assertEqual(_resolve_residual_threshold(None, 20.0, "da"), 5.0)

    def test_missing_tolerance_uses_fixed_default(self):
        self.assertEqual(_resolve_residual_threshold(None, None, None), 5.0)

    def test_invalid_explicit_value_raises(self):
        with self.assertRaises(ValueError):
            _resolve_residual_threshold(-1.0, None, None)


if __name__ == "__main__":
    unittest.main()
