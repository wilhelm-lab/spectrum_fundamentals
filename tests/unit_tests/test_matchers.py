import unittest

import numpy as np
import pandas as pd

from spectrum_fundamentals.annotation import annotation
from spectrum_fundamentals.annotation.matchers import resolve_matches
from spectrum_fundamentals.annotation.matchers.dp_calibrated import (
    _fit_line_ols,
    dp_calibrated_resolver,
)
from spectrum_fundamentals.annotation.matchers.dp_ladder import (
    _resolve_ppm_scale,
    dp_ladder_resolver,
)
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

    def test_dispatch_dp_ladder(self):
        """The dp_ladder matcher is reachable through the registry."""
        cands = _line_candidates()
        via_registry, _ = resolve_matches("dp_ladder", cands, None, None, "PEPTIDE", ppm_scale=10.0)
        direct, _ = dp_ladder_resolver(cands, ppm_scale=10.0)
        pd.testing.assert_frame_equal(
            via_registry.reset_index(drop=True), direct.reset_index(drop=True)
        )

    def test_dispatch_dp_calibrated(self):
        """The dp_calibrated matcher is reachable through the registry."""
        cands = _line_candidates()
        via_registry, _ = resolve_matches("dp_calibrated", cands, None, None, "PEPTIDE", ppm_scale=10.0)
        direct, _ = dp_calibrated_resolver(cands, ppm_scale=10.0)
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


def _b_ladder(error_ppm: float = 8.0) -> list[dict]:
    """Four singly-charged b ions on a constant ppm drift line (theo 200..500)."""
    return [
        {"ion_type": "b", "no": k + 1, "charge": 1, "theoretical_mass": theo,
         "exp_mass": _ppm(theo, error_ppm), "intensity": inten}
        for k, (theo, inten) in enumerate([(200.0, 0.5), (300.0, 0.7), (400.0, 0.4), (500.0, 0.6)])
    ]


class TestDPLadderResolver(unittest.TestCase):
    """Dynamic-programming matcher over the b/y ion ladder."""

    def test_empty_and_none_candidates(self):
        """None or empty candidates return an empty frame without running the DP."""
        for arg in (None, []):
            df, dropped = dp_ladder_resolver(arg)
            self.assertTrue(df.empty)
            self.assertEqual(dropped, 0)

    def test_clean_ladder_matches_every_slot(self):
        """A clean single-candidate ladder is matched in full, like nearest."""
        cands = _b_ladder(error_ppm=3.0)
        dp_df, dp_dropped = dp_ladder_resolver(cands)
        near_df, _ = nearest_resolver(cands)
        self.assertEqual(len(dp_df), len(near_df))  # nothing dropped
        self.assertEqual(dp_dropped, 0)
        self.assertEqual(
            set(map(tuple, dp_df[["ion_type", "no", "charge"]].values.tolist())),
            set(map(tuple, near_df[["ion_type", "no", "charge"]].values.tolist())),
        )

    def test_ladder_consistency_beats_absolute_closeness(self):
        """On a drifted ladder, the DP keeps the gap-consistent peak where
        nearest would grab a closer-to-theoretical noise peak."""
        cands = _b_ladder(error_ppm=8.0)
        # b3 gets a second candidate that is closer in absolute ppm (+1) but off
        # the local +8 ppm drift line.
        cands.append(
            {"ion_type": "b", "no": 3, "charge": 1, "theoretical_mass": 400.0,
             "exp_mass": _ppm(400.0, 1.0), "intensity": 0.9}
        )
        near_df, _ = nearest_resolver(cands)
        near_b3 = near_df[(near_df["ion_type"] == "b") & (near_df["no"] == 3)]["exp_mass"].iloc[0]
        self.assertAlmostEqual(near_b3, _ppm(400.0, 1.0))  # nearest grabs the off-line peak

        dp_df, _ = dp_ladder_resolver(cands)
        dp_b3 = dp_df[(dp_df["ion_type"] == "b") & (dp_df["no"] == 3)]["exp_mass"].iloc[0]
        self.assertAlmostEqual(dp_b3, _ppm(400.0, 8.0))  # DP keeps the on-line peak

    def test_ladder_inconsistent_peak_is_skipped(self):
        """A lone in-window peak that breaks the ladder gap is dropped as noise."""
        cands = _b_ladder(error_ppm=8.0)
        cands.append(
            {"ion_type": "b", "no": 5, "charge": 1, "theoretical_mass": 600.0,
             "exp_mass": _ppm(600.0, 19.0), "intensity": 0.3}  # in 20 ppm window, off the line
        )
        dp_df, dropped = dp_ladder_resolver(cands)  # defaults: skip_penalty=1, ladder_weight=2
        self.assertEqual(len(dp_df), 4)  # the four on-line slots
        self.assertEqual(dropped, 1)
        self.assertFalse(((dp_df["ion_type"] == "b") & (dp_df["no"] == 5)).any())

    def test_missing_fragment_is_bridged(self):
        """A gap in the ladder (absent b3) doesn't break consistency: the b2->b4
        transition spans the missing residue and the present peaks stay matched."""
        cands = [c for c in _b_ladder(error_ppm=8.0) if c["no"] != 3]
        dp_df, dropped = dp_ladder_resolver(cands)
        self.assertEqual(len(dp_df), 3)  # b1, b2, b4 all kept
        self.assertEqual(dropped, 0)
        self.assertEqual(sorted(dp_df["no"].tolist()), [1, 2, 4])

    def test_monotonicity_prevents_peak_reuse_within_ladder(self):
        """Two adjacent slots whose windows share one peak can't both claim it;
        the ladder's monotonic m/z constraint forces uniqueness."""
        shared = 250.0
        cands = [
            {"ion_type": "b", "no": 1, "charge": 1, "theoretical_mass": 249.999,
             "exp_mass": shared, "intensity": 0.5},
            {"ion_type": "b", "no": 2, "charge": 1, "theoretical_mass": 250.001,
             "exp_mass": shared, "intensity": 0.5},
        ]
        dp_df, _ = dp_ladder_resolver(cands)
        self.assertFalse(dp_df["exp_mass"].duplicated().any())
        self.assertLessEqual(len(dp_df), 1)

    def test_unique_peak_dedup_across_ladders(self):
        """unique_peak resolves a peak claimed by both a b ladder and a y ladder."""
        shared = _ppm(400.0, 2.0)
        cands = [
            {"ion_type": "b", "no": 1, "charge": 1, "theoretical_mass": 200.0,
             "exp_mass": _ppm(200.0, 2.0), "intensity": 0.5},
            {"ion_type": "b", "no": 2, "charge": 1, "theoretical_mass": 400.0,
             "exp_mass": shared, "intensity": 0.5},
            {"ion_type": "y", "no": 1, "charge": 1, "theoretical_mass": 400.0001,
             "exp_mass": shared, "intensity": 0.5},
            {"ion_type": "y", "no": 2, "charge": 1, "theoretical_mass": 600.0,
             "exp_mass": _ppm(600.0, 2.0), "intensity": 0.5},
        ]
        unique_df, _ = dp_ladder_resolver(cands, unique_peak=True)
        reuse_df, _ = dp_ladder_resolver(cands, unique_peak=False)
        self.assertFalse(unique_df["exp_mass"].duplicated().any())
        self.assertEqual((reuse_df["exp_mass"].round(6) == round(shared, 6)).sum(), 2)

    def test_min_match_fraction_triggers_fallback(self):
        """If the DP would drop too many slots, defer to nearest."""
        cands = _b_ladder(error_ppm=8.0)
        # Three of four slots are gross outliers the DP would skip on the ladder.
        for c in cands[1:]:
            c["exp_mass"] = _ppm(c["theoretical_mass"], 19.5)
        floored_df, _ = dp_ladder_resolver(cands, ladder_weight=10.0, min_match_fraction=0.9)
        self.assertEqual(len(floored_df), 4)  # nearest fills every slot
        self.assertNotIn("ppm_residual", floored_df.columns)  # fallback strips diagnostics

    def test_nonfinite_rows_dropped(self):
        """NaN/inf/non-positive masses are dropped and counted."""
        cands = _b_ladder(error_ppm=3.0) + [
            {"ion_type": "b", "no": 6, "charge": 1, "theoretical_mass": np.nan,
             "exp_mass": 600.0, "intensity": 0.1},
            {"ion_type": "y", "no": 7, "charge": 1, "theoretical_mass": -5.0,
             "exp_mass": 700.0, "intensity": 0.1},
        ]
        df, dropped = dp_ladder_resolver(cands)
        self.assertEqual(len(df), 4)
        self.assertEqual(dropped, 2)
        self.assertTrue(np.isfinite(df["theoretical_mass"]).all())

    def test_full_name_preserved(self):
        """The multifrag full_name column is carried through to the output."""
        cands = _b_ladder(error_ppm=3.0)
        for i, c in enumerate(cands):
            c["full_name"] = f"frag_{i}"
        df, _ = dp_ladder_resolver(cands)
        self.assertIn("full_name", df.columns)

    def test_nonfinite_intensity_does_not_poison_cost(self):
        """A NaN/inf intensity must not turn a candidate's cost into NaN (and so
        silently drop a valid on-ladder peak), even at intensity_weight=0."""
        for bad in (np.nan, np.inf):
            with self.subTest(bad=bad):
                cands = _b_ladder(error_ppm=3.0)
                cands[2]["intensity"] = bad  # the b3 peak still sits on the ladder
                for weight in (0.0, 0.5):
                    df, _ = dp_ladder_resolver(cands, intensity_weight=weight)
                    self.assertEqual(len(df), 4)  # b3 is kept, not dropped
                    self.assertTrue(((df["ion_type"] == "b") & (df["no"] == 3)).any())

    def test_ppm_residual_diagnostic_exposed(self):
        """The resolver surfaces a ppm_residual column for downstream features."""
        df, _ = dp_ladder_resolver(_b_ladder(error_ppm=3.0))
        self.assertIn("ppm_residual", df.columns)

    def test_missing_required_column_raises(self):
        """Candidate rows missing a contract column raise ValueError."""
        bad = [{"ion_type": "b", "no": 1, "charge": 1, "exp_mass": 200.0, "intensity": 0.5}]
        with self.assertRaises(ValueError):
            dp_ladder_resolver(bad)

    def test_hyperparameter_validation(self):
        """Out-of-range hyperparameters raise ValueError before running."""
        cands = _b_ladder()
        for kwargs in (
            {"skip_penalty": 0},
            {"skip_penalty": -1.0},
            {"ladder_weight": -0.1},
            {"intensity_weight": -1.0},
            {"min_match_fraction": 1.5},
            {"min_match_fraction": -0.1},
            {"ppm_scale": 0},
            {"ppm_scale": float("inf")},
        ):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    dp_ladder_resolver(cands, **kwargs)


class TestDpCalibratedResolver(unittest.TestCase):
    """RANSAC drift fit + ladder DP combined."""

    def test_empty_and_none_candidates(self):
        for arg in (None, []):
            df, dropped = dp_calibrated_resolver(arg)
            self.assertTrue(df.empty)
            self.assertEqual(dropped, 0)

    def test_drift_is_calibrated_away(self):
        """A constant +12 ppm drift is captured by the fit: every matched peak
        ends up ~on the line (dev_from_line ~ 0) and the ladder is matched in full."""
        df, dropped = dp_calibrated_resolver(
            _b_ladder(error_ppm=12.0), mass_tolerance=20, unit_mass_tolerance="ppm"
        )
        self.assertEqual(len(df), 4)
        self.assertEqual(dropped, 0)
        self.assertIn("dev_from_line", df.columns)
        self.assertIn("ppm_residual", df.columns)
        self.assertTrue(np.allclose(df["dev_from_line"].to_numpy(), 0.0, atol=1e-3))
        # the raw residual still reflects the true +12 drift
        self.assertTrue(np.allclose(df["ppm_residual"].to_numpy(), 12.0, atol=1e-3))

    def test_on_line_peak_chosen_over_near_theoretical_noise(self):
        """Under drift, the de-drifted emission keeps the calibration-consistent
        peak where 'closest to theoretical' would grab a near-zero-ppm noise peak."""
        cands = _b_ladder(error_ppm=12.0)
        cands.append(
            {"ion_type": "b", "no": 3, "charge": 1, "theoretical_mass": 400.0,
             "exp_mass": _ppm(400.0, 1.0), "intensity": 0.9}  # near theoretical, off the drift line
        )
        # nearest grabs the +1 ppm peak; dp_calibrated keeps the on-line +12 peak
        near_df, _ = nearest_resolver(cands)
        near_b3 = near_df[(near_df["ion_type"] == "b") & (near_df["no"] == 3)]["exp_mass"].iloc[0]
        self.assertAlmostEqual(near_b3, _ppm(400.0, 1.0))

        df, _ = dp_calibrated_resolver(cands, mass_tolerance=20, unit_mass_tolerance="ppm")
        cal_b3 = df[(df["ion_type"] == "b") & (df["no"] == 3)]["exp_mass"].iloc[0]
        self.assertAlmostEqual(cal_b3, _ppm(400.0, 12.0))

    def test_falls_back_to_raw_ladder_when_no_line(self):
        """Too few candidates to fit a drift line -> raw ladder DP, still matches."""
        single = [{"ion_type": "b", "no": 1, "charge": 1, "theoretical_mass": 200.0,
                   "exp_mass": _ppm(200.0, 3.0), "intensity": 0.5}]
        df, dropped = dp_calibrated_resolver(single, min_samples=2)
        self.assertEqual(len(df), 1)
        self.assertEqual(dropped, 0)
        # no line -> dev_from_line is just the raw residual
        self.assertTrue(np.allclose(df["dev_from_line"].to_numpy(), df["ppm_residual"].to_numpy()))

    def test_min_inlier_fraction_forces_raw_ladder(self):
        """A fit that explains too few slots is rejected; emission stays raw."""
        cands = [
            {"ion_type": "b", "no": 1, "charge": 1, "theoretical_mass": 200.0,
             "exp_mass": _ppm(200.0, 15.0), "intensity": 0.5},
            {"ion_type": "b", "no": 2, "charge": 1, "theoretical_mass": 300.0,
             "exp_mass": _ppm(300.0, 15.0), "intensity": 0.5},
            {"ion_type": "b", "no": 3, "charge": 1, "theoretical_mass": 400.0,
             "exp_mass": _ppm(400.0, 15.0), "intensity": 0.5},
            {"ion_type": "b", "no": 4, "charge": 1, "theoretical_mass": 500.0,
             "exp_mass": _ppm(500.0, 1.0), "intensity": 0.5},  # off the +15 line
        ]
        # Tight inlier band so no sloped line can absorb the off-line point:
        # the best consensus covers only 3/4 slots, below the 1.0 floor.
        df, _ = dp_calibrated_resolver(
            cands, ppm_scale=20, residual_threshold_ppm=2.0, min_inlier_fraction=1.0
        )
        # line rejected -> dev_from_line == raw ppm_residual on whatever was matched
        self.assertGreaterEqual(len(df), 1)
        self.assertTrue(np.allclose(df["dev_from_line"].to_numpy(), df["ppm_residual"].to_numpy()))

    def test_iterations_are_deterministic_and_converge(self):
        """Extra EM passes don't crash and converge to the single-pass result here."""
        cands = _b_ladder(error_ppm=8.0)
        one, _ = dp_calibrated_resolver(cands, mass_tolerance=20, unit_mass_tolerance="ppm", iterations=1)
        many, _ = dp_calibrated_resolver(cands, mass_tolerance=20, unit_mass_tolerance="ppm", iterations=5)
        self.assertEqual(
            set(map(tuple, one[["ion_type", "no", "charge"]].values.tolist())),
            set(map(tuple, many[["ion_type", "no", "charge"]].values.tolist())),
        )

    def test_unique_peak_dedup_across_ladders(self):
        shared = _ppm(400.0, 2.0)
        cands = [
            {"ion_type": "b", "no": 1, "charge": 1, "theoretical_mass": 200.0,
             "exp_mass": _ppm(200.0, 2.0), "intensity": 0.5},
            {"ion_type": "b", "no": 2, "charge": 1, "theoretical_mass": 400.0,
             "exp_mass": shared, "intensity": 0.5},
            {"ion_type": "y", "no": 1, "charge": 1, "theoretical_mass": 400.0001,
             "exp_mass": shared, "intensity": 0.5},
            {"ion_type": "y", "no": 2, "charge": 1, "theoretical_mass": 600.0,
             "exp_mass": _ppm(600.0, 2.0), "intensity": 0.5},
        ]
        unique_df, _ = dp_calibrated_resolver(cands, mass_tolerance=20, unit_mass_tolerance="ppm", unique_peak=True)
        reuse_df, _ = dp_calibrated_resolver(cands, mass_tolerance=20, unit_mass_tolerance="ppm", unique_peak=False)
        self.assertFalse(unique_df["exp_mass"].duplicated().any())
        self.assertEqual((reuse_df["exp_mass"].round(6) == round(shared, 6)).sum(), 2)

    def test_nonfinite_rows_dropped(self):
        cands = _b_ladder(error_ppm=3.0) + [
            {"ion_type": "b", "no": 6, "charge": 1, "theoretical_mass": np.nan,
             "exp_mass": 600.0, "intensity": 0.1},
            {"ion_type": "y", "no": 7, "charge": 1, "theoretical_mass": -5.0,
             "exp_mass": 700.0, "intensity": 0.1},
        ]
        df, dropped = dp_calibrated_resolver(cands, mass_tolerance=20, unit_mass_tolerance="ppm")
        self.assertEqual(len(df), 4)
        self.assertEqual(dropped, 2)
        self.assertTrue(np.isfinite(df["theoretical_mass"]).all())

    def test_nonfinite_intensity_does_not_poison_cost(self):
        cands = _b_ladder(error_ppm=3.0)
        cands[2]["intensity"] = np.nan
        df, _ = dp_calibrated_resolver(
            cands, mass_tolerance=20, unit_mass_tolerance="ppm", intensity_weight=0.5
        )
        self.assertEqual(len(df), 4)
        self.assertTrue(((df["ion_type"] == "b") & (df["no"] == 3)).any())

    def test_full_name_preserved(self):
        cands = _b_ladder(error_ppm=3.0)
        for i, c in enumerate(cands):
            c["full_name"] = f"frag_{i}"
        df, _ = dp_calibrated_resolver(cands, mass_tolerance=20, unit_mass_tolerance="ppm")
        self.assertIn("full_name", df.columns)

    def test_missing_required_column_raises(self):
        bad = [{"ion_type": "b", "no": 1, "charge": 1, "exp_mass": 200.0, "intensity": 0.5}]
        with self.assertRaises(ValueError):
            dp_calibrated_resolver(bad)

    def test_hyperparameter_validation(self):
        cands = _b_ladder()
        for kwargs in (
            {"skip_penalty": 0},
            {"ladder_weight": -0.1},
            {"min_match_fraction": 1.5},
            {"min_samples": 1},
            {"max_trials": 0},
            {"min_inlier_fraction": 1.5},
            {"iterations": 0},
            {"ppm_scale": 0},
            {"residual_threshold_ppm": -3.0},
        ):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    dp_calibrated_resolver(cands, **kwargs)


class TestFitLineOls(unittest.TestCase):
    """The OLS re-fit helper used by the EM iterations."""

    def test_recovers_line(self):
        theo = np.array([200.0, 300.0, 400.0, 500.0])
        ppm = 5.0 + 0.01 * theo  # a=5, b=0.01
        a, b = _fit_line_ols(theo, ppm)
        self.assertAlmostEqual(a, 5.0, places=6)
        self.assertAlmostEqual(b, 0.01, places=8)

    def test_degenerate_inputs_return_none(self):
        self.assertIsNone(_fit_line_ols(np.array([200.0]), np.array([5.0])))  # one point
        self.assertIsNone(_fit_line_ols(np.array([200.0, 200.0]), np.array([5.0, 6.0])))  # one distinct x


class TestPpmScaleDerivation(unittest.TestCase):
    """Tolerance-aware default for the DP ppm scale."""

    def test_explicit_value_wins(self):
        self.assertEqual(_resolve_ppm_scale(8.0, 20.0, "ppm"), 8.0)

    def test_derived_from_ppm_tolerance(self):
        self.assertEqual(_resolve_ppm_scale(None, 20.0, "ppm"), 20.0)

    def test_da_tolerance_uses_fixed_default(self):
        self.assertEqual(_resolve_ppm_scale(None, 0.02, "da"), 20.0)

    def test_missing_tolerance_uses_fixed_default(self):
        self.assertEqual(_resolve_ppm_scale(None, None, None), 20.0)

    def test_invalid_explicit_value_raises(self):
        with self.assertRaises(ValueError):
            _resolve_ppm_scale(-1.0, None, None)


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
