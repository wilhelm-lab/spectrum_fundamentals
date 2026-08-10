import unittest
from ast import literal_eval
from pathlib import Path

import numpy as np
import pandas as pd

from spectrum_fundamentals import constants
from spectrum_fundamentals.annotation import annotation


class TestAnnotationPipeline(unittest.TestCase):
    """TestClass for everything in annotation."""

    def test_annotate_spectra(self):
        """Test annotate spectra."""
        spectrum_input = pd.read_csv(
            Path(__file__).parent / "data/spectrum_input.csv",
            index_col=0,
            converters={"INTENSITIES": literal_eval, "MZ": literal_eval},
        )

        expected_result = pd.read_csv(
            Path(__file__).parent / "data/spectrum_output.csv",
            index_col=0,
            converters={"INTENSITIES": literal_eval, "MZ": literal_eval},
        )
        spectrum_input["INTENSITIES"] = spectrum_input["INTENSITIES"].map(lambda intensities: np.array(intensities))
        spectrum_input["MZ"] = spectrum_input["MZ"].map(lambda mz: np.array(mz))

        result = annotation.annotate_spectra(spectrum_input)
        # Only assert columns present in legacy expected output.
        # New per-PSM metrics (sc_features etc.) are tested separately.
        pd.testing.assert_frame_equal(expected_result, result[expected_result.columns])

    def test_annotate_spectra_multifrag(self):
        """Test annotate spectra."""
        spectrum_input = pd.read_csv(
            Path(__file__).parent / "data/spectrum_input_multifrag.csv",
            index_col=0,
            converters={"INTENSITIES": literal_eval, "MZ": literal_eval},
        )

        expected_result = pd.read_csv(
            Path(__file__).parent / "data/spectrum_output_multifrag.csv",
            index_col=0,
            converters={"INTENSITIES": literal_eval, "MZ": literal_eval},
        )
        spectrum_input["INTENSITIES"] = spectrum_input["INTENSITIES"].map(lambda intensities: np.array(intensities))
        spectrum_input["MZ"] = spectrum_input["MZ"].map(lambda mz: np.array(mz))

        result = annotation.annotate_spectra(spectrum_input, multifrag=True, fragmentation_method="ECD")
        # Only assert columns present in legacy expected output.
        # New per-PSM metrics (sc_features etc.) are tested separately.
        pd.testing.assert_frame_equal(expected_result, result[expected_result.columns])

    def test_annotate_spectra_with_custom_mods(self):
        """Test annotate spectra."""
        spectrum_input = pd.read_csv(
            Path(__file__).parent / "data/spectrum_input.csv",
            index_col=0,
            converters={"INTENSITIES": literal_eval, "MZ": literal_eval},
        )

        expected_result = pd.read_csv(
            Path(__file__).parent / "data/spectrum_output.csv",
            index_col=0,
            converters={"INTENSITIES": literal_eval, "MZ": literal_eval},
        )
        spectrum_input["INTENSITIES"] = spectrum_input["INTENSITIES"].map(lambda intensities: np.array(intensities))
        spectrum_input["MZ"] = spectrum_input["MZ"].map(lambda mz: np.array(mz))
        custom_mods = {"[UNIMOD:4]": 57.0215, "[UNIMOD:35]": 15.99}

        result = annotation.annotate_spectra(un_annot_spectra=spectrum_input, custom_mods=custom_mods)
        # Only assert columns present in legacy expected output.
        # New per-PSM metrics (sc_features etc.) are tested separately.
        pd.testing.assert_frame_equal(expected_result, result[expected_result.columns])

    def test_annotate_spectra_noncl_xl(self):
        """Test annotate spectra non cleavable crosslinked peptides."""
        spectrum_input = pd.read_json(
            Path(__file__).parent / "data" / "annotation_xl_noncl_input.json", orient="records"
        )

        expected_result = pd.read_json(
            Path(__file__).parent / "data" / "annotation_xl_noncl_output.json", orient="records"
        )

        result = annotation.annotate_spectra(spectrum_input)
        pd.testing.assert_frame_equal(expected_result, result)

    def test_annotate_spectra_cl_xl(self):
        """Test annotate spectra cleavable crosslinked peptides."""
        spectrum_input = pd.read_json(Path(__file__).parent / "data" / "annotation_xl_cl_input.json", orient="records")
        expected_result = pd.read_json(
            Path(__file__).parent / "data" / "annotation_xl_cl_output.json", orient="records"
        )

        result = annotation.annotate_spectra(spectrum_input)
        pd.testing.assert_frame_equal(expected_result, result)

    def test_annotate_spectra_tmt(self):
        """Test annotate TMT spectra."""
        spectrum_input = pd.read_json(
            Path(__file__).parent / "data/tmt_spectrum_input.json",
            # converters={"INTENSITIES": literal_eval, "MZ": literal_eval},
        )
        expected_result = pd.read_json(
            Path(__file__).parent / "data/tmt_spectrum_output.json",
        )

        spectrum_input["INTENSITIES"] = spectrum_input["INTENSITIES"].map(lambda intensities: np.array(intensities))
        spectrum_input["MZ"] = spectrum_input["MZ"].map(lambda mz: np.array(mz))

        result = annotation.annotate_spectra(spectrum_input)
        # Only assert columns present in legacy expected output.
        # New per-PSM metrics (sc_features etc.) are tested separately.
        pd.testing.assert_frame_equal(expected_result, result[expected_result.columns])

    def test_annotate_spectra_matching_method_nearest_is_default(self):
        """matching_method='nearest' must reproduce the default behaviour exactly."""
        spectrum_input = pd.read_csv(
            Path(__file__).parent / "data/spectrum_input.csv",
            index_col=0,
            converters={"INTENSITIES": literal_eval, "MZ": literal_eval},
        )
        spectrum_input["INTENSITIES"] = spectrum_input["INTENSITIES"].map(lambda intensities: np.array(intensities))
        spectrum_input["MZ"] = spectrum_input["MZ"].map(lambda mz: np.array(mz))

        default = annotation.annotate_spectra(spectrum_input.copy())
        explicit = annotation.annotate_spectra(spectrum_input.copy(), matching_method="nearest")
        pd.testing.assert_frame_equal(default, explicit)

    def test_annotate_spectra_global_ransac_runs(self):
        """The global_ransac resolver runs end-to-end and yields a valid matrix."""
        spectrum_input = pd.read_csv(
            Path(__file__).parent / "data/spectrum_input.csv",
            index_col=0,
            converters={"INTENSITIES": literal_eval, "MZ": literal_eval},
        )
        spectrum_input["INTENSITIES"] = spectrum_input["INTENSITIES"].map(lambda intensities: np.array(intensities))
        spectrum_input["MZ"] = spectrum_input["MZ"].map(lambda mz: np.array(mz))

        default = annotation.annotate_spectra(spectrum_input.copy())
        result = annotation.annotate_spectra(
            spectrum_input.copy(),
            matching_method="global_ransac",
            mass_tolerance=20,
            unit_mass_tolerance="ppm",
            matching_method_params={"unique_peak": False},
        )
        # same shape/columns as the default path, finite intensities
        self.assertEqual(len(result), len(default))
        self.assertListEqual(list(result.columns), list(default.columns))
        self.assertFalse(np.isnan(np.stack(result["INTENSITIES"].values)).any())

    def test_annotate_spectra_dp_ladder_runs(self):
        """The dp_ladder resolver runs end-to-end and yields a valid matrix."""
        spectrum_input = pd.read_csv(
            Path(__file__).parent / "data/spectrum_input.csv",
            index_col=0,
            converters={"INTENSITIES": literal_eval, "MZ": literal_eval},
        )
        spectrum_input["INTENSITIES"] = spectrum_input["INTENSITIES"].map(lambda intensities: np.array(intensities))
        spectrum_input["MZ"] = spectrum_input["MZ"].map(lambda mz: np.array(mz))

        default = annotation.annotate_spectra(spectrum_input.copy())
        result = annotation.annotate_spectra(
            spectrum_input.copy(),
            matching_method="dp_ladder",
            mass_tolerance=20,
            unit_mass_tolerance="ppm",
            matching_method_params={"ladder_weight": 2.0, "intensity_weight": 0.1},
        )
        # same shape/columns as the default path, finite intensities
        self.assertEqual(len(result), len(default))
        self.assertListEqual(list(result.columns), list(default.columns))
        self.assertFalse(np.isnan(np.stack(result["INTENSITIES"].values)).any())

    def test_annotate_spectra_dp_calibrated_runs(self):
        """The dp_calibrated resolver runs end-to-end and yields a valid matrix."""
        spectrum_input = pd.read_csv(
            Path(__file__).parent / "data/spectrum_input.csv",
            index_col=0,
            converters={"INTENSITIES": literal_eval, "MZ": literal_eval},
        )
        spectrum_input["INTENSITIES"] = spectrum_input["INTENSITIES"].map(lambda intensities: np.array(intensities))
        spectrum_input["MZ"] = spectrum_input["MZ"].map(lambda mz: np.array(mz))

        default = annotation.annotate_spectra(spectrum_input.copy())
        result = annotation.annotate_spectra(
            spectrum_input.copy(),
            matching_method="dp_calibrated",
            mass_tolerance=20,
            unit_mass_tolerance="ppm",
            matching_method_params={"iterations": 2, "ladder_weight": 1.0},
        )
        self.assertEqual(len(result), len(default))
        self.assertListEqual(list(result.columns), list(default.columns))
        self.assertFalse(np.isnan(np.stack(result["INTENSITIES"].values)).any())

    def test_annotate_spectra_unknown_matching_method_raises(self):
        """An unregistered matching_method surfaces as a ValueError."""
        spectrum_input = pd.read_csv(
            Path(__file__).parent / "data/spectrum_input.csv",
            index_col=0,
            converters={"INTENSITIES": literal_eval, "MZ": literal_eval},
        )
        spectrum_input["INTENSITIES"] = spectrum_input["INTENSITIES"].map(lambda intensities: np.array(intensities))
        spectrum_input["MZ"] = spectrum_input["MZ"].map(lambda mz: np.array(mz))

        with self.assertRaises(ValueError):
            annotation.annotate_spectra(spectrum_input, matching_method="not_a_matcher")

    def test_handle_multiple_matches(self):
        """Test handle_multiple_matches function."""
        # Example input data with multiple matches. They don't make biological sense but it tests
        # the mathematical correctness.
        matched_peaks = [
            {"ion_type": "b", "no": 2, "charge": 1, "exp_mass": 200, "theoretical_mass": 198, "intensity": 0.05},
            {"ion_type": "b", "no": 2, "charge": 1, "exp_mass": 205, "theoretical_mass": 198, "intensity": 0.01},
            {"ion_type": "y", "no": 3, "charge": 1, "exp_mass": 300, "theoretical_mass": 303, "intensity": 0.1},
            {"ion_type": "y", "no": 3, "charge": 1, "exp_mass": 303, "theoretical_mass": 303, "intensity": 0.05},
        ]

        # Expected output with sorted by mass_diff, the forth and first, i.e. index 3 and 0 should be returned.
        expected_df_mass_diff = pd.DataFrame(
            [
                {
                    "ion_type": "y",
                    "no": 3,
                    "charge": 1,
                    "exp_mass": 303,
                    "theoretical_mass": 303,
                    "intensity": 0.05,
                    "mass_diff": 0,
                },
                {
                    "ion_type": "b",
                    "no": 2,
                    "charge": 1,
                    "exp_mass": 200,
                    "theoretical_mass": 198,
                    "intensity": 0.05,
                    "mass_diff": 2,
                },
            ],
            index=[3, 0],
        )
        expected_diff_mass_diff = 2

        # Expected output with sorted by intensity, the third and first, i.e. index 2 and 0 should be returned.
        expected_df_intensity = pd.DataFrame(
            [
                {"ion_type": "y", "no": 3, "charge": 1, "exp_mass": 300, "theoretical_mass": 303, "intensity": 0.1},
                {"ion_type": "b", "no": 2, "charge": 1, "exp_mass": 200, "theoretical_mass": 198, "intensity": 0.05},
            ],
            index=[2, 0],
        )
        expected_diff_intensity = 2

        # Expected output with sorted by exp_mass, the third and first, i.e. index 2 and 0 should be returned.
        expected_df_exp_mass = pd.DataFrame(
            [
                {"ion_type": "y", "no": 3, "charge": 1, "exp_mass": 303, "theoretical_mass": 303, "intensity": 0.05},
                {"ion_type": "b", "no": 2, "charge": 1, "exp_mass": 205, "theoretical_mass": 198, "intensity": 0.01},
            ],
            index=[3, 1],
        )
        expected_diff_exp_mass = 2

        # Test with sort_by=diff_mass
        actual_df_mass_diff, actual_diff_mass_diff = annotation.handle_multiple_matches(
            matched_peaks, sort_by="mass_diff"
        )
        pd.testing.assert_frame_equal(expected_df_mass_diff, actual_df_mass_diff)
        self.assertEqual(expected_diff_mass_diff, actual_diff_mass_diff)

        # Test with sort_by=intensity
        actual_df_intensity, actual_diff_intensity = annotation.handle_multiple_matches(
            matched_peaks, sort_by="intensity"
        )
        pd.testing.assert_frame_equal(expected_df_intensity, actual_df_intensity)
        self.assertEqual(expected_diff_intensity, actual_diff_intensity)

        # Test with sort_by=exp_mass
        actual_df_exp_mass, length_diff_exp_mass = annotation.handle_multiple_matches(matched_peaks, sort_by="exp_mass")
        pd.testing.assert_frame_equal(expected_df_exp_mass, actual_df_exp_mass)

        self.assertEqual(expected_diff_exp_mass, length_diff_exp_mass)

        # Test with illegal sort_by
        self.assertRaises(
            ValueError,
            annotation.handle_multiple_matches,
            matched_peaks,
            sort_by="illegal",
        )

    def test_annotate_spectra_returns_sc_features(self):
        """annotate_spectra always returns sc_features dict with ppm_error stats per PSM."""
        spectrum_input = pd.read_csv(
            Path(__file__).parent / "data/spectrum_input.csv",
            index_col=0,
            converters={"INTENSITIES": literal_eval, "MZ": literal_eval},
        )
        spectrum_input["INTENSITIES"] = spectrum_input["INTENSITIES"].map(lambda intensities: np.array(intensities))
        spectrum_input["MZ"] = spectrum_input["MZ"].map(lambda mz: np.array(mz))

        result = annotation.annotate_spectra(spectrum_input)

        # sc_features column must always be present
        self.assertIn("sc_features", result.columns)

        # every PSM must contain exactly the expected keys (ppm_error + intensity + peak-coverage)
        expected_keys = set(constants.SC_FEATURE_KEYS)
        for sc_feat in result["sc_features"]:
            self.assertEqual(set(sc_feat.keys()), expected_keys)
            # peak-coverage features are fractions in [0, 1] (or NaN when undefined)
            for key in constants.PEAK_COVERAGE_FEATURES:
                value = sc_feat[key]
                if not np.isnan(value):
                    self.assertGreaterEqual(value, 0.0)
                    self.assertLessEqual(value, 1.0)

    def test_intensity_coverage_is_between_0_and_1(self):
        """intensity_coverage must be in [0, 1] for matched spectra."""
        spectrum_input = pd.read_csv(
            Path(__file__).parent / "data/spectrum_input.csv",
            index_col=0,
            converters={"INTENSITIES": literal_eval, "MZ": literal_eval},
        )
        spectrum_input["INTENSITIES"] = spectrum_input["INTENSITIES"].map(lambda intensities: np.array(intensities))
        spectrum_input["MZ"] = spectrum_input["MZ"].map(lambda mz: np.array(mz))

        result = annotation.annotate_spectra(spectrum_input)

        for sc_feat in result["sc_features"]:
            cov = sc_feat["intensity_coverage"]
            if not (cov != cov):  # skip NaN
                self.assertGreaterEqual(cov, 0.0)
                self.assertLessEqual(cov, 1.0)


class TestPeakCoverageFeatures(unittest.TestCase):
    """Unit tests for the peak-coverage sc_features helpers."""

    def test_peak_coverage_features_thresholds(self):
        """Threshold-based coverage fractions match a hand-computed example."""
        peaks_mz = np.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0])
        peaks_int = np.array([60.0, 200.0, 30.0, 90.0, 100.0, 12.0])
        # Matched observed peaks: m/z 200 (int 200) and 500 (int 100).
        matched_exp_mass = np.array([200.0, 500.0])

        feats = annotation._peak_coverage_features(matched_exp_mass, peaks_mz, peaks_int)

        # avg_matched = 150, n_matched = 2, unmatched int = [60, 30, 90, 12]
        # avg20 (thr 30):  competing {60, 30, 90}        -> 2 / (2 + 3) = 0.4
        # all:             2 / 6                          -> 0.3333...
        # intensity_coverage: matched raw (200 + 100) / total raw (492) -> 300 / 492
        self.assertAlmostEqual(feats["annotated_frac_avg20"], 0.4)
        self.assertAlmostEqual(feats["annotated_frac_all"], 2.0 / 6.0)
        self.assertAlmostEqual(feats["intensity_coverage"], 300.0 / 492.0)

    def test_peak_coverage_features_no_matches_returns_nan(self):
        """No matches (or no observed peaks) -> all coverage features are NaN."""
        peaks_mz = np.array([100.0, 200.0])
        peaks_int = np.array([10.0, 20.0])

        feats = annotation._peak_coverage_features(np.array([]), peaks_mz, peaks_int)
        expected_keys = set(constants.INTENSITY_COVERAGE_FEATURES) | set(constants.PEAK_COVERAGE_FEATURES)
        self.assertEqual(set(feats.keys()), expected_keys)
        for value in feats.values():
            self.assertTrue(np.isnan(value))

        feats_empty = annotation._peak_coverage_features(np.array([200.0]), np.array([]), np.array([]))
        for value in feats_empty.values():
            self.assertTrue(np.isnan(value))


class TestIonSeriesFeatures(unittest.TestCase):
    """Unit tests for the fragment-ion series-continuity sc_features."""

    def test_longest_consecutive(self):
        """_longest_consecutive finds the longest run of consecutive integers."""
        self.assertEqual(annotation._longest_consecutive(set()), 0)
        self.assertEqual(annotation._longest_consecutive({4}), 1)
        # 1,2,3 is a run of 3; 7,8 is a run of 2 -> 3 wins
        self.assertEqual(annotation._longest_consecutive({1, 2, 3, 7, 8}), 3)
        # scattered singletons never beat a run of 1
        self.assertEqual(annotation._longest_consecutive({1, 5, 9}), 1)

    def test_series_features_collapse_charge_states(self):
        """Position k counts once however many charge states matched there."""
        matched = pd.DataFrame(
            {
                # y: positions 1,2,3 (3 consecutive, 2 sees two charges) and 6
                # b: positions 2 and 5 -> longest run 1
                "ion_type": ["y", "y", "y", "y", "y", "b", "b"],
                "no": [1, 2, 2, 3, 6, 2, 5],
                "charge": [1, 1, 2, 1, 1, 1, 1],
            }
        )
        feats = annotation._ion_series_features(matched, "PEPTIDEK")  # 8 residues -> 7 sites

        self.assertEqual(set(feats.keys()), set(constants.SERIES_FEATURES))
        self.assertEqual(feats["longest_y_series"], 3.0)
        self.assertEqual(feats["longest_b_series"], 1.0)
        self.assertAlmostEqual(feats["longest_series_frac"], 3.0 / 7.0)

    def test_series_features_missing_columns_return_nan(self):
        """Without the ion_type/no columns the features are NaN, not zero."""
        feats = annotation._ion_series_features(pd.DataFrame({"intensity": [1.0]}), "PEPTIDEK")
        for value in feats.values():
            self.assertTrue(np.isnan(value))


class TestReporterFeatures(unittest.TestCase):
    """Unit tests for the TMT11 reporter-ion sc_features."""

    def test_reporter_features_counts_channels(self):
        """Channels inside the ppm window are counted; the most intense peak wins."""
        rep = constants.TMT11_REPORTER_MZ
        # three channels present (one with two peaks in-window), plus a b/y peak far above
        peaks_mz = np.array([rep[0], rep[1], rep[1] + rep[1] * 5e-6, rep[2], 800.0])
        peaks_int = np.array([100.0, 20.0, 50.0, 30.0, 300.0])

        feats = annotation._reporter_features(peaks_mz, peaks_int, tolerance_ppm=20.0)

        self.assertEqual(set(feats.keys()), set(constants.REPORTER_FEATURES))
        self.assertEqual(feats["n_reporter_channels"], 3.0)
        # channel intensities: 100 (126C), 50 (127N, max of 20/50), 30 (127C) = 180
        self.assertAlmostEqual(feats["reporter_intensity_frac"], 180.0 / 500.0)
        self.assertAlmostEqual(feats["reporter_max_frac"], 100.0 / 180.0)

    def test_reporter_features_outside_window_not_counted(self):
        """A peak beyond the ppm window does not count as a channel."""
        rep = constants.TMT11_REPORTER_MZ
        off = rep[0] + rep[0] * 50e-6  # 50 ppm away
        feats = annotation._reporter_features(np.array([rep[0], off]), np.array([10.0, 99.0]), tolerance_ppm=20.0)
        self.assertEqual(feats["n_reporter_channels"], 1.0)

    def test_reporter_features_no_reporter_region_returns_nan(self):
        """An empty reporter region is NaN, not '0 channels'."""
        feats = annotation._reporter_features(np.array([500.0, 800.0]), np.array([10.0, 20.0]))
        for value in feats.values():
            self.assertTrue(np.isnan(value))

        feats_empty = annotation._reporter_features(np.array([]), np.array([]))
        for value in feats_empty.values():
            self.assertTrue(np.isnan(value))
