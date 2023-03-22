import unittest

import numpy as np
import pandas as pd

from spectrum_fundamentals.annotation import annotation


class TestMatchPeaks(unittest.TestCase):
    """TestClass for everything in annotation."""

    def test_match_peaks(self):
        """Test match_peaks function."""
        # Input data
        fragments_meta_data = [
            {"ion_type": "b", "no": 1, "charge": 1, "mass": 101.07, "min_mass": 101.05, "max_mass": 101.1},
            {"ion_type": "b", "no": 2, "charge": 1, "mass": 216.12, "min_mass": 216.1, "max_mass": 216.15},
            {"ion_type": "y", "no": 1, "charge": 1, "mass": 175.12, "min_mass": 175.1, "max_mass": 175.15},
            {"ion_type": "y", "no": 2, "charge": 1, "mass": 290.17, "min_mass": 290.15, "max_mass": 290.2},
        ]
        peaks_intensity = np.array([1.0, 0.5, 0.2, 0.1])
        peaks_masses = np.array([101.07, 216.12, 175.12, 290.17])
        tmt_n_term = 1
        unmod_sequence = "RKPQQFFGLM"
        charge = 1

        # Expected output
        expected_output = [
            {"ion_type": "b", "no": 1, "charge": 1, "exp_mass": 101.07, "theoretical_mass": 101.07, "intensity": 1.0},
            {"ion_type": "b", "no": 2, "charge": 1, "exp_mass": 216.12, "theoretical_mass": 216.12, "intensity": 0.5},
            {"ion_type": "y", "no": 1, "charge": 1, "exp_mass": 175.12, "theoretical_mass": 175.12, "intensity": 0.2},
            {"ion_type": "y", "no": 2, "charge": 1, "exp_mass": 290.17, "theoretical_mass": 290.17, "intensity": 0.1},
        ]

        # Test the function
        result = annotation.match_peaks(
            fragments_meta_data, peaks_intensity, peaks_masses, tmt_n_term, unmod_sequence, charge
        )
        self.assertEqual(result, expected_output)

    def test_handle_multiple_matches(self):
        """Test handle_multiple_matches function."""
        # Example input data with multiple matches
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
                {"ion_type": "b", "no": 2, "charge": 1, "exp_mass": 200, "theoretical_mass": 198, "intensity": 0.05},
                {"ion_type": "y", "no": 3, "charge": 1, "exp_mass": 300, "theoretical_mass": 303, "intensity": 0.1},
            ],
            index=[0, 3],
        )
        expected_diff_exp_mass = 2

        # Test with sort_by=diff_mass
        actual_df_mass_diff, actual_diff_mass_diff = annotation.handle_multiple_matches(
            matched_peaks, sort_by="mass_diff"
        )
        pd.testing.assertEqual(expected_df_mass_diff, actual_df_mass_diff)
        self.assertEqual(expected_diff_mass_diff, actual_diff_mass_diff)

        # Test with sort_by=intensity
        actual_df_intensity, actual_diff_intensity = annotation.handle_multiple_matches(
            matched_peaks, sort_by="intensity"
        )
        pd.testing.assertEqual(expected_df_intensity, actual_df_intensity)
        self.assertEqual(expected_diff_intensity, actual_diff_intensity)

        # Test with sort_by=exp_mass
        actual_df_exp_mass, length_diff_exp_mass = annotation.handle_multiple_matches(matched_peaks, sort_by="exp_mass")
        pd.testing.assertEqual(expected_df_exp_mass, actual_df_exp_mass)

        self.assertEqual(expected_diff_exp_mass, length_diff_exp_mass)

        # Test with illegal sort_by
        self.assertRaises(ValueError, annotation.handle_multiple_matches, matched_peaks, sort_by="illegal")
