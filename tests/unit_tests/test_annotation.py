import unittest
from ast import literal_eval
from pathlib import Path

import numpy as np
import pandas as pd

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
        pd.testing.assert_frame_equal(expected_result, result)

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
        pd.testing.assert_frame_equal(expected_result, result)

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
        pd.testing.assert_frame_equal(expected_result, result)

    def test_handle_multiple_matches(self):
        """Test handle_multiple_matches function."""
        # Example input data with multiple matches. They don't make biological sense but it tests the mathematical correctness.
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
