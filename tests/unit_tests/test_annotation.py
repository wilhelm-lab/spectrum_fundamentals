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
