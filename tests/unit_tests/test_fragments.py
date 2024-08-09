import json
import unittest
from pathlib import Path

from numpy.testing import assert_almost_equal

import spectrum_fundamentals.constants as c
import spectrum_fundamentals.fragments as fragments


class TestInitializePeaks(unittest.TestCase):
    """Class to test initialize_peaks function."""

    def _test_outputs(self, expected_input_file: Path, fragmentation_method: str):

        with open(expected_input_file) as file:
            expected_list_out = json.load(file)

        expected_tmt_nt_term = 1
        expected_peptide_sequence = "PEPTIDE"
        expected_mass_s = 799.3599646700001

        actual_list_out, actual_tmt_n_term, actual_peptide_sequence, actual_calc_mass_s = fragments.initialize_peaks(
            sequence="PEPTIDE",
            mass_analyzer="FTMS",
            charge=3,
            fragmentation_method=fragmentation_method,
        )

        self.assertEqual(actual_list_out, expected_list_out)
        self.assertEqual(actual_tmt_n_term, expected_tmt_nt_term)
        self.assertEqual(actual_peptide_sequence, expected_peptide_sequence)
        assert_almost_equal(actual_calc_mass_s, expected_mass_s, decimal=5)

    def test_initialize_peaks_hcd_cid(self):
        """Test initialize_peaks for HCD / CID input."""
        self._test_outputs(Path(__file__).parent / "data/fragments_meta_data_hcd_cid.json", "HCD")
        self._test_outputs(Path(__file__).parent / "data/fragments_meta_data_hcd_cid.json", "CID")

    def test_initialize_peaks_etd_ecd(self):
        """Test initialize_peaks for ETD / ECD input."""
        self._test_outputs(Path(__file__).parent / "data/fragments_meta_data_etd_ecd.json", "ETD")
        self._test_outputs(Path(__file__).parent / "data/fragments_meta_data_etd_ecd.json", "ECD")

    def test_initialize_peaks_ethcd_etcid(self):
        """Test initialize_peaks for ETCID / ETHCD input."""
        self._test_outputs(Path(__file__).parent / "data/fragments_meta_data_ethcd_etcid.json", "ETHCD")
        self._test_outputs(Path(__file__).parent / "data/fragments_meta_data_ethcd_etcid.json", "ETCID")

    def test_initialize_peaks_uvpd(self):
        """Test initialize_peaks with basic input, but for all six ion types for UVPD."""
        self._test_outputs(Path(__file__).parent / "data/fragments_meta_data_uvpd.json", "UVPD")

    def _test_xl_outputs(self, expected_input_file: Path, **fragments_input):

        with open(expected_input_file) as file:
            expected_list_out = json.load(file)

        expected_tmt_nt_term = 1
        expected_peptide_sequence = "PEKTIDE"
        expected_mass = 830.40216367

        (
            actual_fragments_meta_data,
            actual_tmt_n_term,
            actual_peptide_sequence,
            actual_mass,
        ) = fragments.initialize_peaks_xl(**fragments_input)

        self.assertEqual(actual_fragments_meta_data, expected_list_out)
        self.assertEqual(actual_tmt_n_term, expected_tmt_nt_term)
        self.assertEqual(actual_peptide_sequence, expected_peptide_sequence)
        assert_almost_equal(actual_mass, expected_mass, decimal=5)

    def test_initialize_peaks_non_cl_xl(self):
        """Test initialize_peaks_xl with basic input for non-cleavable crosslinked peptides."""
        initialize_peaks_xl_input = {
            "sequence": "PEK[UNIMOD:1898]TIDE",
            "mass_analyzer": "FTMS",
            "crosslinker_position": 3,
            "crosslinker_type": "BS3",
            "sequence_beta": "AK[UNIMOD:1898]AT",
        }
        self._test_xl_outputs(
            Path(__file__).parent / "data/fragments_meta_data_non_cl_xl.json", **initialize_peaks_xl_input
        )

    def test_initialize_peaks_cl_xl(self):
        """Test initialize_peaks_xl with basic input for cleavable crosslinked peptides."""
        initialize_peaks_xl_input = {
            "sequence": "PEK[UNIMOD:1896]TIDE",
            "mass_analyzer": "FTMS",
            "crosslinker_position": 3,
            "crosslinker_type": "DSSO",
        }
        self._test_xl_outputs(
            Path(__file__).parent / "data/fragments_meta_data_cl_xl.json", **initialize_peaks_xl_input
        )


class TestFragmentationMethod(unittest.TestCase):
    """Class to test the retrieving of the IonTypes."""

    def test_get_ion_types_hcd(self):
        """Test retrieving ion types for HCD."""
        assert fragments.retrieve_ion_types("HCD") == ["y", "b"]

    def test_get_ion_types_etd(self):
        """Test retrieving ion types for ETD."""
        assert fragments.retrieve_ion_types("ETD") == ["z", "c"]

    def test_get_ion_types_etcid(self):
        """Test retrieving ion types for ETCID."""
        assert fragments.retrieve_ion_types("ETCID") == ["y", "b", "z", "c"]

    def test_get_ion_types_lower_case(self):
        """Test lower case fragmentation method."""
        assert fragments.retrieve_ion_types("uvpd") == ["y", "b", "z", "c", "x", "a"]

    def test_invalid_fragmentation_method(self):
        """Test if error is raised for invalid fragmentation method."""
        self.assertRaises(ValueError, fragments.retrieve_ion_types, "XYZ")


class TestFragmentationMethodForPeakInitialization(unittest.TestCase):
    """Class to test the retrieving of the IonTypes for peak initialization."""

    def test_get_ion_types_hcd(self):
        """Test retrieving ion types for HCD."""
        assert fragments.retrieve_ion_types_for_peak_initialization("HCD") == ["y", "b"]

    def test_get_ion_types_etd(self):
        """Test retrieving ion types for ETD."""
        assert fragments.retrieve_ion_types_for_peak_initialization("ETD") == ["z_r", "c"]

    def test_get_ion_types_etcid(self):
        """Test retrieving ion types for ETCID."""
        assert fragments.retrieve_ion_types_for_peak_initialization("ETCID") == ["y", "z", "b", "c"]

    def test_get_ion_types_lower_case(self):
        """Test lower case fragmentation method."""
        assert fragments.retrieve_ion_types_for_peak_initialization("uvpd") == ["x", "y", "z", "a", "b", "c"]

    def test_invalid_fragmentation_method(self):
        """Test if error is raised for invalid fragmentation method."""
        self.assertRaises(ValueError, fragments.retrieve_ion_types_for_peak_initialization, "XYZ")


class TestFragmentIonAnnotation(unittest.TestCase):
    """Tests for fragment ion annotation generation."""

    def test_generate_fragment_ion_types(self):
        """Test if output ordering is valid."""
        for ion_types in [["y", "b"], ["b", "y"], ["y", "b", "x", "a"]]:
            order = ("position", "ion_type", "charge")
            annotations = [
                f"{ion_type}{pos}+{charge}" for pos in c.POSITIONS for ion_type in ion_types for charge in c.CHARGES
            ]
            with self.subTest(order=order, ion_types=ion_types):
                self.assertEqual(
                    fragments.generate_fragment_ion_annotations(ion_types=ion_types, order=order), annotations
                )

            order = ("ion_type", "position", "charge")
            annotations = [
                f"{ion_type}{pos}+{charge}" for ion_type in ion_types for pos in c.POSITIONS for charge in c.CHARGES
            ]
            with self.subTest(order=order, ion_types=ion_types):
                self.assertEqual(
                    fragments.generate_fragment_ion_annotations(ion_types=ion_types, order=order), annotations
                )

    def test_catches_unsupported_ion_types(self):
        """Check if unsupported ion types are caught."""
        for ion_types in [[], ["y", "y"], ["b", "y", "q"]]:
            with self.subTest(ion_types=ion_types), self.assertRaises(ValueError):
                _ = fragments.generate_fragment_ion_annotations(
                    ion_types=["b", "y", "q"], order=("position", "ion_type", "charge")
                )

    def test_catches_redundant_order(self):
        """Check if redundant order is caught."""
        with self.assertRaises(ValueError):
            _ = fragments.generate_fragment_ion_annotations(
                ion_types=["y", "b"], order=("ion_type", "position", "ion_type")
            )
