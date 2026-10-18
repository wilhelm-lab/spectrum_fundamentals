import json
import unittest
from pathlib import Path

from numpy.testing import assert_almost_equal

import spectrum_fundamentals.constants as c
import spectrum_fundamentals.fragments as fragments


class TestInitializePeaks(unittest.TestCase):
    """Class to test initialize_peaks function."""

    def _test_outputs(
        self,
        expected_input_file: Path,
        fragmentation_method: str,
        multifrag: bool = False,
        featured_ions: list[str] | None = None,
    ):
        if featured_ions is None:
            featured_ions = ["y", "b"]

        with open(expected_input_file) as file:
            expected_list_out = json.load(file)

        expected_tmt_nt_term = 1
        expected_peptide_sequence = "PEPTIDE"
        expected_mass_s = 799.3599646700001
        expected_nl_annotated = 0
        expected_window = [266.26059802354666, 268.66059802354664]

        (
            actual_list_out,
            actual_tmt_n_term,
            actual_peptide_sequence,
            actual_calc_mass_s,
            actual_nl_annotated,
            actual_window,
        ) = fragments.initialize_peaks(
            sequence="PEPTIDE",
            mass_analyzer="FTMS",
            charge=3,
            fragmentation_method=fragmentation_method,
            multifrag=multifrag,
            featured_ions=featured_ions,
        )

        self.assertEqual(actual_list_out, expected_list_out)
        self.assertEqual(actual_tmt_n_term, expected_tmt_nt_term)
        self.assertEqual(actual_peptide_sequence, expected_peptide_sequence)
        assert_almost_equal(actual_calc_mass_s, expected_mass_s, decimal=5)
        self.assertEqual(actual_nl_annotated, expected_nl_annotated)
        self.assertEqual(actual_window, expected_window)

    def test_initialize_peaks_hcd_cid(self):
        """Test initialize_peaks for HCD / CID input."""
        self._test_outputs(Path(__file__).parent / "data/fragments_meta_data_hcd_cid.json", "HCD")
        self._test_outputs(Path(__file__).parent / "data/fragments_meta_data_hcd_cid.json", "CID")

    def test_initialize_peaks_ecd_etcid_eid_uvpd(self):
        """Test initialize_peaks for ECD/ETCID/EID/UVPD input."""
        self._test_outputs(
            Path(__file__).parent / "data/fragments_meta_data_ecd.json",
            "ECD",
            True,
            ["a", "A", "b", "c", "C", "y", "x", "X", "z", "Z"],
        )
        self._test_outputs(
            Path(__file__).parent / "data/fragments_meta_data_etcid.json",
            "ETCID",
            True,
            ["a", "A", "b", "c", "C", "y", "x", "X", "z", "Z"],
        )
        self._test_outputs(
            Path(__file__).parent / "data/fragments_meta_data_eid.json",
            "EID",
            True,
            ["a", "A", "b", "c", "C", "y", "x", "X", "z", "Z"],
        )
        self._test_outputs(
            Path(__file__).parent / "data/fragments_meta_data_uvpd.json",
            "UVPD",
            True,
            ["a", "A", "b", "c", "C", "y", "x", "X", "z", "Z"],
        )

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

    def test_get_ion_types_etcid(self):
        """Test retrieving ion types for ETCID."""
        assert fragments.retrieve_ion_types("ETCID") == ["A", "b", "c", "C", "y", "z", "Z"]

    def test_invalid_fragmentation_method(self):
        """Test if error is raised for invalid fragmentation method."""
        self.assertRaises(ValueError, fragments.retrieve_ion_types, "XYZ")


class TestFragmentationMethodForPeakInitialization(unittest.TestCase):
    """Class to test the retrieving of the IonTypes for peak initialization."""

    def test_get_ion_types_hcd(self):
        """Test retrieving ion types for HCD."""
        assert fragments.retrieve_ion_types_for_peak_initialization("HCD") == ["y", "b"]

    def test_get_ion_types_etcid(self):
        """Test retrieving ion types for ETCID."""
        assert fragments.retrieve_ion_types_for_peak_initialization("ETCID") == ["A", "b", "c", "C", "y", "z", "Z"]

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
                (ion_type, pos, charge) for pos in c.POSITIONS for ion_type in ion_types for charge in c.CHARGES
            ]
            with self.subTest(order=order, ion_types=ion_types):
                self.assertEqual(
                    fragments.generate_fragment_ion_annotations(ion_types=ion_types, order=order), annotations
                )

            order = ("ion_type", "position", "charge")
            annotations = [
                (ion_type, pos, charge) for ion_type in ion_types for pos in c.POSITIONS for charge in c.CHARGES
            ]
            with self.subTest(order=order, ion_types=ion_types):
                self.assertEqual(
                    fragments.generate_fragment_ion_annotations(ion_types=ion_types, order=order), annotations
                )

    def test_catches_redundant_order(self):
        """Check if redundant order is caught."""
        with self.assertRaises(ValueError):
            _ = fragments.generate_fragment_ion_annotations(
                ion_types=["y", "b"], order=("ion_type", "position", "ion_type")
            )


class TestNeutralLossFunctions(unittest.TestCase):
    """Tests for neutral loss annotation."""

    def test_add_nl(self):
        """Test the _add_nl function with a range of amino acids and neutral losses."""
        neutral_losses = ["H2O", "NH3"]
        nl_dict = {
            0: ["CO2"],  # Initial dictionary contains CO2 for index 0
            1: ["H2O"],  # Initial dictionary contains H2O for index 1
        }
        start_aa_index = 0
        end_aa_index = 2  # Only indices 0 and 1 will be modified

        # Expected output after the neutral losses are added
        expected_nl_dict = {
            0: ["CO2", "H2O", "NH3"],  # Both H2O and NH3 added
            1: ["H2O", "NH3"],  # NH3 added, H2O already present
        }

        result = fragments._add_nl(neutral_losses, nl_dict, start_aa_index, end_aa_index)
        self.assertEqual(result, expected_nl_dict)

    def test_calculate_nl_score_mass(self):
        """Test the _calculate_nl_score_mass function with various neutral losses."""
        # Test for H2O, which should reduce the score by 5
        score, mass = fragments._calculate_nl_score_mass("H2O")
        self.assertEqual(score, 95)  # Starting score of 100 - 5
        self.assertEqual(mass, 18.01056467)  # Mass of H2O

        # Test for NH3, which should also reduce the score by 5
        score, mass = fragments._calculate_nl_score_mass("NH3")
        self.assertEqual(score, 95)  # Starting score of 100 - 5
        self.assertEqual(mass, 17.026549105)  # Mass of NH3

        # Test for CO2, which should reduce the score by 30
        score, mass = fragments._calculate_nl_score_mass("CO2")
        self.assertEqual(score, 70)  # Starting score of 100 - 30
        self.assertEqual(mass, 43.9898292)  # Mass of CO2

        # Test for C2H4O2, which should reduce the score by 30
        score, mass = fragments._calculate_nl_score_mass("C2H4O2")
        self.assertEqual(score, 70)  # Starting score of 100 - 30
        self.assertEqual(mass, 60.02112934)


class TestNeutralLosses(unittest.TestCase):
    """Test the annotation of the phosphoric acid neutral loss."""

    def test_phospho_loss_on_s_and_t_only_by_default(self):
        """Phosphorylated S loses H3PO4, phosphorylated Y and A do not."""
        for sequence, expected in (("PES[UNIMOD:21]K", True), ("PEY[UNIMOD:21]K", False), ("PEA[UNIMOD:21]K", False)):
            nl_b, nl_y = fragments._get_neutral_losses(
                sequence.replace("[UNIMOD:21]", ""), fragments._get_modifications(sequence)
            )
            has_loss = any("H3O4P" in losses for losses in list(nl_b.values()) + list(nl_y.values()))
            self.assertEqual(has_loss, expected, sequence)

    def test_phospho_loss_on_additional_residues(self):
        """Phosphorylation on a decoy residue (A) loses H3PO4 like S if A is added to the loss residues."""
        sequence = "PEA[UNIMOD:21]K"
        nl_b, nl_y = fragments._get_neutral_losses("PEAK", fragments._get_modifications(sequence), "STA")
        # b ions from b3 (index 2) and y ions from y2 (index 1) contain the phosphorylated A
        self.assertEqual([i for i, losses in nl_b.items() if "H3O4P" in losses], [2])
        self.assertEqual([i for i, losses in nl_y.items() if "H3O4P" in losses], [1, 2])


class TestPlainNeutralLossCounts(unittest.TestCase):
    """Test counting the neutral losses of decoys with and without the loss on the decoy site."""

    def test_plain_counts(self):
        """A phosphorylated A annotated like S expects losses, counted without A (plain) it does not."""
        import numpy as np

        from spectrum_fundamentals.annotation.annotation import _annotate_linear_spectrum

        sequence = "PEA[UNIMOD:21]K"
        # a spectrum with the theoretical b and y ions of the sequence
        peaks = fragments.initialize_peaks(sequence, "FTMS", 2, 20, "ppm")[0]
        mz = np.sort(np.array([peak["mass"] for peak in peaks]))
        spectrum = np.array([sequence, "FTMS", 2, mz, np.ones(len(mz))], dtype=object)
        index = {"MODIFIED_SEQUENCE": 0, "MASS_ANALYZER": 1, "PRECURSOR_CHARGE": 2, "MZ": 3, "INTENSITIES": 4}
        result = _annotate_linear_spectrum(
            spectrum, index, 20, "ppm", add_neutral_losses=True, phospho_loss_residues="STA", plain_loss_residues="ST"
        )
        expected_nl, plain_expected_nl = result[5], result[8]
        self.assertGreater(expected_nl, 0)
        self.assertEqual(plain_expected_nl, 0)
        # without plain residues, the result has no plain counts
        self.assertEqual(len(_annotate_linear_spectrum(spectrum, index, 20, "ppm", add_neutral_losses=True)), 7)
