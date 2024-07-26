import json
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal

import spectrum_fundamentals.fragments as fragments


class TestGetModifications:
    """Class to test get modifications."""

    def test_get_modifications(self):
        """Test get_modifications."""
        assert fragments._get_modifications("ABC") == {}

    def test_get_modifications_carbamidomethylation(self):
        """Test get_modifications."""
        assert fragments._get_modifications("ABC[UNIMOD:4]") == {2: 57.021464}

    def test_get_modifications_tmt_tag(self):
        """Test get_modifications."""
        assert fragments._get_modifications("[UNIMOD:737]-ABC[UNIMOD:4]") == {-2: 229.162932, 2: 57.021464}

    def test_get_modifications_tmtpro_tag(self):
        """Test get_modifications."""
        assert fragments._get_modifications("[UNIMOD:2016]-ABC[UNIMOD:4]") == {-2: 304.207146, 2: 57.021464}


class TestComputeMasses(unittest.TestCase):
    """Class to test compute ion and peptide masses."""

    def test_compute_ion_masses(self):
        """
        Test compute ion masses.

        >>> from pyteomics import mass
        >>> mass.calculate_mass(sequence='AD', ion_type='b', charge=1)
        """
        masses = fragments.compute_ion_masses([1, 3, 4] + [0] * 27, [1, 0, 0, 0, 0, 0])  # peptide = ADE, charge = 1
        assert_almost_equal(masses[0], 148.06044, decimal=5)  # y1 E.-
        assert_almost_equal(masses[1], -1.0, decimal=5)  # y1;2+: n.a.
        assert_almost_equal(masses[3], 72.04439, decimal=5)  # b1: -.A
        assert_almost_equal(masses[6], 263.08738, decimal=5)  # y2 DE.-
        assert_almost_equal(masses[9], 187.07133, decimal=5)  # b2: -.AD

    def test_compute_ion_masses_tmtpro(self):
        """
        Test compute ion masses.

        >>> from pyteomics import mass
        >>> mass.calculate_mass(sequence='AD', ion_type='b', charge=1)
        """
        masses = fragments.compute_ion_masses(
            [1, 3, 4] + [0] * 27, [1, 0, 0, 0, 0, 0], "tmtpro"
        )  # peptide = [UNIMOD:2016]ADE, charge = 1
        assert_almost_equal(masses[0], 148.06044, decimal=5)  # y1 E.-
        assert_almost_equal(masses[1], -1.0, decimal=5)  # y1;2+: n.a.
        assert_almost_equal(masses[3], 72.04439 + 304.207146, decimal=5)  # b1: -.A
        assert_almost_equal(masses[6], 263.08738, decimal=5)  # y2 DE.-
        self.assertAlmostEqual(masses[9], 187.07133 + 304.207146, places=5)  # b2: -.AD

    def test_compute_peptide_masses(self):
        """Test computation of peptide masses with valid input."""
        seq = "SEQUENC[UNIMOD:4]E"
        self.assertEqual(fragments.compute_peptide_mass(seq), 1045.2561556699998)

    def test_compute_peptide_masses_tmtpro(self):
        """Test computation of peptide masses with valid input and tmt tag."""
        seq = "[UNIMOD:737]-SEQUENC[UNIMOD:4]E"
        self.assertEqual(fragments.compute_peptide_mass(seq), 1274.41908767)

    def test_compute_peptide_masses_with_invalid_syntax(self):
        """Negative testing of comuptation of peptide mass with unsupported syntax of mod string."""
        seq = "SEQUEM(Ox.)CE"
        self.assertRaises(KeyError, fragments.compute_peptide_mass, seq)

    def test_compute_peptide_masses_with_invalid_mod(self):
        """Negative testing of computation of peptide mass with unknown modification in mod string."""
        seq = "SEQUENC[UNIMOD:0]E"
        self.assertRaises(KeyError, fragments.compute_peptide_mass, seq)


class TestMassTolerances(unittest.TestCase):
    """Testing the mass tolerance calculations in various scenarios."""

    def test_mass_tol_with_ppm(self):
        """Test get_min_max_mass with a user defined ppm measure."""
        window = fragments.get_min_max_mass(
            mass_analyzer="FTMS", mass=np.array([10.0]), mass_tolerance=15, unit_mass_tolerance="ppm"
        )
        assert_array_equal(window, np.array([[9.99985], [10.00015]]))

    def test_mass_tol_with_da(self):
        """Test get_min_max_mass with a user defined da measure."""
        window = fragments.get_min_max_mass(
            mass_analyzer="FTMS", mass=np.array([10.0]), mass_tolerance=0.3, unit_mass_tolerance="da"
        )
        assert_array_equal(window, np.array([[9.7], [10.3]]))

    def test_mass_tol_with_defaults(self):
        """Test get_min_max_mass with mass analyzer defaults."""
        window_ftms = fragments.get_min_max_mass(mass_analyzer="FTMS", mass=np.array([10.0]))
        window_itms = fragments.get_min_max_mass(mass_analyzer="ITMS", mass=np.array([10.0]))
        window_tof = fragments.get_min_max_mass(mass_analyzer="TOF", mass=np.array([10.0]))

        assert_array_equal(window_ftms, np.array([[9.9998], [10.0002]]))
        assert_array_equal(window_tof, np.array([[9.9996], [10.0004]]))
        assert_array_equal(window_itms, np.array([[9.65], [10.35]]))


class TestInitializePeaks(unittest.TestCase):
    """Class to test initialize_peaks function."""

    def test_initialize_peaks(self):
        """Test initialize_peaks with basic input."""
        fragments_input = {"sequence": "AAAA", "mass_analyzer": "FTMS", "charge": 3, "noncl_xl": False}

        with open(Path(__file__).parent / "data/fragments_meta_data.json") as file:
            expected_list_out = json.load(file)
        expected_tmt_nt_term = 1
        expected_peptide_sequence = "AAAA"
        expected_mass_s = 302.15902

        actual_list_out, actual_tmt_n_term, actual_peptide_sequence, actual_calc_mass_s = fragments.initialize_peaks(
            **fragments_input
        )

        self.assertEqual(actual_list_out, expected_list_out)
        self.assertEqual(actual_tmt_n_term, expected_tmt_nt_term)
        self.assertEqual(actual_peptide_sequence, expected_peptide_sequence)
        assert_almost_equal(actual_calc_mass_s, expected_mass_s, decimal=5)

    def test_initialize_peaks_non_cl_xl(self):
        """Test initialize_peaks_xl with basic input for non-cleavable crosslinked peptides."""
        initialize_peaks_xl_input = {
            "sequence": "AKC",
            "mass_analyzer": "FTMS",
            "crosslinker_position": 2,
            "crosslinker_type": "BS3",
            "sequence_beta": "AKA",
        }

        with open(Path(__file__).parent / "data/fragments_meta_data_non_cl_xl.json") as file:
            expected_fragments_meta_data = json.load(file)
        expected_tmt_nt_term = 1
        expected_peptide_sequence = "AKC"
        expected_mass = 320.15182

        (
            actual_fragments_meta_data,
            actual_tmt_n_term,
            actual_peptide_sequence,
            actual_mass,
        ) = fragments.initialize_peaks_xl(**initialize_peaks_xl_input)
        self.assertEqual(actual_fragments_meta_data, expected_fragments_meta_data)
        self.assertEqual(actual_tmt_n_term, expected_tmt_nt_term)
        self.assertEqual(actual_peptide_sequence, expected_peptide_sequence)
        assert_almost_equal(actual_mass, expected_mass, decimal=5)

    def test_initialize_peaks_cl_xl(self):
        """Test initialize_peaks_xl with basic input for cleavable crosslinked peptides."""
        initialize_peaks_xl_input = {
            "sequence": "AKC",
            "mass_analyzer": "FTMS",
            "crosslinker_position": 2,
            "crosslinker_type": "DSSO",
            "sequence_beta": "AKA",
        }

        with open(Path(__file__).parent / "data/fragments_meta_data_cl_xl.json") as file:
            expected_fragments_meta_data = json.load(file)
        expected_tmt_nt_term = 1
        expected_peptide_sequence = "AKC"
        expected_mass = 320.15182

        (
            actual_fragments_meta_data,
            actual_tmt_n_term,
            actual_peptide_sequence,
            actual_mass,
        ) = fragments.initialize_peaks_xl(**initialize_peaks_xl_input)
        self.assertEqual(actual_fragments_meta_data, expected_fragments_meta_data)
        self.assertEqual(actual_tmt_n_term, expected_tmt_nt_term)
        self.assertEqual(actual_peptide_sequence, expected_peptide_sequence)
        assert_almost_equal(actual_mass, expected_mass, decimal=5)

    def test_initialize_peaks_all_ions(self):
        """Test initialize_peaks with basic input, but for all six ion types."""
        fragments_input = {
            "sequence": "PEPTIDE",
            "mass_analyzer": "FTMS",
            "charge": 3,
            "noncl_xl": False,
            "fragmentation_method": "UVPD",
        }

        with open(Path(__file__).parent / "data/fragments_meta_data_all_ions.json") as file:
            expected_list_out = json.load(file)
        expected_tmt_nt_term = 1
        expected_peptide_sequence = "PEPTIDE"
        expected_mass_s = 799.359964

        actual_list_out, actual_tmt_n_term, actual_peptide_sequence, actual_calc_mass_s = fragments.initialize_peaks(
            **fragments_input
        )

        self.assertEqual(actual_list_out, expected_list_out)
        self.assertEqual(actual_tmt_n_term, expected_tmt_nt_term)
        self.assertEqual(actual_peptide_sequence, expected_peptide_sequence)
        assert_almost_equal(actual_calc_mass_s, expected_mass_s, decimal=5)


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
        assert fragments.retrieve_ion_types("ETCID") == ["y", "z", "b", "c"]

    def test_get_ion_types_lower_case(self):
        """Test lower case fragmentation method."""
        assert fragments.retrieve_ion_types("uvpd") == ["x", "y", "z", "a", "b", "c"]

    def test_invalid_fragmentation_method(self):
        """Test if error is raised for invalid fragmentation method."""
        self.assertRaises(ValueError, fragments.retrieve_ion_types, "XYZ")
