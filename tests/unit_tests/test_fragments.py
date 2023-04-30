import unittest

from numpy.testing import assert_almost_equal

import spectrum_fundamentals.fragments as fragments


class TestGetModifications:
    """Class to test get modifications."""

    def test_get_modifications(self):
        """Test get_modifications."""
        assert fragments._get_modifications("ABC") == ({}, 1, "ABC")

    def test_get_modifications_carbamidomethylation(self):
        """Test get_modifications."""
        assert fragments._get_modifications("ABC[UNIMOD:4]") == ({2: 57.02146}, 1, "ABC")

    def test_get_modifications_tmt_tag(self):
        """Test get_modifications."""
        assert fragments._get_modifications("[UNIMOD:737]ABC[UNIMOD:4]") == ({0: 229.162932, 2: 57.02146}, 2, "ABC")

    def test_get_modifications_tmtpro_tag(self):
        """Test get_modifications."""
        assert fragments._get_modifications("[UNIMOD:2016]ABC[UNIMOD:4]") == ({0: 304.207146, 2: 57.02146}, 2, "ABC")


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
        self.assertEqual(fragments.compute_peptide_mass(seq), 1045.2561516699998)

    def test_compute_peptide_masses_with_invalid_syntax(self):
        """Negative testing of comuptation of peptide mass with unsupported syntax of mod string."""
        seq = "SEQUEM(Ox.)CE"
        self.assertRaises(AssertionError, fragments.compute_peptide_mass, seq)

    def test_compute_peptide_masses_with_invalid_mod(self):
        """Negative testing of computation of peptide mass with unknown modification in mod string."""
        seq = "SEQUENC[UNIMOD:0]E"
        self.assertRaises(AssertionError, fragments.compute_peptide_mass, seq)
