import unittest

import spectrum_fundamentals.constants as c


class UpdateModMasses(unittest.TestCase):
    """Class to test updating mod masses."""

    def test_update_mod_masses(self):
        """Test _update_mod_masses."""
        custom_mods = {"stat_mods": {"57.0215": ("[UNIMOD:737]", 229.1628)}}
        updated_mods = c.MOD_MASSES.copy()
        updated_mods.update({"[UNIMOD:737]": 229.1628})
        self.assertEqual(c.update_mod_masses(custom_mods), updated_mods)

    """
    def test_update_mod_masses_error(self):
        ""Test update_mod_masses.""
        custom_mods = {"stat_mods": {"57.0215": ("[UNIMOD:275]", "MOD")}}
        self.assertRaises(AssertionError, c.update_mod_masses, custom_mods)
    """
