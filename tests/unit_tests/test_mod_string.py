import unittest

import spectrum_fundamentals.mod_string as mod
from spectrum_fundamentals.constants import ALPHABET


class TestMSP:
    """Class to test msp."""

    def test_internal_to_mod_names(self):
        """Test internal_to_mod_names."""
        assert mod.internal_to_mod_names(["AAAC[UNIMOD:4]CC[UNIMOD:4]CKR", "AAACILKKR"]) == [
            (
                "2/3,C,Carbamidomethyl/5,C,Carbamidomethyl",
                "AAACCCCKR//Carbamidomethyl@C3; Carbamidomethyl@C5",
            ),
            ("0", "AAACILKKR//"),
        ]


class TestMaxQuantToInternal(unittest.TestCase):
    """Class to test MaxQuant to internal."""

    def test_maxquant_to_internal_carbamidomethylation(self):
        """Test maxquant_to_internal_carbamidomethylation."""
        self.assertEqual(mod.maxquant_to_internal(["_ABCDEFGH_"]), ["ABC[UNIMOD:4]DEFGH"])

    def test_maxquant_to_internal_variable_oxidation(self):
        """Test maxquant_to_internal_variable_oxidation."""
        self.assertEqual(mod.maxquant_to_internal(["_ABCDM(ox)EFGH_"]), ["ABC[UNIMOD:4]DM[UNIMOD:35]EFGH"])

    def test_maxquant_to_internal_variable_oxidation_long(self):
        """Test maxquant_to_internal_variable_oxidation_long."""
        self.assertEqual(mod.maxquant_to_internal(["_ABCDM(Oxidation (M))EFGH_"]), ["ABC[UNIMOD:4]DM[UNIMOD:35]EFGH"])

    def test_maxquant_to_internal_variable_dehydration_long(self):
        """Test maxquant_to_internal_variable_dehydration_long."""
        self.assertEqual(mod.maxquant_to_internal(["_ABCDS(Dehydrated (ST))EFGH_"]), ["ABC[UNIMOD:4]DS[UNIMOD:23]EFGH"])

    def test_maxquant_to_internal_tmt(self):
        """Test maxquant_to_internal_tmt."""
        fixed_mods = {"C": "C[UNIMOD:4]", "^_": "_[UNIMOD:737]-", "K": "K[UNIMOD:737]"}
        self.assertEqual(
            mod.maxquant_to_internal(["_ABCDEFGHK_"], fixed_mods), ["[UNIMOD:737]-ABC[UNIMOD:4]DEFGHK[UNIMOD:737]"]
        )

    def test_maxquant_to_internal_silac(self):
        """Test maxquant_to_internal_silac."""
        fixed_mods = {"C": "C[UNIMOD:4]", "K": "K[UNIMOD:259]", "R": "R[UNIMOD:267]"}
        self.assertEqual(
            mod.maxquant_to_internal(["_ABCDEFGHRK_"], fixed_mods), ["ABC[UNIMOD:4]DEFGHR[UNIMOD:267]K[UNIMOD:259]"]
        )


class TestInternalToSpectronaut(unittest.TestCase):
    """Class to test Internal to Spectronaut."""

    def test_internal_to_spectronaut_carbamidomethylation(self):
        """Test internal_to_spectronaut_carbamidomethylation."""
        self.assertEqual(mod.internal_to_spectronaut(["ABC[UNIMOD:4]DEFGH"]), ["ABC[Carbamidomethyl (C)]DEFGH"])

    def test_internal_to_spectronaut_variable_oxidation(self):
        """Test internal_to_spectronaut_carbamidomethylation."""
        self.assertEqual(
            mod.internal_to_spectronaut(["ABC[UNIMOD:4]DM[UNIMOD:35]EFGH"]),
            ["ABC[Carbamidomethyl (C)]DM[Oxidation (O)]EFGH"],
        )

    def test_internal_to_spectronaut_variable_dehydration_long(self):
        """Test internal_to_spectronaut_variable_dehydration_long."""
        self.assertEqual(
            mod.internal_to_spectronaut(["ABC[UNIMOD:4]DS[UNIMOD:23]EFGH"]),
            ["ABC[Carbamidomethyl (C)]DS[UNIMOD:23]EFGH"],
        )


class TestInternalTransformations(unittest.TestCase):
    """Class to test internal to interla without mods."""

    def test_internal_without_mods(self):
        """Test internal with mods to internal without_mods."""
        self.assertEqual(mod.internal_without_mods(["[UNIMOD:737]ABC[UNIMOD:4]DEFGHK[UNIMOD:737]"]), ["ABCDEFGHK"])

    def test_internal_to_mod_masses(self):
        """Test internal with mods to internal without_mods."""
        self.assertEqual(
            mod.internal_to_mod_mass(["[UNIMOD:737]ABC[UNIMOD:4]DEFGHK[UNIMOD:737]"]),
            ["[+229.162932]ABC[+57.02146]DEFGHK[+229.162932]"],
        )

    def test_proteomicsdb_to_internal(self):
        """Test proteomicsdb sequence to internal sequence with fixed and variable modifications."""
        prdb_sequence = "AAMCGHK"
        variable_mods = (
            "Oxidation@M3;Acetylation@^0"  # TODO check if n-terminal acetylation is really written like that
        )
        fixed_mods = "Carbamidomethyl@C"

        target_sequence = "[UNIMOD:1]-AAM[UNIMOD:35]C[UNIMOD:4]GHK-[]"

        self.assertEqual(mod.proteomicsdb_to_internal(prdb_sequence, variable_mods, fixed_mods), target_sequence)

    def test_proteomicsdb_to_internal_without_mods(self):
        """Test proteomicsdb sequence to internal sequence without any modifications."""
        prdb_sequence = "AAMCGHK"
        target_sequence = "[]-" + prdb_sequence + "-[]"
        self.assertEqual(mod.proteomicsdb_to_internal(prdb_sequence), target_sequence)

    def test_proteomicsdb_to_internal_variable_only(self):
        """Test proteomicsdb sequence to internal sequence with just variable modifications."""
        prdb_sequence = "AAMCGHK"
        variable_mods = "Oxidation@M3;Oxidation@M45"  # having outside of range on purpose here (pos 45)
        target_sequence = "[]-AAM[UNIMOD:35]CGHK-[]"

        self.assertEqual(mod.proteomicsdb_to_internal(prdb_sequence, mods_variable=variable_mods), target_sequence)

    def test_proteomicsdb_to_internal_fixed_only(self):
        """Test proteomicsdb sequence to internal sequence with just fixed modifications."""
        prdb_sequence = "AAMCGHK"
        fixed_mods = "Carbamidomethyl@C"
        target_sequence = "[]-AAMC[UNIMOD:4]GHK-[]"

        self.assertEqual(mod.proteomicsdb_to_internal(prdb_sequence, mods_fixed=fixed_mods), target_sequence)


class TestParsing(unittest.TestCase):
    """Class to test the modstring parsing."""

    def test_parse_modstrings(self):
        """Test parse_modstrings with only valid elements."""
        valid_seq = ["A", "C[UNIMOD:4]", "C[UNIMOD:4]", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N"]
        valid_seq.extend(["M[UNIMOD:35]", "P", "Q", "R", "S", "T", "V", "W", "Y", "M[UNIMOD:35]", "S", "T", "Y"])
        self.assertEqual(next(mod.parse_modstrings(["".join(valid_seq)], alphabet=ALPHABET)), valid_seq)

    def test_parse_modstrings_with_translation(self):
        """Test parse_modstrings with only valid elements."""
        valid_seq = ["A", "C[UNIMOD:4]", "C[UNIMOD:4]", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N"]
        valid_seq.extend(["M[UNIMOD:35]", "P", "Q", "R", "S", "T", "V", "W", "Y", "M[UNIMOD:35]", "S", "T", "Y"])
        values = [ALPHABET[elem] for elem in valid_seq]
        self.assertEqual(next(mod.parse_modstrings(["".join(valid_seq)], alphabet=ALPHABET, translate=True)), values)

    def test_parse_modstrings_invalid(self):
        """Test correct behaviour of  parse_modstrings when invalid sequence is encountered."""
        invalid_seq = "SEQUENCE"
        generator_yielding_invalid = mod.parse_modstrings([invalid_seq], alphabet=ALPHABET)
        self.assertRaises(ValueError, list, generator_yielding_invalid)

    def test_parse_modstrings_invalid_with_filtering(self):
        """Test correct behaviour of parse_modstrings when invalid sequence is handled."""
        invalid_seq = "testing"
        self.assertEqual(next(mod.parse_modstrings([invalid_seq], alphabet=ALPHABET, filter=True)), [0])
