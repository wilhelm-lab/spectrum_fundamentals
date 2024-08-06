import unittest

import spectrum_fundamentals.constants as c
import spectrum_fundamentals.mod_string as mod


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


class TestSageToInternal(unittest.TestCase):
    """Class to test Sage to internal."""

    def test_sage_to_internal_carbamidomethylation(self):
        """Test sage_to_internal_carbamidomethylation."""
        self.assertEqual(mod.sage_to_internal(["ABC[+57.0215]DEFGH"], c.MOD_MASSES_SAGE), ["ABC[UNIMOD:4]DEFGH"])

    def test_sage_to_internal_variable_oxidation(self):
        """Test sage_to_internal_variable_oxidation."""
        self.assertEqual(
            mod.sage_to_internal(["ABC[+57.0215]DM[+15.9949]EFGH"], c.MOD_MASSES_SAGE),
            ["ABC[UNIMOD:4]DM[UNIMOD:35]EFGH"],
        )

    def test_sage_to_internal_tmt(self):
        """Test sage_to_internal_tmt."""
        self.assertEqual(
            mod.sage_to_internal(["[+229.1629]-ABC[+57.0215]DEFGHK[+229.1629]"], c.MOD_MASSES_SAGE),
            ["[UNIMOD:737]-ABC[UNIMOD:4]DEFGHK[UNIMOD:737]"],
        )

    def test_sage_to_internal_custom(self):
        """Test sage_to_internal_custom."""
        # difference in rounding
        mods = {**c.MOD_MASSES_SAGE, **{"229.1628": "[UNIMOD:737]"}}
        self.assertEqual(
            mod.sage_to_internal(["[+229.1628]-ABC[+57.0215]DEFGHK[+229.1629]"], mods),
            ["[UNIMOD:737]-ABC[UNIMOD:4]DEFGHK[UNIMOD:737]"],
        )

    def test_sage_to_internal_custom_overwrite(self):
        """Test sage_to_internal_custom_overwrite."""
        mods = {**c.MOD_MASSES_SAGE, **{"57.0215": "[UNIMOD:977]"}}
        self.assertEqual(
            mod.sage_to_internal(["[+229.1629]-ABC[+57.0215]DEFGHK[+229.1629]"], mods),
            ["[UNIMOD:737]-ABC[UNIMOD:977]DEFGHK[UNIMOD:737]"],
        )


class TestXisearchToInternal(unittest.TestCase):
    """Class to test Internal to Xisearch."""

    def test_internal_to_xisearch_carbamidomethylation_oxidation_dsso(self):
        """Test internal_to_xisearch_carbamidomethylation and oxidation along with DSSO as crosslinker ."""
        self.assertEqual(
            mod.xisearch_to_internal(
                xl="DSSO", seq="SNVPALEACPQKR", mod="cm", crosslinker_position=12, mod_positions="9"
            ),
            ("SNVPALEAC[UNIMOD:4]PQK[UNIMOD:1896]R"),
        )

    def test_internal_to_xisearch_no_modification_dsso(self):
        """Test internal_to_xisearch_no_variable along with DSSO as crosslinker."""
        self.assertEqual(
            mod.xisearch_to_internal(
                xl="DSSO", seq="SNVPALEACPQKR", mod="NaN", crosslinker_position=12, mod_positions="NaN"
            ),
            ("SNVPALEACPQK[UNIMOD:1896]R"),
        )

    def test_internal_to_xisearch_carbamidomethylation_oxidation_dsbu(self):
        """Test internal_to_xisearch_carbamidomethylation and oxidation along with DSBU as crosslinker."""
        self.assertEqual(
            mod.xisearch_to_internal(
                xl="DSBU", seq="SNVPALEACPQKR", mod="cm", crosslinker_position=12, mod_positions="9"
            ),
            ("SNVPALEAC[UNIMOD:4]PQK[UNIMOD:1884]R"),
        )

    def test_internal_to_xisearch_no_modification_dsbu(self):
        """Test internal_to_xisearch_no_variable along with DSBU as crosslinker."""
        self.assertEqual(
            mod.xisearch_to_internal(
                xl="DSBU", seq="SNVPALEACPQKR", mod="NaN", crosslinker_position=12, mod_positions="NaN"
            ),
            ("SNVPALEACPQK[UNIMOD:1884]R"),
        )

    def test_internal_to_xisearch_double_modifications(self):
        """Test internal_to_xisearch_double_variable."""
        self.assertEqual(
            mod.xisearch_to_internal(xl="DSSO", seq="MKRM", mod="ox;ox", crosslinker_position=2, mod_positions="1;4"),
            ("M[UNIMOD:35]K[UNIMOD:1896]RM[UNIMOD:35]"),
        )


class TestMaxQuantToInternal(unittest.TestCase):
    """Class to test MaxQuant to internal."""

    def test_maxquant_to_internal_carbamidomethylation(self):
        """Test maxquant_to_internal_carbamidomethylation."""
        mods = {**c.MAXQUANT_VAR_MODS, **{"C": "C[UNIMOD:4]"}}
        self.assertEqual(mod.maxquant_to_internal(["_ABCDEFGH_"], mods=mods), ["ABC[UNIMOD:4]DEFGH"])

    def test_maxquant_to_internal_variable_oxidation(self):
        """Test maxquant_to_internal_variable_oxidation."""
        mods = {**c.MAXQUANT_VAR_MODS, **{"C": "C[UNIMOD:4]"}}
        self.assertEqual(mod.maxquant_to_internal(["_ABCDM(ox)EFGH_"], mods=mods), ["ABC[UNIMOD:4]DM[UNIMOD:35]EFGH"])

    def test_maxquant_to_internal_variable_oxidation_long(self):
        """Test maxquant_to_internal_variable_oxidation_long."""
        mods = {**c.MAXQUANT_VAR_MODS, **{"C": "C[UNIMOD:4]"}}
        self.assertEqual(
            mod.maxquant_to_internal(["_ABCDM(Oxidation (M))EFGH_"], mods=mods), ["ABC[UNIMOD:4]DM[UNIMOD:35]EFGH"]
        )

    def test_maxquant_to_internal_variable_dehydration_long(self):
        """Test maxquant_to_internal_variable_dehydration_long."""
        mods = {**c.MAXQUANT_VAR_MODS, **{"C": "C[UNIMOD:4]"}}
        self.assertEqual(
            mod.maxquant_to_internal(["_ABCDS(Dehydrated (ST))EFGH_"], mods=mods), ["ABC[UNIMOD:4]DS[UNIMOD:23]EFGH"]
        )

    def test_maxquant_to_internal_tmt(self):
        """Test maxquant_to_internal_tmt."""
        fixed_mods = {"C": "C[UNIMOD:4]", "^_": "_[UNIMOD:737]-", "K": "K[UNIMOD:737]"}
        mods = {**c.MAXQUANT_VAR_MODS, **fixed_mods}
        self.assertEqual(
            mod.maxquant_to_internal(["_ABCDEFGHK_"], mods=mods), ["[UNIMOD:737]-ABC[UNIMOD:4]DEFGHK[UNIMOD:737]"]
        )

    def test_maxquant_to_internal_custom_stat_overwrite(self):
        # overwrite fixed_mod with stat mod
        """Test maxquant_to_internal_custom_stat_overwrite."""
        fixed_mods = {"C": "C[UNIMOD:4]", "^_": "_[UNIMOD:737]-", "K": "K[UNIMOD:737]"}
        stat_mod = {"K": "[UNIMOD:738]"}
        mods = {**c.MAXQUANT_VAR_MODS, **fixed_mods, **stat_mod}
        self.assertEqual(
            mod.maxquant_to_internal(["_ABCDEFGHK_"], mods=mods), ["[UNIMOD:737]-ABC[UNIMOD:4]DEFGHK[UNIMOD:738]"]
        )

    def test_maxquant_to_internal_custom_var(self):
        # overwrite maxquant_var_mod
        """Test maxquant_to_internal_custom_var."""
        fixed_mods = {"C": "C[UNIMOD:4]"}
        var_mod = {"(ox)": "[UNIMOD:425]"}
        mods = {**c.MAXQUANT_VAR_MODS, **fixed_mods, **var_mod}
        self.assertEqual(mod.maxquant_to_internal(["_ABCDM(ox)EFGH_"], mods=mods), ["ABC[UNIMOD:4]DM[UNIMOD:425]EFGH"])

    def test_maxquant_to_internal_custom_stat(self):
        # overwrite maxquant_var_mod
        """Test maxquant_to_internal_custom_var."""
        fixed_mods = {"C": "C[UNIMOD:4]"}
        stat_mod = {"M": "[UNIMOD:425]"}
        mods = {**c.MAXQUANT_VAR_MODS, **fixed_mods, **stat_mod}
        self.assertEqual(mod.maxquant_to_internal(["_ABCDMEFGH_"], mods=mods), ["ABC[UNIMOD:4]DM[UNIMOD:425]EFGH"])


class TestMSFraggerToInternal(unittest.TestCase):
    """Class to test MSFragger to internal."""

    def test_msfragger_to_internal_carbamidomethylation(self):
        """Test msfragger_to_internal_carbamidomethylation."""
        mods = {**c.MSFRAGGER_VAR_MODS, **{"C": "C[UNIMOD:4]"}}
        self.assertEqual(mod.msfragger_to_internal(["ABCDEFGH"], mods), ["ABC[UNIMOD:4]DEFGH"])

    def test_msfragger_to_internal_variable_oxidation(self):
        """Test msfragger_to_internal_variable_oxidation."""
        mods = {**c.MSFRAGGER_VAR_MODS, **{"C": "C[UNIMOD:4]"}}
        self.assertEqual(mod.msfragger_to_internal(["ABCDM[147]EFGH"], mods), ["ABC[UNIMOD:4]DM[UNIMOD:35]EFGH"])

    def test_msfragger_to_internal_tmt(self):
        """Test msfragger_to_internal_tmt."""
        fixed_mods = {"C": "C[UNIMOD:4]", r"n[\d+]": "[UNIMOD:737]-", "K": "K[UNIMOD:737]"}
        self.assertEqual(
            mod.msfragger_to_internal(["n[230]ABCDEFGHK"], {**c.MSFRAGGER_VAR_MODS, **fixed_mods}),
            ["[UNIMOD:737]-ABC[UNIMOD:4]DEFGHK[UNIMOD:737]"],
        )

    def test_msfragger_to_internal_ms_custom_mod(self):
        """Test msfragger_to_internal_ms_custom_mod."""
        fixed_mods = {"C": "C[UNIMOD:4]"}
        custom_mod = {"M[35]": "[UNIMOD:35]"}
        mods = {**c.MSFRAGGER_VAR_MODS, **fixed_mods, **custom_mod}
        self.assertEqual(
            mod.msfragger_to_internal(["n[230]ABCDEFGHM[35]"], mods), ["[UNIMOD:737]-ABC[UNIMOD:4]DEFGHM[UNIMOD:35]"]
        )

    def test_msfragger_to_internal_custom_mod(self):
        """Test msfragger_to_internal_custom_mod."""
        custom_mod = {"M[35]": "[UNIMOD:35]", "(cm)": "[UNIMOD:4]"}
        mods = {**c.MSFRAGGER_VAR_MODS, **custom_mod}
        self.assertEqual(mod.msfragger_to_internal(["ABC(cm)DEFGHM[35]"], mods), ["ABC[UNIMOD:4]DEFGHM[UNIMOD:35]"])


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
    """Class to test internal to interal without mods."""

    def test_internal_without_mods(self):
        """Test internal with mods to internal without_mods."""
        self.assertEqual(mod.internal_without_mods(["[UNIMOD:737]-ABC[UNIMOD:4]DEFGHK[UNIMOD:737]"]), ["ABCDEFGHK"])

    def test_internal_to_mod_masses(self):
        """Test internal with mods to internal without_mods."""
        self.assertEqual(
            mod.internal_to_mod_mass(["[UNIMOD:737]-ABC[UNIMOD:4]DEFGHK[UNIMOD:737]"]),
            ["[+229.162932]-ABC[+57.021464]DEFGHK[+229.162932]"],
        )

    def test_internal_to_mod_masses_custom(self):
        """Test internal with custom mods to internal without_mods."""
        mods = {"[UNIMOD:977]": 57.0215}
        self.assertEqual(
            mod.internal_to_mod_mass(["[UNIMOD:737]-ABC[UNIMOD:977]DEFGHK[UNIMOD:737]"], mods),
            ["[+229.162932]-ABC[+57.0215]DEFGHK[+229.162932]"],
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
        self.assertEqual(next(mod.parse_modstrings(["".join(valid_seq)], alphabet=c.ALPHABET)), valid_seq)

    def test_parse_modstrings_with_translation(self):
        """Test parse_modstrings with only valid elements."""
        valid_seq = ["A", "C[UNIMOD:4]", "C[UNIMOD:4]", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N"]
        valid_seq.extend(["M[UNIMOD:35]", "P", "Q", "R", "S", "T", "V", "W", "Y", "M[UNIMOD:35]", "S", "T", "Y"])
        values = [c.ALPHABET[elem] for elem in valid_seq]
        self.assertEqual(next(mod.parse_modstrings(["".join(valid_seq)], alphabet=c.ALPHABET, translate=True)), values)

    def test_parse_modstrings_invalid(self):
        """Test correct behaviour of  parse_modstrings when invalid sequence is encountered."""
        invalid_seq = "SEQUENCE"
        generator_yielding_invalid = mod.parse_modstrings([invalid_seq], alphabet=c.ALPHABET)
        self.assertRaises(ValueError, list, generator_yielding_invalid)

    def test_parse_modstrings_invalid_with_filtering(self):
        """Test correct behaviour of parse_modstrings when invalid sequence is handled."""
        invalid_seq = "testing"
        self.assertEqual(next(mod.parse_modstrings([invalid_seq], alphabet=c.ALPHABET, filter=True)), [0])


class TestCustomToInternal(unittest.TestCase):
    """Class to test custom to internal."""

    def test_custom_to_internal_carbamidomethylation(self):
        """Test custom_to_internal_carbamidomethylation."""
        mods = {"C": "C[UNIMOD:4]"}
        self.assertEqual(mod.custom_to_internal(["ABCDEFGH"], mods), ["ABC[UNIMOD:4]DEFGH"])

    def test_custom_to_internal_custom_mods(self):
        """Test custom_to_internal_custom_mods."""
        fixed_mods = {"C": "C[UNIMOD:4]"}
        custom_mod = {"M[35]": "[UNIMOD:35]"}
        mods = {**fixed_mods, **custom_mod}
        self.assertEqual(mod.custom_to_internal(["ABCDEFGHM[35]"], mods), ["ABC[UNIMOD:4]DEFGHM[UNIMOD:35]"])
