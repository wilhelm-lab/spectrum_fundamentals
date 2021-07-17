import pytest

import fundamentals.mod_string as mod
import fundamentals.constants as constants

class TestMSP:
    def test_internal_to_mod_names(self):
        assert mod.internal_to_mod_names(['AAAC(U:4)CC(U:4)CKR', 'AAACILKKR']) == [('2/3,C,Carbamidomethyl/5,C,Carbamidomethyl', 'AAACCCCKR//Carbamidomethyl@C3; Carbamidomethyl@C5'), ('0', 'AAACILKKR//')]
