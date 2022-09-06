import pytest

import fundamentals.mod_string as mod
import fundamentals.constants as constants

class TestMSP:
    def test_internal_to_mod_names(self):
        assert mod.internal_to_mod_names(['AAAC[UNIMOD:4]CC[UNIMOD:4]CKR', 'AAACILKKR']) == [('2/3,C,Carbamidomethyl/5,C,Carbamidomethyl', 'AAACCCCKR//Carbamidomethyl@C3; Carbamidomethyl@C5'), ('0', 'AAACILKKR//')]


class TestMaxQuantToInternal:
  def test_maxquant_to_internal_carbamidomethylation(self):
    assert mod.maxquant_to_internal(["_ABCDEFGH_"]) == ["ABC[UNIMOD:4]DEFGH"]
    
  def test_maxquant_to_internal_variable_oxidation(self):
    assert mod.maxquant_to_internal(["_ABCDM(ox)EFGH_"]) == ["ABC[UNIMOD:4]DM[UNIMOD:35]EFGH"]
  
  def test_maxquant_to_internal_variable_oxidation_long(self):
    assert mod.maxquant_to_internal(["_ABCDM(Oxidation (M))EFGH_"]) == ["ABC[UNIMOD:4]DM[UNIMOD:35]EFGH"]
 
  def test_maxquant_to_internal_variable_dehydration_long(self):
    assert mod.maxquant_to_internal(["_ABCDS(Dehydrated (ST))EFGH_"]) == ["ABC[UNIMOD:4]DS[UNIMOD:23]EFGH"]

  def test_maxquant_to_internal_acytelation_on_terminus(self):
    assert mod.maxquant_to_internal(["_(ac)ABCDEFGH_"]) == ["[UNIMOD:1]-ABC[UNIMOD:4]DEFGH"]
  
  def test_maxquant_to_internal_empty_termini(self):
    assert mod.maxquant_to_internal(["_ABCDS(Dehydrated (ST))EFGH_"], empty_termini=True) == ["[]-ABC[UNIMOD:4]DS[UNIMOD:23]EFGH-[]"]

  def test_maxquant_to_internal_tmt(self):
    fixed_mods = {'C': 'C[UNIMOD:4]',
                  '^_':'_[UNIMOD:737]', 
                  'K': 'K[UNIMOD:737]'}
    assert mod.maxquant_to_internal(["_ABCDEFGHK_"], fixed_mods) == ["[UNIMOD:737]ABC[UNIMOD:4]DEFGHK[UNIMOD:737]"]
  
  def test_maxquant_to_internal_silac(self):
    fixed_mods = {'C': 'C[UNIMOD:4]',
                  'K': 'K[UNIMOD:259]', 
                  'R': 'R[UNIMOD:267]'}
    assert mod.maxquant_to_internal(["_ABCDEFGHRK_"], fixed_mods) == ["ABC[UNIMOD:4]DEFGHR[UNIMOD:267]K[UNIMOD:259]"]
  
