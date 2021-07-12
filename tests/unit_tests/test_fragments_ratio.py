import pytest
import numpy as np

import fundamentals.metrics.fragments_ratio as fr

class TestSeen:
    def test_get_mask_observed_invalid(self):
        assert fr.get_mask_observed_invalid([10.2, 0, -1, np.nan]) == [False, False, True, True]
    
    
    def test_make_boolean(self):
        assert fr.make_boolean([10.2, 0, -1, np.nan], [False, False, True, True]) == [True, False, np.nan, np.nan]
        
    
    def test_get_observation_state(self):
        observed_boolean = [False, False, True, True, np.nan]
        predicted_boolean = [False, True, False, True, np.nan]
        assert fr.get_observation_state(observed_boolean, predicted_boolean) == [2, 1, -1, 0, np.nan]
    
    
    
        
