import pytest
import numpy as np

import fundamentals.metrics.fragments_ratio as fr


class TestObservationState:

    def test_get_mask_observed_invalid(self):
        observed_mz = np.array([10.2, 0, -1, np.nan])
        np.testing.assert_equal(fr.FragmentsRatio.get_mask_observed_invalid(observed_mz), np.array([False, False, True, True]))
    
    def test_make_boolean(self):
        observed = np.array([10.2, 0, -1, np.nan])
        mask = np.array([False, False, True, True])
        np.testing.assert_equal(fr.FragmentsRatio.make_boolean(observed, mask), np.array([True, False, False, False]))
    
    def test_make_boolean_cutoff(self):
        predicted = np.array([10.2, 0, -1, 0.02, np.nan])
        mask = np.array([False, False, True, False, True])
        np.testing.assert_equal(fr.FragmentsRatio.make_boolean(predicted, mask, cutoff = 0.05), np.array([True, False, False, False, False]))
        
    def test_get_observation_state(self):
        observed_boolean = np.array([False, False, True, True])
        predicted_boolean = np.array([False, True, False, True])
        np.testing.assert_equal(fr.FragmentsRatio.get_observation_state(observed_boolean, predicted_boolean), np.array([2, 1, -1, 0]))
    

