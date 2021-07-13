import pytest
import numpy as np

import fundamentals.metrics.fragments_ratio as fr
import fundamentals.constants as constants

class TestObservationState:

    def test_get_mask_observed_invalid(self):
        observed_mz = get_padded_array([10.2, 0, -1, np.nan])
        np.testing.assert_equal(fr.FragmentsRatio.get_mask_observed_invalid(observed_mz), get_padded_array([False, False, True, True]))
    
    def test_make_boolean(self):
        observed = get_padded_array([10.2, 0, -1, np.nan])
        mask = get_padded_array([False, False, True, True])
        np.testing.assert_equal(fr.FragmentsRatio.make_boolean(observed, mask), get_padded_array([True, False, False, False]))
    
    def test_make_boolean_cutoff_below(self):
        predicted = get_padded_array([10.2, 0, -1, 0.02, np.nan])
        mask = get_padded_array([False, False, True, False, True])
        np.testing.assert_equal(fr.FragmentsRatio.make_boolean(predicted, mask, cutoff = 0.05), get_padded_array([True, False, False, False, False]))
    
    def test_make_boolean_cutoff_above(self):
        predicted = get_padded_array([10.2, 0, -1, 0.02, np.nan])
        mask = get_padded_array([False, False, True, False, True])
        np.testing.assert_equal(fr.FragmentsRatio.make_boolean(predicted, mask, cutoff = 0.01), get_padded_array([True, False, False, True, False]))
        
    def test_get_observation_state(self):
        observed_boolean = get_padded_array([False, False, True, True])
        predicted_boolean = get_padded_array([False, True, False, True])
        mask = get_padded_array([True, True, True, True])
        np.testing.assert_equal(fr.FragmentsRatio.get_observation_state(observed_boolean, predicted_boolean, mask), get_padded_array([fr.ObservationState.NOT_OBS_AND_NOT_PRED, 
                                              fr.ObservationState.NOT_OBS_BUT_PRED, 
                                              fr.ObservationState.OBS_BUT_NOT_PRED,
                                              fr.ObservationState.OBS_AND_PRED],
                                              fr.ObservationState.INVALID_ION))
    

class TestCountPredicted:
    
    def test_count_predicted(self):
        predicted = get_padded_array([10.2, 0, -1, 0.02, 0])
        np.testing.assert_equal(fr.FragmentsRatio.count_predicted(predicted, cutoff = 0.05), 1)
    
    def test_count_predicted_b(self):
        predicted = get_padded_array([10.2, 0, -1, 0.02, 0])
        np.testing.assert_equal(fr.FragmentsRatio.count_predicted_b(predicted, cutoff = 0.05), 0)
    
    def test_count_predicted_y(self):
        predicted = get_padded_array([10.2, 0, -1, 0.02, 0])
        np.testing.assert_equal(fr.FragmentsRatio.count_predicted_y(predicted, cutoff = 0.05), 1)

class TestCountObservedAndPredicted:

    def test_count_observed_and_predicted(self):
        observation_state = get_padded_array([fr.ObservationState.NOT_OBS_AND_NOT_PRED, 
                                              fr.ObservationState.NOT_OBS_BUT_PRED, 
                                              fr.ObservationState.OBS_BUT_NOT_PRED,
                                              fr.ObservationState.OBS_AND_PRED],
                                              fr.ObservationState.INVALID_ION)
        np.testing.assert_equal(fr.FragmentsRatio.count_observed_and_predicted(observation_state), 1)
    
    def test_count_observed_and_predicted_b(self):
        observation_state = get_padded_array([fr.ObservationState.NOT_OBS_AND_NOT_PRED, 
                                              fr.ObservationState.NOT_OBS_BUT_PRED, 
                                              fr.ObservationState.OBS_BUT_NOT_PRED,
                                              fr.ObservationState.OBS_AND_PRED],
                                              fr.ObservationState.INVALID_ION)
        np.testing.assert_equal(fr.FragmentsRatio.count_observed_and_predicted_b(observation_state), 1)
    
    def test_count_observed_and_predicted_y(self):
        observation_state = get_padded_array([fr.ObservationState.NOT_OBS_AND_NOT_PRED, 
                                              fr.ObservationState.NOT_OBS_BUT_PRED, 
                                              fr.ObservationState.OBS_BUT_NOT_PRED,
                                              fr.ObservationState.OBS_AND_PRED],
                                              fr.ObservationState.INVALID_ION)
        np.testing.assert_equal(fr.FragmentsRatio.count_observed_and_predicted_y(observation_state), 0)

class TestCountNotObservedAndNotPredicted:

    def test_count_not_observed_and_not_predicted(self):
        observation_state = get_padded_array([fr.ObservationState.NOT_OBS_AND_NOT_PRED, 
                                              fr.ObservationState.NOT_OBS_BUT_PRED, 
                                              fr.ObservationState.OBS_BUT_NOT_PRED,
                                              fr.ObservationState.OBS_AND_PRED,
                                              fr.ObservationState.NOT_OBS_AND_NOT_PRED],
                                              fr.ObservationState.INVALID_ION)
        np.testing.assert_equal(fr.FragmentsRatio.count_not_observed_and_not_predicted(observation_state), 2)
    
    def test_count_not_observed_and_not_predicted_b(self):
        observation_state = get_padded_array([fr.ObservationState.NOT_OBS_AND_NOT_PRED, 
                                              fr.ObservationState.NOT_OBS_BUT_PRED, 
                                              fr.ObservationState.OBS_BUT_NOT_PRED,
                                              fr.ObservationState.OBS_AND_PRED,
                                              fr.ObservationState.NOT_OBS_AND_NOT_PRED],
                                              fr.ObservationState.INVALID_ION)
        np.testing.assert_equal(fr.FragmentsRatio.count_not_observed_and_not_predicted(observation_state), 2)
    
    def test_count_not_observed_and_not_predicted_y(self):
        observation_state = get_padded_array([fr.ObservationState.NOT_OBS_AND_NOT_PRED, 
                                              fr.ObservationState.NOT_OBS_BUT_PRED, 
                                              fr.ObservationState.OBS_BUT_NOT_PRED,
                                              fr.ObservationState.OBS_AND_PRED,
                                              fr.ObservationState.NOT_OBS_AND_NOT_PRED],
                                              fr.ObservationState.INVALID_ION)
        np.testing.assert_equal(fr.FragmentsRatio.count_not_observed_and_not_predicted(observation_state), 2)

class TestCountNotObservedButPredicted:

    def test_count_not_observed_but_predicted(self):
        observation_state = get_padded_array([fr.ObservationState.NOT_OBS_AND_NOT_PRED, 
                                              fr.ObservationState.NOT_OBS_BUT_PRED, 
                                              fr.ObservationState.OBS_BUT_NOT_PRED,
                                              fr.ObservationState.OBS_AND_PRED,
                                              fr.ObservationState.NOT_OBS_BUT_PRED,
                                              fr.ObservationState.NOT_OBS_BUT_PRED],
                                              fr.ObservationState.INVALID_ION)
        np.testing.assert_equal(fr.FragmentsRatio.count_not_observed_but_predicted(observation_state), 3)
        
    def test_count_not_observed_but_predicted_b(self):
        observation_state = get_padded_array([fr.ObservationState.NOT_OBS_AND_NOT_PRED, 
                                              fr.ObservationState.NOT_OBS_BUT_PRED, 
                                              fr.ObservationState.OBS_BUT_NOT_PRED,
                                              fr.ObservationState.OBS_AND_PRED,
                                              fr.ObservationState.NOT_OBS_BUT_PRED,
                                              fr.ObservationState.NOT_OBS_BUT_PRED],
                                              fr.ObservationState.INVALID_ION)
        np.testing.assert_equal(fr.FragmentsRatio.count_not_observed_but_predicted_b(observation_state), 2)

    def test_count_not_observed_but_predicted_y(self):
        observation_state = get_padded_array([fr.ObservationState.NOT_OBS_AND_NOT_PRED, 
                                              fr.ObservationState.NOT_OBS_BUT_PRED, 
                                              fr.ObservationState.OBS_BUT_NOT_PRED,
                                              fr.ObservationState.OBS_AND_PRED,
                                              fr.ObservationState.NOT_OBS_BUT_PRED,
                                              fr.ObservationState.NOT_OBS_BUT_PRED],
                                              fr.ObservationState.INVALID_ION)
        np.testing.assert_equal(fr.FragmentsRatio.count_not_observed_but_predicted_y(observation_state), 1)

class TestCountObservedButNotPredicted:

    def test_count_observed_but_not_predicted(self):
        observation_state = get_padded_array([fr.ObservationState.NOT_OBS_AND_NOT_PRED, 
                                              fr.ObservationState.NOT_OBS_BUT_PRED, 
                                              fr.ObservationState.OBS_BUT_NOT_PRED,
                                              fr.ObservationState.OBS_AND_PRED,
                                              fr.ObservationState.OBS_BUT_NOT_PRED,
                                              fr.ObservationState.OBS_BUT_NOT_PRED,
                                              fr.ObservationState.OBS_BUT_NOT_PRED],
                                              fr.ObservationState.INVALID_ION)
        np.testing.assert_equal(fr.FragmentsRatio.count_observed_but_not_predicted(observation_state), 4)
    
    def test_count_observed_but_not_predicted_b(self):
        observation_state = get_padded_array([fr.ObservationState.NOT_OBS_AND_NOT_PRED, 
                                              fr.ObservationState.NOT_OBS_BUT_PRED, 
                                              fr.ObservationState.OBS_BUT_NOT_PRED,
                                              fr.ObservationState.OBS_AND_PRED,
                                              fr.ObservationState.OBS_BUT_NOT_PRED,
                                              fr.ObservationState.OBS_BUT_NOT_PRED,
                                              fr.ObservationState.OBS_BUT_NOT_PRED],
                                              fr.ObservationState.INVALID_ION)
        np.testing.assert_equal(fr.FragmentsRatio.count_observed_but_not_predicted_b(observation_state), 2)

    def test_count_observed_but_not_predicted_y(self):
        observation_state = get_padded_array([fr.ObservationState.NOT_OBS_AND_NOT_PRED, 
                                              fr.ObservationState.NOT_OBS_BUT_PRED, 
                                              fr.ObservationState.OBS_BUT_NOT_PRED,
                                              fr.ObservationState.OBS_AND_PRED,
                                              fr.ObservationState.OBS_BUT_NOT_PRED,
                                              fr.ObservationState.OBS_BUT_NOT_PRED,
                                              fr.ObservationState.OBS_BUT_NOT_PRED],
                                              fr.ObservationState.INVALID_ION)
        np.testing.assert_equal(fr.FragmentsRatio.count_observed_but_not_predicted_y(observation_state), 2)

def get_padded_array(l, padding_value = 0):
    return np.pad(l, (0, constants.NUM_IONS - len(l)), 'constant', constant_values=padding_value)
