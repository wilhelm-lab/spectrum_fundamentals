import pytest
import numpy as np

import fundamentals.metrics.fragments_ratio as fr
import fundamentals.constants as constants

class TestObservationState:

    def test_get_mask_observed_valid(self):
        observed_mz = get_padded_array([10.2, constants.EPSILON, 0, np.nan])
        np.testing.assert_equal(fr.FragmentsRatio.get_mask_observed_valid(observed_mz), get_padded_array([True, True, False, False]))
    
    def test_make_boolean(self):
        observed = get_padded_array([10.2, constants.EPSILON, 0, np.nan])
        mask = get_padded_array([True, True, False, False])
        np.testing.assert_equal(fr.FragmentsRatio.make_boolean(observed, mask), get_padded_array([True, False, False, False]))
    
    def test_make_boolean_cutoff_below(self):
        predicted = get_padded_array([10.2, constants.EPSILON, 0, 0.02, np.nan])
        mask = get_padded_array([True, True, False, True, False])
        np.testing.assert_equal(fr.FragmentsRatio.make_boolean(predicted, mask, cutoff = 0.05), get_padded_array([True, False, False, False, False]))
    
    def test_make_boolean_cutoff_above(self):
        predicted = get_padded_array([10.2, constants.EPSILON, 0, 0.02, np.nan])
        mask = get_padded_array([True, True, False, True, False])
        np.testing.assert_equal(fr.FragmentsRatio.make_boolean(predicted, mask, cutoff = 0.01), get_padded_array([True, False, False, True, False]))
        
    def test_get_observation_state(self):
        observed_boolean = get_padded_array([False, False, True, True])
        predicted_boolean = get_padded_array([False, True, False, True])
        mask = get_padded_array([True, True, True, True])
        np.testing.assert_equal(fr.FragmentsRatio.get_observation_state(observed_boolean, predicted_boolean, mask), 
                                get_padded_array([fr.ObservationState.NOT_OBS_AND_NOT_PRED, 
                                                  fr.ObservationState.NOT_OBS_BUT_PRED, 
                                                  fr.ObservationState.OBS_BUT_NOT_PRED,
                                                  fr.ObservationState.OBS_AND_PRED],
                                                  fr.ObservationState.INVALID_ION))
    

class TestCountIons:
    
    def test_count_ions(self):
        predicted_boolean = get_padded_array([True, False, False, False, False])
        np.testing.assert_equal(fr.FragmentsRatio.count_ions(predicted_boolean), 1)
    
    def test_count_ions_b(self):
        predicted_boolean = get_padded_array([True, False, False, False, False])
        np.testing.assert_equal(fr.FragmentsRatio.count_ions_b(predicted_boolean), 0)
    
    def test_count_ions_y(self):
        predicted_boolean = get_padded_array([True, False, False, False, False])
        np.testing.assert_equal(fr.FragmentsRatio.count_ions_y(predicted_boolean), 1)


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


class TestCalc:
    
    def test_calc(self):
        z = constants.EPSILON
        #                                         y1.1  y1.2  y1.3  b1.1  b1.2  b1.3  y2.1  y2.2  y2.3
        predicted_intensities = get_padded_array([ 7.2,  2.3, 0.01, 0.02,  6.1,  3.1,    z,    z,    0])
        observed_intensities =  get_padded_array([10.2,    z,  1.3,    z,  8.2,    z,  3.2,    z,    0])
        fragmentsRatio = fr.FragmentsRatio(predicted_intensities, observed_intensities)
        fragmentsRatio.calc()
        
        np.testing.assert_equal(fragmentsRatio.metrics_val['count_predicted'][0], 4)
        np.testing.assert_equal(fragmentsRatio.metrics_val['count_predicted_b'][0], 2)
        np.testing.assert_equal(fragmentsRatio.metrics_val['count_predicted_y'][0], 2)
        
        np.testing.assert_equal(fragmentsRatio.metrics_val['count_observed'][0], 4)
        np.testing.assert_equal(fragmentsRatio.metrics_val['count_observed_b'][0], 1)
        np.testing.assert_equal(fragmentsRatio.metrics_val['count_observed_y'][0], 3)
        
        np.testing.assert_equal(fragmentsRatio.metrics_val['count_observed_and_predicted'][0], 2)
        np.testing.assert_equal(fragmentsRatio.metrics_val['count_observed_and_predicted_b'][0], 1)
        np.testing.assert_equal(fragmentsRatio.metrics_val['count_observed_and_predicted_y'][0], 1)
        
        np.testing.assert_equal(fragmentsRatio.metrics_val['count_not_observed_and_not_predicted'][0], 2)
        np.testing.assert_equal(fragmentsRatio.metrics_val['count_not_observed_and_not_predicted_b'][0], 1)
        np.testing.assert_equal(fragmentsRatio.metrics_val['count_not_observed_and_not_predicted_y'][0], 1)
        
        np.testing.assert_equal(fragmentsRatio.metrics_val['count_observed_but_not_predicted'][0], 2)
        np.testing.assert_equal(fragmentsRatio.metrics_val['count_observed_but_not_predicted_b'][0], 0)
        np.testing.assert_equal(fragmentsRatio.metrics_val['count_observed_but_not_predicted_y'][0], 2)
        
        np.testing.assert_equal(fragmentsRatio.metrics_val['count_not_observed_but_predicted'][0], 2)
        np.testing.assert_equal(fragmentsRatio.metrics_val['count_not_observed_but_predicted_b'][0], 1)
        np.testing.assert_equal(fragmentsRatio.metrics_val['count_not_observed_but_predicted_y'][0], 1)
        
        
def get_padded_array(l, padding_value = 0):
    return np.array([np.pad(l, (0, constants.NUM_IONS - len(l)), 'constant', constant_values=padding_value)])
