import enum

import numpy as np

from .metric import Metric
from .. import constants

class ObservationState(enum.Enum):
    """
    - 4: not seen in either
    - 3: predicted but not in observed
    - 2: seen in both
    - 1: observed but not in predicted
    - 0: invalid
    """
    NOT_OBS_AND_NOT_PRED = 4
    NOT_OBS_BUT_PRED = 3
    OBS_AND_PRED = 2
    OBS_BUT_NOT_PRED = 1
    INVALID_ION = 0

class FragmentsRatio(Metric):

    @staticmethod
    def count_ions(boolean_array):
        """
        add value to metrics val
        :param boolean_array: boolean array with True for observed/predicted peaks and False for missing observed/predicted peaks, array of length 174
        :return: number of observed/predicted peaks
        """
        return FragmentsRatio.count_with_ion_mask(boolean_array)
    
    @staticmethod
    def count_ions_b(boolean_array):
        """
        add value to metrics val
        :param boolean_array: boolean array with True for observed/predicted peaks and False for missing observed/predicted peaks, array of length 174
        :return: number of observed/predicted b-ions
        """
        return FragmentsRatio.count_with_ion_mask(boolean_array, constants.B_ION_MASK)
    
    @staticmethod
    def count_ions_y(boolean_array):
        """
        add value to metrics val
        :param boolean_array: boolean array with True for observed/predicted peaks and False for missing observed/predicted peaks, array of length 174
        :return: number of observed/predicted y-ions
        """
        return FragmentsRatio.count_with_ion_mask(boolean_array, constants.Y_ION_MASK)
    
    @staticmethod
    def count_with_ion_mask(boolean_array, ion_mask = []):
        """
        add value to metrics val
        :param intensities: intensities array, array of length 174
        :param ion_mask: mask with 1s for the ions that should be counted and 0s for ions that should be ignored, integer array of length 174
        :return: number of observed/predicted peaks not masked by ion_mask
        """
        if len(ion_mask) > 0:
            boolean_array = np.multiply(boolean_array, ion_mask)
        return np.sum(boolean_array, axis = 1)
    
    @staticmethod
    def count_observed_and_predicted(observation_state):
        """
        add value to metrics val
        :param observation_state: integer observation_state, array of length 174
        :return: number of peaks that were both observed and predicted
        """
        return FragmentsRatio.count_observation_states(observation_state, ObservationState.OBS_AND_PRED)
    
    @staticmethod
    def count_observed_and_predicted_b(observation_state):
        """
        add value to metrics val
        :param observation_state: integer observation_state, array of length 174
        :return: number of b-ions that were both observed and predicted
        """
        return FragmentsRatio.count_observation_states(observation_state, ObservationState.OBS_AND_PRED, constants.B_ION_MASK)
    
    @staticmethod
    def count_observed_and_predicted_y(observation_state):
        """
        add value to metrics val
        :param observation_state: integer observation_state, array of length 174
        :return: number of y-ions that were both observed and predicted
        """
        return FragmentsRatio.count_observation_states(observation_state, ObservationState.OBS_AND_PRED, constants.Y_ION_MASK)
        
    @staticmethod
    def count_not_observed_and_not_predicted(observation_state):
        """
        :param observation_state: integer observation_state, array of length 174
        :return: number of peaks that were observed but not predicted
        """
        return FragmentsRatio.count_observation_states(observation_state, ObservationState.NOT_OBS_AND_NOT_PRED)
    
    @staticmethod
    def count_not_observed_and_not_predicted_b(observation_state):
        """
        :param observation_state: integer observation_state, array of length 174
        :return: number of b-ions that were observed but not predicted
        """
        return FragmentsRatio.count_observation_states(observation_state, ObservationState.NOT_OBS_AND_NOT_PRED, constants.B_ION_MASK)
    
    @staticmethod
    def count_not_observed_and_not_predicted_y(observation_state):
        """
        :param observation_state: integer observation_state, array of length 174
        :return: number of y-ions that were observed but not predicted
        """
        return FragmentsRatio.count_observation_states(observation_state, ObservationState.NOT_OBS_AND_NOT_PRED, constants.Y_ION_MASK)
    
    @staticmethod
    def count_not_observed_but_predicted(observation_state):
        """
        :param observation_state: integer observation_state, array of length 174
        :return: number of peaks that were not observed but predicted
        """
        return FragmentsRatio.count_observation_states(observation_state, ObservationState.NOT_OBS_BUT_PRED)
    
    @staticmethod
    def count_not_observed_but_predicted_b(observation_state):
        """
        :param observation_state: integer observation_state, array of length 174
        :return: number of b-ions that were not observed but predicted
        """
        return FragmentsRatio.count_observation_states(observation_state, ObservationState.NOT_OBS_BUT_PRED, constants.B_ION_MASK)
    
    @staticmethod
    def count_not_observed_but_predicted_y(observation_state):
        """
        :param observation_state: integer observation_state, array of length 174
        :return: number of y-ions that were not observed but predicted
        """
        return FragmentsRatio.count_observation_states(observation_state, ObservationState.NOT_OBS_BUT_PRED, constants.Y_ION_MASK)
    
    @staticmethod
    def count_observed_but_not_predicted(observation_state):
        """
        :param observation_state: integer observation_state, array of length 174
        :return: number of peaks that were both not observed and not predicted
        """
        return FragmentsRatio.count_observation_states(observation_state, ObservationState.OBS_BUT_NOT_PRED)
    
    @staticmethod
    def count_observed_but_not_predicted_b(observation_state):
        """
        :param observation_state: integer observation_state, array of length 174
        :return: number of b-ions that were both not observed and not predicted
        """
        return FragmentsRatio.count_observation_states(observation_state, ObservationState.OBS_BUT_NOT_PRED, constants.B_ION_MASK)
    
    @staticmethod
    def count_observed_but_not_predicted_y(observation_state):
        """
        :param observation_state: integer observation_state, array of length 174
        :return: number of y-ions that were both not observed and not predicted
        """
        return FragmentsRatio.count_observation_states(observation_state, ObservationState.OBS_BUT_NOT_PRED, constants.Y_ION_MASK)
    
    @staticmethod
    def count_observation_states(observation_state, test_state, ion_mask = []):
        """
        :param observation_state: integer observation_state, array of length 174
        :param i: integer, which observation state we want to count for
        :return:
        """
        state_boolean = (observation_state == test_state)
        if len(ion_mask) > 0:
            matrix_dimensions = (len(observation_state), 1)
            row_mask = (ion_mask == 0)[np.newaxis, :]
            state_boolean[np.tile(row_mask, matrix_dimensions)] = False
        return np.sum(state_boolean)
    
    @staticmethod
    def get_mask_observed_valid(observed_mz):
        """
        Creates a mask out of an observed m/z array with True for invalid entries and False for valid entries in the observed intensities array
        :param observed: observed m/z, array of length 174
        :return: boolean array, array of length 174
        """
        valids = (observed_mz > 0)
        valids[np.isnan(observed_mz)] = False
        return valids
    
    @staticmethod
    def make_boolean(intensities, mask, cutoff = constants.EPSILON):
        """
        Transform array of intensities into boolean array with True if > cutoff and False otherwise
        :param intensities: observed or predicted intensities, array of length 174
        :param mask: mask with True for invalid values in the observed intensities array, boolean array of length 174
        :param cutoff: minimum intensity value to be considered a peak, for observed intensities use the default cutoff of 0.0, for predicted intensities, set a cutoff, e.g. 0.05
        :return: boolean array, array of length 174
        """
        intensities_above_cutoff = (intensities > cutoff)
        intensities_above_cutoff[~mask] = False
        return intensities_above_cutoff
       
    @staticmethod 
    def get_observation_state(observed_boolean, predicted_boolean, mask):
        """
        Computes the observation state between the observed and predicted boolean arrays.
        possible values:
        - 4: not seen in either
        - 3: predicted but not in observed
        - 2: seen in both
        - 1: observed but not in predicted
        - 0: invalid
        :param observed_boolean: boolean observed intensities, boolean array of length 174
        :param predicted_boolean : boolean predicted intensities, boolean array of length 174
        :return: integer array, array of length 174
        """
        observation_state = np.ones_like(observed_boolean, dtype = ObservationState)
        observation_state[observed_boolean & predicted_boolean] = ObservationState.OBS_AND_PRED
        observation_state[observed_boolean & ~predicted_boolean] = ObservationState.OBS_BUT_NOT_PRED
        observation_state[~observed_boolean & predicted_boolean] = ObservationState.NOT_OBS_BUT_PRED
        observation_state[~observed_boolean & ~predicted_boolean] = ObservationState.NOT_OBS_AND_NOT_PRED
        observation_state[~mask] = ObservationState.INVALID_ION
        return observation_state
    
    def calc(self):
        """
        Adds columns with count, fraction and fraction_predicted features to metrics_val dataframe
        """
        mask_observed_valid = FragmentsRatio.get_mask_observed_valid(self.true_intensities)
        observed_boolean = FragmentsRatio.make_boolean(self.true_intensities, mask_observed_valid)
        predicted_boolean = FragmentsRatio.make_boolean(self.pred_intensities, mask_observed_valid, cutoff = 0.05)
        observation_state = FragmentsRatio.get_observation_state(observed_boolean, predicted_boolean, mask_observed_valid)
        valid_ions = np.maximum(1, np.sum(mask_observed_valid))
        valid_ions_b = np.maximum(1, np.sum(np.multiply(mask_observed_valid, constants.B_ION_MASK)))
        valid_ions_y = np.maximum(1, np.sum(np.multiply(mask_observed_valid, constants.Y_ION_MASK)))
        
        
        # counting metrics
        self.metrics_val['count_predicted'] = FragmentsRatio.count_ions(predicted_boolean)
        self.metrics_val['count_predicted_b'] = FragmentsRatio.count_ions_b(predicted_boolean)
        self.metrics_val['count_predicted_y'] = FragmentsRatio.count_ions_y(predicted_boolean)
        
        self.metrics_val['count_observed'] = FragmentsRatio.count_ions(observed_boolean)
        self.metrics_val['count_observed_b'] = FragmentsRatio.count_ions_b(observed_boolean)
        self.metrics_val['count_observed_y'] = FragmentsRatio.count_ions_y(observed_boolean)
        
        self.metrics_val['count_observed_and_predicted'] = FragmentsRatio.count_observed_and_predicted(observation_state)
        self.metrics_val['count_observed_and_predicted_b'] = FragmentsRatio.count_observed_and_predicted_b(observation_state)
        self.metrics_val['count_observed_and_predicted_y'] = FragmentsRatio.count_observed_and_predicted_y(observation_state)
        
        self.metrics_val['count_not_observed_and_not_predicted'] = FragmentsRatio.count_not_observed_and_not_predicted(observation_state)
        self.metrics_val['count_not_observed_and_not_predicted_b'] = FragmentsRatio.count_not_observed_and_not_predicted_b(observation_state)
        self.metrics_val['count_not_observed_and_not_predicted_y'] = FragmentsRatio.count_not_observed_and_not_predicted_y(observation_state)
        
        self.metrics_val['count_observed_but_not_predicted'] = FragmentsRatio.count_observed_but_not_predicted(observation_state)
        self.metrics_val['count_observed_but_not_predicted_b'] = FragmentsRatio.count_observed_but_not_predicted_b(observation_state)
        self.metrics_val['count_observed_but_not_predicted_y'] = FragmentsRatio.count_observed_but_not_predicted_y(observation_state)
        
        self.metrics_val['count_not_observed_but_predicted'] = FragmentsRatio.count_not_observed_but_predicted(observation_state)
        self.metrics_val['count_not_observed_but_predicted_b'] = FragmentsRatio.count_not_observed_but_predicted_b(observation_state)
        self.metrics_val['count_not_observed_but_predicted_y'] = FragmentsRatio.count_not_observed_but_predicted_y(observation_state)
        
        
        # fractional count metrics
        print(self.metrics_val['count_predicted'])
        self.metrics_val['fraction_predicted'] = self.metrics_val['count_predicted'].values / valid_ions
        self.metrics_val['fraction_predicted_b'] = self.metrics_val['count_predicted_b'] / valid_ions_b
        self.metrics_val['fraction_predicted_y'] = self.metrics_val['count_predicted_y'] / valid_ions_y
        
        self.metrics_val['fraction_observed'] = self.metrics_val['count_observed'] / valid_ions
        self.metrics_val['fraction_observed_b'] = self.metrics_val['count_observed_b'] / valid_ions_b
        self.metrics_val['fraction_observed_y'] = self.metrics_val['count_observed_y'] / valid_ions_y
        
        self.metrics_val['fraction_observed_and_predicted'] = self.metrics_val['count_observed_and_predicted'] / valid_ions
        self.metrics_val['fraction_observed_and_predicted_b'] = self.metrics_val['count_observed_and_predicted_b'] / valid_ions_b
        self.metrics_val['fraction_observed_and_predicted_y'] = self.metrics_val['count_observed_and_predicted_y'] / valid_ions_y
        
        self.metrics_val['fraction_not_observed_and_not_predicted'] = self.metrics_val['count_not_observed_and_not_predicted'] / valid_ions
        self.metrics_val['fraction_not_observed_and_not_predicted_b'] = self.metrics_val['count_not_observed_and_not_predicted_b'] / valid_ions_b
        self.metrics_val['fraction_not_observed_and_not_predicted_y'] = self.metrics_val['count_not_observed_and_not_predicted_y'] / valid_ions_y
        
        self.metrics_val['fraction_observed_but_not_predicted'] = self.metrics_val['count_observed_but_not_predicted'] / valid_ions
        self.metrics_val['fraction_observed_but_not_predicted_b'] = self.metrics_val['count_observed_but_not_predicted_b'] / valid_ions_b
        self.metrics_val['fraction_observed_but_not_predicted_y'] = self.metrics_val['count_observed_but_not_predicted_y'] / valid_ions_y
        
        self.metrics_val['fraction_not_observed_but_predicted'] = self.metrics_val['count_not_observed_but_predicted'] / valid_ions
        self.metrics_val['fraction_not_observed_but_predicted_b'] = self.metrics_val['count_not_observed_but_predicted_b'] / valid_ions_b
        self.metrics_val['fraction_not_observed_but_predicted_y'] = self.metrics_val['count_not_observed_but_predicted_y'] / valid_ions_y
        
        
        # fractional count metrics relative to predictions
        num_predicted_ions = np.maximum(1, self.metrics_val['count_predicted'])
        num_predicted_ions_b = np.maximum(1, self.metrics_val['count_predicted_b'])
        num_predicted_ions_y = np.maximum(1, self.metrics_val['count_predicted_y'])
        
        self.metrics_val['fraction_predicted_observed_and_predicted'] = self.metrics_val['count_observed_and_predicted'] / num_predicted_ions
        self.metrics_val['fraction_predicted_observed_and_predicted_b'] = self.metrics_val['count_observed_and_predicted_b'] / num_predicted_ions_b
        self.metrics_val['fraction_predicted_observed_and_predicted_y'] = self.metrics_val['count_observed_and_predicted_y'] / num_predicted_ions_y
        
        self.metrics_val['fraction_predicted_not_observed_and_not_predicted'] = self.metrics_val['count_not_observed_and_not_predicted'] / num_predicted_ions
        self.metrics_val['fraction_predicted_not_observed_and_not_predicted_b'] = self.metrics_val['count_not_observed_and_not_predicted_b'] / num_predicted_ions_b
        self.metrics_val['fraction_predicted_not_observed_and_not_predicted_y'] = self.metrics_val['count_not_observed_and_not_predicted_y'] / num_predicted_ions_y
        
        self.metrics_val['fraction_predicted_observed_but_not_predicted'] = self.metrics_val['count_observed_but_not_predicted'] / num_predicted_ions
        self.metrics_val['fraction_predicted_observed_but_not_predicted_b'] = self.metrics_val['count_observed_but_not_predicted_b'] / num_predicted_ions_b
        self.metrics_val['fraction_predicted_observed_but_not_predicted_y'] = self.metrics_val['count_observed_but_not_predicted_y'] / num_predicted_ions_y
        
        self.metrics_val['fraction_predicted_not_observed_but_predicted'] = self.metrics_val['count_not_observed_but_predicted'] / num_predicted_ions
        self.metrics_val['fraction_predicted_not_observed_but_predicted_b'] = self.metrics_val['count_not_observed_but_predicted_b'] / num_predicted_ions_b
        self.metrics_val['fraction_predicted_not_observed_but_predicted_y'] = self.metrics_val['count_not_observed_but_predicted_y'] / num_predicted_ions_y
        
