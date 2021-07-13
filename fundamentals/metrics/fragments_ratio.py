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
    def count_predicted(predicted, cutoff):
        """
        add value to metrics val
        :param predicted: predicted intensities array, array of length 174
        :param cutoff: minimum intensity value to be considered a peak, e.g. 0.05
        :return: number of predicted peaks with intensity above cutoff
        """
        return np.sum(predicted > cutoff)
    
    @staticmethod
    def count_predicted_b(predicted, cutoff):
        """
        add value to metrics val
        :param predicted: predicted intensities array, array of length 174
        :param cutoff: minimum intensity value to be considered a peak, e.g. 0.05
        :return: number of predicted b-ion peaks with intensity above cutoff
        """
        return FragmentsRatio.count_predicted_with_ion_mask(predicted, constants.B_ION_MASK, cutoff)
    
    @staticmethod
    def count_predicted_y(predicted, cutoff):
        """
        add value to metrics val
        :param predicted: predicted intensities array, array of length 174
        :param cutoff: minimum intensity value to be considered a peak, e.g. 0.05
        :return: number of predicted y-ion peaks with intensity above cutoff
        """
        return FragmentsRatio.count_predicted_with_ion_mask(predicted, constants.Y_ION_MASK, cutoff)
    
    @staticmethod
    def count_predicted_with_ion_mask(predicted, ion_mask, cutoff):
        """
        add value to metrics val
        :param predicted: predicted intensities array, array of length 174
        :param ion_mask: mask with 1s for the ions that should be counted and 0s for ions that should be ignored, integer array of length 174
        :param cutoff: minimum intensity value to be considered a peak, e.g. 0.05
        :return: number of predicted b-ion peaks with intensity above cutoff
        """
        predicted_b = np.multiply(predicted, ion_mask)
        return np.sum(predicted_b > cutoff)
    
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
        if len(ion_mask) > 0:
            observation_state[ion_mask == 0] = ObservationState.INVALID_ION
        return np.sum(observation_state == test_state)
        
    def pred_raw_(self):
        pass

    def calc(self):
        pass
    
    @staticmethod
    def get_mask_observed_invalid(observed_mz):
        """
        Creates a mask out of an observed m/z array with True for invalid entries and False for valid entries in the observed intensities array
        :param observed: observed m/z, array of length 174
        :return: boolean array, array of length 174
        """
        invalids = (observed_mz == -1)
        invalids[np.isnan(observed_mz)] = True
        return invalids
    
    @staticmethod
    def make_boolean(intensities, mask, cutoff = 0.0):
        """
        Transform array of intensities into boolean array with True if > cutoff and False otherwise
        :param intensities: observed or predicted intensities, array of length 174
        :param mask: mask with True for invalid values in the observed intensities array, boolean array of length 174
        :param cutoff: minimum intensity value to be considered a peak, for observed intensities use the default cutoff of 0.0, for predicted intensities, set a cutoff, e.g. 0.05
        :return: boolean array, array of length 174
        """
        intensities_above_cutoff = (intensities > cutoff)
        intensities_above_cutoff[mask] = False
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
        observation_state = np.ones(constants.NUM_IONS, dtype = ObservationState)
        observation_state[observed_boolean & predicted_boolean] = ObservationState.OBS_AND_PRED
        observation_state[observed_boolean & ~predicted_boolean] = ObservationState.OBS_BUT_NOT_PRED
        observation_state[~observed_boolean & predicted_boolean] = ObservationState.NOT_OBS_BUT_PRED
        observation_state[~observed_boolean & ~predicted_boolean] = ObservationState.NOT_OBS_AND_NOT_PRED
        observation_state[~mask] = ObservationState.INVALID_ION
        return observation_state
    
    
