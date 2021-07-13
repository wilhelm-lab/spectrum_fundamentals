import numpy as np

from .metric import Metric


class FragmentsRatio(Metric):

    def pred_not_seen(self):
        """
        add value to metrics val
        """
        pass

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
    def get_observation_state(observed_boolean, predicted_boolean):
        """
        Computes the observation state between the observed and predicted boolean arrays.
        possible values:
        - 2: not seen in either
        - 1: predicted but not in observed
        - 0: seen in both
        - -1: observed but not in predicted
        :param observed_boolean: boolean observed intensities, boolean array of length 174
        :param predicted_boolean : boolean predicted intensities, boolean array of length 174
        :return: integer array, array of length 174
        """
        observation_state = predicted_boolean.astype(int) - observed_boolean.astype(int)
        both_zero = (observation_state == 0) & ~predicted_boolean
        observation_state[both_zero] = 2
        return observation_state
    
    
