import numpy as np

from .metric import Metric
from .. import constants


class SimilarityMetrics(Metric):
    
    @staticmethod
    def spectral_angle(observed_intensities, predicted_intensities):
        """
        calculate spectral angle
        :param observed_intensities: observed intensities, constants.EPSILON intensity indicates zero intensity peaks, 0 intensity indicates invalid peaks (charge state > peptide charge state or position >= peptide length), array of length 174
        :param predicted_intensities: predicted intensities, see observed_intensities for details, array of length 174
        """
        epsilon = 1e-7
        
        observed_peaks = np.argwhere(observed_intensities > constants.EPSILON)
        predicted_peaks = np.argwhere(predicted_intensities > constants.EPSILON)
        
        not_both_zero = np.union1d(observed_peaks, predicted_peaks)
        
        if len(not_both_zero) == 0:
            return 0.0
        
        observed_masked = observed_intensities[not_both_zero]
        predicted_masked = predicted_intensities[not_both_zero]
        
        observed_masked += epsilon
        predicted_masked += epsilon
        
        observed_normalized = normalize_vector(observed_masked)
        predicted_normalized = normalize_vector(predicted_masked)
        
        dot_product = np.sum(observed_normalized * predicted_normalized, axis=0)
        
        arccos = np.arccos(dot_product)
        return 1 - 2 * arccos / np.pi
    
    @staticmethod
    def l2_norm(vector):
        # = np.sqrt(np.sum(np.square(vector), axis=0))
        return np.linalg.norm(vector, axis=0)
    
    @staticmethod
    def normalize_vector(vector):
        return vector / SimilarityMetrics.l2_norm(vector)
    
    def pearson(self):
        pass

    def calc(self):
        pass
