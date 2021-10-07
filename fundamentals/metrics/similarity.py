import numpy as np
import scipy.sparse
import scipy.sparse.linalg

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
        epsilon = 2e-7

        # TODO: clean this up
        if False:
            spectral_angle = []
            for predicted_intensities, observed_intensities in zip(predicted_intensities_a,observed_intensities_a):
                #observed_peaks = np.argwhere(observed_intensities > constants.EPSILON)
                predicted_peaks = np.argwhere(predicted_intensities > constants.EPSILON)

                #print(observed_peaks)
                #print(predicted_peaks)
                #not_both_zero = np.union1d(observed_peaks, predicted_peaks)
                #print(not_both_zero)
                #if len(not_both_zero) == 0:
                    #return 0.0

                observed_masked = observed_intensities[predicted_peaks]
                predicted_masked = predicted_intensities[predicted_peaks]

                true_norm = observed_masked * (1 / np.sqrt(np.sum(np.square(observed_masked), axis=0)))
                pred_norm = predicted_masked * (1 / np.sqrt(np.sum(np.square(predicted_masked), axis=0)))
                product = np.sum(true_norm * pred_norm, axis=0)
                arccos = np.arccos(product)
                spectral_angle.append(1 - 2 * arccos / np.pi)
            return spectral_angle
            
            #observed_masked += epsilon
            #predicted_masked += epsilon
        else:
            #print(predicted_intensities)
            not_zero_mask = predicted_intensities > 0
            observed_masked = observed_intensities.multiply(not_zero_mask)
            predicted_masked = predicted_intensities.multiply(not_zero_mask)
            #print(predicted_masked)
        
        observed_normalized = SimilarityMetrics.unit_normalization(observed_masked)
        predicted_normalized = SimilarityMetrics.unit_normalization(predicted_masked)
        
        dot_product = SimilarityMetrics.rowwise_dot_product(observed_normalized, predicted_normalized)
        
        arccos = np.arccos(dot_product)
        return 1 - 2 * arccos / np.pi
    
    @staticmethod
    def l2_norm(matrix):
        """
        compute the l2-norm ( sqrt(sum(x^2) ) for each row of the matrix
        :param matrix: matrix with intensities, constants.EPSILON intensity indicates zero intensity peaks, 0 intensity indicates invalid peaks (charge state > peptide charge state or position >= peptide length), matrix of size (nspectra, 174)
        :return: vector with rowwise norms of the matrix
        """
        # = np.sqrt(np.sum(np.square(matrix), axis=0))
        if scipy.sparse.issparse(matrix):
          return scipy.sparse.linalg.norm(matrix, axis=1)
        else:
          return np.linalg.norm(matrix, axis=1)
        
    
    @staticmethod
    def unit_normalization(matrix):
        """
        normalize each row of the matrix such that the norm equals 1.0
        :param matrix: matrix with intensities, constants.EPSILON intensity indicates zero intensity peaks, 0 intensity indicates invalid peaks (charge state > peptide charge state or position >= peptide length), matrix of size (nspectra, 174)
        :return: normalized matrix
        """
        rowwise_norm = SimilarityMetrics.l2_norm(matrix)
        # prevent divide by zero
        rowwise_norm[rowwise_norm == 0] = 1
        if scipy.sparse.issparse(matrix):
            reciprocal_rowwise_norm_matrix = scipy.sparse.csr_matrix(1 / rowwise_norm[:, np.newaxis])
            return scipy.sparse.csr_matrix.multiply(matrix, reciprocal_rowwise_norm_matrix)
        else:
            return matrix / rowwise_norm[:, np.newaxis]
    
    @staticmethod
    def rowwise_dot_product(observed_normalized, predicted_normalized):
        if scipy.sparse.issparse(observed_normalized):
            return np.array(np.sum(scipy.sparse.csr_matrix.multiply(observed_normalized, predicted_normalized), axis = 1)).flatten()
        else:
            return np.sum(np.multiply(observed_normalized, predicted_normalized), axis = 1)
        
    def pearson(self):
        pass

    def calc(self):
        """
        Adds columns with spectral angle feature to metrics_val dataframe
        """
        self.metrics_val['spectral_angle'] = SimilarityMetrics.spectral_angle(self.true_intensities, self.pred_intensities)
        
