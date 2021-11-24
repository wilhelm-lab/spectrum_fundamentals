import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from .metric import Metric
from .. import constants


class SimilarityMetrics(Metric):
    
    @staticmethod
    def spectral_angle(observed_intensities, predicted_intensities):
        """
        calculate spectral angle, we only consider fragments for which a non-zero intensity was predicted
        :param observed_intensities: observed intensities, constants.EPSILON intensity indicates zero intensity peaks, 0 intensity indicates invalid peaks (charge state > peptide charge state or position >= peptide length), array of length 174
        :param predicted_intensities: predicted intensities, see observed_intensities for details, array of length 174
        """
        valid_ion_mask = predicted_intensities > constants.EPSILON
        if scipy.sparse.issparse(valid_ion_mask):
            observed_masked = observed_intensities.multiply(valid_ion_mask)
            predicted_masked = predicted_intensities.multiply(valid_ion_mask)
        else:
            observed_masked = np.multiply(observed_intensities, valid_ion_mask)
            predicted_masked = np.multiply(predicted_intensities, valid_ion_mask)
        
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
