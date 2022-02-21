import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import scipy.stats
from scipy import spatial
from numpy import mean, absolute, std
from .metric import Metric
from .. import constants
from sklearn.metrics import mean_squared_error


class SimilarityMetrics(Metric):
    
    @staticmethod
    def spectral_angle(observed_intensities, predicted_intensities, charge=0):
        """
        calculate spectral angle
        :param observed_intensities: observed intensities, constants.EPSILON intensity indicates zero intensity peaks, 0 intensity indicates invalid peaks (charge state > peptide charge state or position >= peptide length), array of length 174
        :param predicted_intensities: predicted intensities, see observed_intensities for details, array of length 174
        :param charge: to filter by the peak charges, 0 means everything.
        """
        epsilon = 1e-7
        valid_ion_mask = predicted_intensities > epsilon

        if scipy.sparse.issparse(valid_ion_mask):
            observed_masked = observed_intensities.multiply(valid_ion_mask)
            predicted_masked = predicted_intensities.multiply(valid_ion_mask)
        else:
            observed_masked = np.multiply(observed_intensities, valid_ion_mask)
            predicted_masked = np.multiply(predicted_intensities, valid_ion_mask)
        
        observed_normalized = SimilarityMetrics.unit_normalization(observed_masked)
        predicted_normalized = SimilarityMetrics.unit_normalization(predicted_masked)

        if charge != 0:
            if charge == 1:
                boolean_array = constants.SINGLE_CHARGED_MASK
            elif charge == 2:
                boolean_array = constants.DOUBLE_CHARGED_MASK
            elif charge == 3:
                boolean_array = constants.TRIPLE_CHARGED_MASK
            elif charge == 4:
                boolean_array = constants.B_ION_MASK
            else:
                boolean_array = constants.Y_ION_MASK

            boolean_array = scipy.sparse.csr_matrix(boolean_array)
            observed_normalized = scipy.sparse.csr_matrix(observed_normalized)
            predicted_normalized = scipy.sparse.csr_matrix(predicted_normalized)
            observed_normalized = observed_normalized.multiply(boolean_array).toarray()
            predicted_normalized = predicted_normalized.multiply(boolean_array).toarray()

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
            return np.array(np.sum(scipy.sparse.csr_matrix.multiply(observed_normalized, predicted_normalized), axis=1)).flatten()
        else:
            return np.sum(np.multiply(observed_normalized, predicted_normalized), axis=1)

    @staticmethod
    def correlation(observed_intensities, predicted_intensities, charge=0, method="pearson"):
        epsilon = 1e-7
        observed_intensities = observed_intensities.toarray()
        predicted_intensities = predicted_intensities.toarray()

        if charge != 0:
            if charge == 1:
                boolean_array = constants.SINGLE_CHARGED_MASK
            elif charge == 2:
                boolean_array = constants.DOUBLE_CHARGED_MASK
            elif charge == 3:
                boolean_array = constants.TRIPLE_CHARGED_MASK
            elif charge == 4:
                boolean_array = constants.B_ION_MASK
            else:
                boolean_array = constants.Y_ION_MASK

            boolean_array = scipy.sparse.csr_matrix(boolean_array)
            observed_intensities = scipy.sparse.csr_matrix(observed_intensities)
            predicted_intensities = scipy.sparse.csr_matrix(predicted_intensities)
            observed_intensities = observed_intensities.multiply(boolean_array).toarray()
            predicted_intensities = predicted_intensities.multiply(boolean_array).toarray()

        pear_corr = []
        for obs, pred in zip(observed_intensities, predicted_intensities):
            valid_ion_mask = pred > epsilon
            obs = obs[valid_ion_mask]
            pred = pred[valid_ion_mask]
            obs = obs[~np.isnan(obs)]
            pred = pred[~np.isnan(pred)]
            if len(obs) > 2 and len(pred) > 2:
                corr = scipy.stats.pearsonr(obs, pred)[0] if method == "pearson" else scipy.stats.spearmanr(obs, pred)[0]
            else:
                corr = 0
            if np.isnan(corr):
                corr = 0
            pear_corr.append(corr)

        return pear_corr

    @staticmethod
    def cos(observed_intensities, predicted_intensities):
        epsilon = 1e-7
        observed_normalized = SimilarityMetrics.unit_normalization(observed_intensities)
        predicted_normalized = SimilarityMetrics.unit_normalization(predicted_intensities)
        observed_intensities = observed_normalized.toarray()
        predicted_intensities = predicted_normalized.toarray()

        cos_values = []
        for obs, pred in zip(observed_intensities, predicted_intensities):
            valid_ion_mask = pred > epsilon
            obs = obs[valid_ion_mask]
            pred = pred[valid_ion_mask]
            obs = obs[~np.isnan(obs)]
            pred = pred[~np.isnan(pred)]
            cos = 1 - spatial.distance.cosine(obs, pred)
            if np.isnan(cos):
                cos = 0
            cos_values.append(cos)

        return cos_values

    @staticmethod
    def abs_diff(observed_intensities, predicted_intensities, mean_std):
        epsilon = 1e-7
        observed_normalized = SimilarityMetrics.unit_normalization(observed_intensities)
        predicted_normalized = SimilarityMetrics.unit_normalization(predicted_intensities)
        observed_intensities = observed_normalized.toarray()
        predicted_intensities = predicted_normalized.toarray()

        diff_values = []
        for obs, pred in zip(observed_intensities, predicted_intensities):
            valid_ion_mask = pred > epsilon
            obs = obs[valid_ion_mask]
            pred = pred[valid_ion_mask]
            obs = obs[~np.isnan(obs)]
            pred = pred[~np.isnan(pred)]
            if mean_std == "mean":
                diff = mean(absolute(obs - mean(pred)))
            elif mean_std == "std":
                diff = std(absolute(obs - mean(pred)))
            elif mean_std == "q3":
                diff = np.quantile(absolute(obs - mean(pred)), .75)
            elif mean_std == "q2":
                diff = np.quantile(absolute(obs - mean(pred)), .5)
            elif mean_std == "q1":
                diff = np.quantile(absolute(obs - mean(pred)), .25)
            elif mean_std == "max":
                diff = np.max(absolute(obs - mean(pred)))
            elif mean_std == "min":
                diff = np.min(absolute(obs - mean(pred)))
            elif mean_std == "mse":
                diff = mean_squared_error(obs, pred)
            if np.isnan(diff):
                diff = 0
            diff_values.append(diff)

        return diff_values

    def calc(self, all_features):
        """
        Adds columns with spectral angle feature to metrics_val dataframe
        """

        self.metrics_val['spectral_angle'] = SimilarityMetrics.spectral_angle(self.true_intensities,
                                                                              self.pred_intensities, 0)
        self.metrics_val['pearson_corr'] = SimilarityMetrics.correlation(self.true_intensities, self.pred_intensities,
                                                                         0, "pearson")
        if all_features:
            self.metrics_val['spectral_angle_single_charge'] = SimilarityMetrics.spectral_angle(self.true_intensities,
                                                                                            self.pred_intensities, 1)
            self.metrics_val['spectral_angle_double_charge'] = SimilarityMetrics.spectral_angle(self.true_intensities,
                                                                                            self.pred_intensities, 2)
            self.metrics_val['spectral_angle_triple_charge'] = SimilarityMetrics.spectral_angle(self.true_intensities,
                                                                                            self.pred_intensities, 3)
            self.metrics_val['spectral_angle_b_ions'] = SimilarityMetrics.spectral_angle(self.true_intensities,
                                                                                     self.pred_intensities, 4)
            self.metrics_val['spectral_angle_y_ions'] = SimilarityMetrics.spectral_angle(self.true_intensities,
                                                                                     self.pred_intensities, 5)
            self.metrics_val['pearson_corr_single_charge'] = SimilarityMetrics.correlation(
                self.true_intensities, self.pred_intensities, 1, "pearson")
            self.metrics_val['pearson_corr_double_charge'] = SimilarityMetrics.correlation(
                self.true_intensities, self.pred_intensities, 2, "pearson")
            self.metrics_val['pearson_corr_triple_charge'] = SimilarityMetrics.correlation(
                self.true_intensities, self.pred_intensities, 3, "pearson")
            self.metrics_val['pearson_corr_b_ions'] = SimilarityMetrics.correlation(
                self.true_intensities, self.pred_intensities, 4, "pearson")
            self.metrics_val['pearson_corr_y_ions'] = SimilarityMetrics.correlation(
                self.true_intensities, self.pred_intensities, 5, "pearson")
            self.metrics_val['cos'] = SimilarityMetrics.cos(
            self.true_intensities, self.pred_intensities)
            self.metrics_val['mean_abs_diff'] = SimilarityMetrics.abs_diff(
            self.true_intensities, self.pred_intensities, "mean")
            self.metrics_val['std_abs_diff'] = SimilarityMetrics.abs_diff(
            self.true_intensities, self.pred_intensities, "std")
            self.metrics_val['abs_diff_Q3'] = SimilarityMetrics.abs_diff(
            self.true_intensities, self.pred_intensities, "q3")
            self.metrics_val['abs_diff_Q2'] = SimilarityMetrics.abs_diff(
            self.true_intensities, self.pred_intensities, "q2")
            self.metrics_val['abs_diff_Q1'] = SimilarityMetrics.abs_diff(
            self.true_intensities, self.pred_intensities, "q1")
            self.metrics_val['min_abs_diff'] = SimilarityMetrics.abs_diff(
            self.true_intensities, self.pred_intensities, "min")
            self.metrics_val['max_abs_diff'] = SimilarityMetrics.abs_diff(
            self.true_intensities, self.pred_intensities, "max")
            self.metrics_val['mse'] = SimilarityMetrics.abs_diff(self.true_intensities, self.pred_intensities, "mse")
            self.metrics_val['spearman_corr'] = SimilarityMetrics.correlation(self.true_intensities,
                                                                          self.pred_intensities, 0, "spearman")
            self.metrics_val['spearman_corr_single_charge'] = SimilarityMetrics.correlation(
            self.true_intensities, self.pred_intensities, 1, "spearman")
            self.metrics_val['spearman_corr_double_charge'] = SimilarityMetrics.correlation(
            self.true_intensities, self.pred_intensities, 2, "spearman")
            self.metrics_val['spearman_corr_triple_charge'] = SimilarityMetrics.correlation(
            self.true_intensities, self.pred_intensities, 3, "spearman")
            self.metrics_val['spearman_corr_b_ions'] = SimilarityMetrics.correlation(
            self.true_intensities, self.pred_intensities, 4, "spearman")
            self.metrics_val['spearman_corr_y_ions'] = SimilarityMetrics.correlation(
            self.true_intensities, self.pred_intensities, 5, "spearman")
