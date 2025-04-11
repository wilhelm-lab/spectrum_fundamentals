from typing import List, Union

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import scipy.stats
from numpy import absolute, mean, std
from scipy import spatial
from sklearn.metrics import mean_squared_error

from .. import constants
from .metric import Metric


def get_metric_func(metric: str):
    """
    Return a callable function for a given metric shortcut.

    :param metric: a shortcut for the desired metric.
    :raises ValueError: if the provided metric is not known

    :return: callable metric function

    """
    if metric == "mean":
        return lambda obs, pred: mean(absolute(obs - mean(pred)))
    if metric == "std":
        return lambda obs, pred: std(absolute(obs - mean(pred)))
    if metric == "max":
        return lambda obs, pred: np.max(absolute(obs - mean(pred)))
    if metric == "min":
        return lambda obs, pred: np.min(absolute(obs - mean(pred)))
    if metric == "mse":
        return lambda obs, pred: mean_squared_error(obs, pred)
    if metric.startswith("q"):
        return lambda obs, pred: SimilarityMetrics.calculate_quantiles(obs, pred, metric)

    raise ValueError(f"Unknown metric function {metric}")


class SimilarityMetrics(Metric):
    """Class to generate several features than can be used by percoltor for rescoring."""

    @staticmethod
    def spectral_angle(
        observed_intensities: Union[scipy.sparse.csr_matrix, np.ndarray],
        predicted_intensities: Union[scipy.sparse.csr_matrix, np.ndarray],
        charge: int = 0,
        xl: bool = False,
    ) -> np.ndarray:
        """
        Calculate spectral angle.

        :param observed_intensities: observed intensities, constants.EPSILON intensity indicates zero intensity peaks, \
                                     0 intensity indicates invalid peaks (charge state > peptide charge state or \
                                     position >= peptide length), array of length 174
        :param predicted_intensities: predicted intensities, see observed_intensities for details, array of length 174
        :param charge: to filter by the peak charges, 0 means everything
        :param xl: whether operating on cleavable crosslinked or linear peptides
        :raises ValueError: if charge is smaller than 1 or larger than 5
        :return: SA values
        """
        if charge != 0:
            if not 1 <= charge <= 5:
                raise ValueError("Charge must be between 1 to 5.")
            masks = constants.MASK_DICT_XL if xl else constants.MASK_DICT
            boolean_array = masks[charge]
            boolean_array = scipy.sparse.csr_matrix(boolean_array)
            observed_intensities = scipy.sparse.csr_matrix(observed_intensities)
            predicted_intensities = scipy.sparse.csr_matrix(predicted_intensities)
            observed_intensities = observed_intensities.multiply(boolean_array).toarray()
            predicted_intensities = predicted_intensities.multiply(boolean_array).toarray()

        predicted_non_zero_mask = predicted_intensities > constants.EPSILON

        if isinstance(observed_intensities, scipy.sparse.csr_matrix):
            observed_masked = observed_intensities.multiply(predicted_non_zero_mask)
        else:
            observed_masked = np.multiply(observed_intensities, predicted_non_zero_mask)

        if isinstance(predicted_intensities, scipy.sparse.csr_matrix):
            predicted_masked = predicted_intensities.multiply(predicted_non_zero_mask)
        else:
            predicted_masked = np.multiply(predicted_intensities, predicted_non_zero_mask)

        observed_normalized = SimilarityMetrics.unit_normalization(observed_masked)
        predicted_normalized = SimilarityMetrics.unit_normalization(predicted_masked)

        observed_non_zero_mask = observed_intensities > constants.EPSILON
        fragments_in_common = SimilarityMetrics.rowwise_dot_product(observed_non_zero_mask, predicted_non_zero_mask)

        dot_product = SimilarityMetrics.rowwise_dot_product(observed_normalized, predicted_normalized) * (
            fragments_in_common > 0
        )

        arccos = np.arccos(dot_product)
        sa = 1 - 2 * arccos / np.pi
        sa = np.nan_to_num(sa)
        return sa

    @staticmethod
    def l2_norm(matrix) -> np.ndarray:
        """
        Compute the l2-norm (sqrt(sum(x^2) ) for each row of the matrix.

        :param matrix: matrix with intensities, constants.EPSILON intensity indicates zero intensity peaks, \
                       0 intensity indicates invalid peaks (charge state > peptide charge state or position >= peptide length), \
                       matrix of size (nspectra, 174)
        :return: vector with rowwise norms of the matrix
        """
        # = np.sqrt(np.sum(np.square(matrix), axis=0))
        if scipy.sparse.issparse(matrix):
            return scipy.sparse.linalg.norm(matrix, axis=1)
        else:
            return np.linalg.norm(matrix, axis=1)

    @staticmethod
    def unit_normalization(
        matrix: Union[scipy.sparse.csr_matrix, np.ndarray],
    ) -> Union[scipy.sparse.csr_matrix, np.ndarray]:
        """
        Normalize each row of the matrix such that the norm equals 1.0.

        :param matrix: matrix with intensities, constants.EPSILON intensity indicates zero intensity peaks, \
                       0 intensity indicates invalid peaks (charge state > peptide charge state or position >= peptide length), \
                       matrix of size (nspectra, 174)
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
    def rowwise_dot_product(
        observed_intensities: Union[scipy.sparse.csr_matrix, np.ndarray],
        predicted_intensities: Union[scipy.sparse.csr_matrix, np.ndarray],
    ) -> np.ndarray:
        """
        Calculate rowwise dot product.

        :param observed_intensities: observed intensities, constants.EPSILON intensity indicates zero intensity peaks,
            0 intensity indicates invalid peaks (charge state > peptide charge state or position >= peptide length),
            array of length 174
        :param predicted_intensities: predicted intensities, see observed_intensities for details, array of length 174
        :return: matrix containing the rowwise dotproduct
        """
        if isinstance(observed_intensities, scipy.sparse.csr_matrix):
            return np.array(
                np.sum(scipy.sparse.csr_matrix.multiply(observed_intensities, predicted_intensities), axis=1)
            ).flatten()
        else:
            return np.sum(np.multiply(observed_intensities, predicted_intensities), axis=1)

    @staticmethod
    def spectral_entropy_similarity(
        observed_intensities: Union[scipy.sparse.csr_matrix, np.ndarray],
        predicted_intensities: Union[scipy.sparse.csr_matrix, np.ndarray],
    ) -> List[float]:
        """
        Calculate spectral entropy similarity as defined in Li et al. (Spectral entropy outperforms MS/MS dot product \
        similarity for small-molecule compound identification).

        :param observed_intensities: observed intensities, constants.EPSILON intensity indicates zero intensity peaks, \
                                     0 intensity indicates invalid peaks (charge state > peptide charge state or \
                                     position >= peptide length), array of length 174
        :param predicted_intensities: predicted intensities, see observed_intensities for details, array of length 174
        :return: spectral entropy similarity values
        """
        if isinstance(observed_intensities, scipy.sparse.csr_matrix):
            observed_intensities = observed_intensities.toarray()
        if isinstance(predicted_intensities, scipy.sparse.csr_matrix):
            predicted_intensities = predicted_intensities.toarray()

        entropies = []
        for obs, pred in zip(observed_intensities, predicted_intensities):
            valid_ion_mask = pred > constants.EPSILON
            obs = obs[valid_ion_mask]
            pred = pred[valid_ion_mask]
            obs = obs[~np.isnan(obs)]
            pred = pred[~np.isnan(pred)]
            entropy_merged = scipy.stats.entropy(obs + pred)
            entropy_pred = scipy.stats.entropy(pred)
            entropy_obs = scipy.stats.entropy(obs)
            entropy = 1 - (2 * entropy_merged - entropy_obs - entropy_pred) / np.log(4)
            if np.isnan(entropy):
                entropy = 0
            entropies.append(entropy)

        return entropies

    @staticmethod
    def correlation(
        observed_intensities: scipy.sparse.csr_matrix,
        predicted_intensities: scipy.sparse.csr_matrix,
        charge: int = 0,
        method: str = "pearson",
        xl: bool = False,
    ) -> List[float]:
        """
        Calculate correlation between observed and predicted.

        :param observed_intensities: observed intensities, constants.EPSILON intensity indicates zero intensity peaks, \
                                     0 intensity indicates invalid peaks (charge state > peptide charge state or \
                                     position >= peptide length), array of length 174
        :param predicted_intensities: predicted intensities, see observed_intensities for details, array of length 174
        :param charge: to filter by the peak charges, 0 means everything
        :param method: either pearson or spearman
        :param xl: wheter or not to use xl mode
        :raises ValueError: if charge is smaller than 1 or larger than 5

        :return: calculated correlations
        """
        observed_intensities_array = observed_intensities.toarray()
        predicted_intensities_array = predicted_intensities.toarray()

        if charge != 0:
            if not 1 <= charge <= 5:
                raise ValueError("Charge must be between 1 to 5.")
            masks = constants.MASK_DICT_XL if xl else constants.MASK_DICT
            boolean_array = masks[charge]
            boolean_array = scipy.sparse.csr_matrix(boolean_array)
            observed_intensities_array = observed_intensities.multiply(boolean_array).toarray()
            predicted_intensities_array = predicted_intensities.multiply(boolean_array).toarray()

        pear_corr = []
        for obs, pred in zip(observed_intensities_array, predicted_intensities_array):
            valid_ion_mask = pred > constants.EPSILON
            obs = obs[valid_ion_mask]
            pred = pred[valid_ion_mask]
            obs = obs[~np.isnan(obs)]
            pred = pred[~np.isnan(pred)]
            if len(obs) > 2 and len(pred) > 2:
                corr = (
                    scipy.stats.pearsonr(obs, pred)[0] if method == "pearson" else scipy.stats.spearmanr(obs, pred)[0]
                )
            else:
                corr = 0
            if np.isnan(corr):
                corr = 0
            pear_corr.append(corr)

        return pear_corr

    @staticmethod
    def cos(
        observed_intensities: scipy.sparse.csr_matrix, predicted_intensities: scipy.sparse.csr_matrix
    ) -> List[float]:
        """
        Calculate cosine similarity.

        :param observed_intensities: observed intensities, constants.EPSILON intensity indicates zero intensity peaks, \
                                     0 intensity indicates invalid peaks (charge state > peptide charge state or \
                                     position >= peptide length), array of length 174
        :param predicted_intensities: predicted intensities, see observed_intensities for details, array of length 174
        :return: cosine values
        """
        epsilon = 1e-7
        observed_normalized = SimilarityMetrics.unit_normalization(observed_intensities)
        predicted_normalized = SimilarityMetrics.unit_normalization(predicted_intensities)

        if isinstance(observed_normalized, scipy.sparse.csr_matrix):
            observed_normalized = observed_normalized.toarray()
        if isinstance(predicted_normalized, scipy.sparse.csr_matrix):
            predicted_normalized = predicted_normalized.toarray()

        cos_values = []
        for obs, pred in zip(observed_normalized, predicted_normalized):
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
    def abs_diff(
        observed_intensities: scipy.sparse.csr_matrix, predicted_intensities: scipy.sparse.csr_matrix, metric: str
    ) -> List[float]:
        """
        Calculate several similarity metrics.

        :param observed_intensities: observed intensities, constants.EPSILON intensity indicates zero intensity peaks, \
                                     0 intensity indicates invalid peaks (charge state > peptide charge state or \
                                     position >= peptide length), array of length 174
        :param predicted_intensities: predicted intensities, see observed_intensities for details, array of length 174
        :param metric: metric (mean, std, q1, q2, q3, min, max, or mse)
        :return: calculated similarity values
        """
        chosen_metric = get_metric_func(metric)

        epsilon = 1e-7
        observed_normalized = SimilarityMetrics.unit_normalization(observed_intensities)
        predicted_normalized = SimilarityMetrics.unit_normalization(predicted_intensities)

        if isinstance(observed_normalized, scipy.sparse.csr_matrix):
            observed_normalized = observed_normalized.toarray()
        if isinstance(predicted_normalized, scipy.sparse.csr_matrix):
            predicted_normalized = predicted_normalized.toarray()

        diff_values = []
        for obs, pred in zip(observed_normalized, predicted_normalized):
            valid_ion_mask = pred > epsilon
            obs = obs[valid_ion_mask]
            pred = pred[valid_ion_mask]
            obs = obs[~np.isnan(obs)]
            pred = pred[~np.isnan(pred)]
            diff = chosen_metric(obs, pred)
            if np.isnan(diff):
                diff = 0
            diff_values.append(diff)

        return diff_values

    @staticmethod
    def calculate_quantiles(observed: np.ndarray, predicted: np.ndarray, quantile: str) -> float:
        """
        Helper function to calculcate quantiles.

        :param observed: observed intensities
        :param predicted: predicted intensities
        :param quantile: quantile method
        :return: calculated quantile
        """
        if quantile == "q3":
            return np.quantile(absolute(observed - mean(predicted)), 0.75)
        elif quantile == "q2":
            return np.quantile(absolute(observed - mean(predicted)), 0.5)
        else:
            return np.quantile(absolute(observed - mean(predicted)), 0.25)

    @staticmethod
    def modified_cosine(
        observed_intensities: Union[scipy.sparse.csr_matrix, np.ndarray],
        predicted_intensities: Union[scipy.sparse.csr_matrix, np.ndarray],
        observed_mz: Union[scipy.sparse.csr_matrix, np.ndarray],
        theoretical_mz: Union[scipy.sparse.csr_matrix, np.ndarray],
    ) -> List[float]:
        """
        Calculate modified cosine similarity as defined in Chris D. McGann et al. (Real-time spectral library \
        matching for sample multiplexed quantitative proteomics).

        :param observed_intensities: observed intensities, constants.EPSILON intensity indicates zero intensity peaks, \
                                     0 intensity indicates invalid peaks (charge state > peptide charge state or \
                                     position >= peptide length), array of length 174
        :param predicted_intensities: predicted intensities, see observed_intensities for details, array of length 174
        :param observed_mz: observed mz values
        :param theoretical_mz: theoretical mz values
        :return: calculates cosine values
        """
        epsilon = 1e-7
        observed_normalized = SimilarityMetrics.unit_normalization(observed_intensities)
        predicted_normalized = SimilarityMetrics.unit_normalization(predicted_intensities)

        if isinstance(observed_normalized, scipy.sparse.csr_matrix):
            observed_normalized = observed_normalized.toarray()
        if isinstance(predicted_normalized, scipy.sparse.csr_matrix):
            predicted_normalized = predicted_normalized.toarray()
        if isinstance(observed_mz, scipy.sparse.csr_matrix):
            observed_mz = observed_mz.toarray()
        if isinstance(theoretical_mz, scipy.sparse.csr_matrix):
            theoretical_mz = theoretical_mz.toarray()

        cos_values = []
        mz_power = 0.9
        intensity_power = 0.4
        for obs, pred, obs_mz, th_mz in zip(observed_normalized, predicted_normalized, observed_mz, theoretical_mz):
            valid_ion_mask = pred > epsilon
            obs = obs[valid_ion_mask]
            pred = pred[valid_ion_mask]
            obs_mz = obs_mz[valid_ion_mask]
            th_mz = th_mz[valid_ion_mask]
            obs = obs[~np.isnan(obs)]
            pred = pred[~np.isnan(pred)]
            obs_mz = obs_mz[~np.isnan(obs_mz)]
            th_mz = th_mz[~np.isnan(th_mz)]
            sum_matched = np.sum(
                (obs**intensity_power) * (obs_mz**mz_power) * (pred**intensity_power) * (th_mz**mz_power)
            )
            sqrt_sum_pred = (np.sum(((pred**intensity_power) * (th_mz**mz_power)) ** 2)) ** 0.5
            sqrt_sum_obs = (np.sum(((obs**intensity_power) * (obs_mz**mz_power)) ** 2)) ** 0.5
            cosine = sum_matched / (sqrt_sum_pred * sqrt_sum_obs)
            if np.isnan(cosine):
                cosine = 0
            cos_values.append(cosine)

        return cos_values

    def calc(self, all_features: bool, xl: bool = False):
        """
        Adds columns with spectral angle feature to metrics_val dataframe.

        :param all_features: if True, calculate all metrics
        :param xl: whether calculating for crosslinked or linear peptides
        """
        if xl:
            if self.true_intensities is not None and self.pred_intensities is not None:
                true_intensities_a = (
                    self.true_intensities[:, :348] if self.true_intensities.shape[1] >= 348 else self.true_intensities
                )
                true_intensities_b = self.true_intensities[:, 348:] if self.true_intensities.shape[1] >= 348 else None
                pred_intensities_a = (
                    self.pred_intensities[:, :348] if self.pred_intensities.shape[1] >= 348 else self.pred_intensities
                )
                pred_intensities_b = self.pred_intensities[:, 348:] if self.pred_intensities.shape[1] >= 348 else None

                if true_intensities_a is not None and pred_intensities_a is not None:
                    self.metrics_val["spectral_angle_a"] = SimilarityMetrics.spectral_angle(
                        true_intensities_a, pred_intensities_a, 0
                    )
                    self.metrics_val["pearson_corr_a"] = SimilarityMetrics.correlation(
                        true_intensities_a, pred_intensities_a, 0
                    )
                    if all_features:
                        self._calc_additional_metrics(true_intensities_a, pred_intensities_a, key_suffix="_a")

                if true_intensities_b is not None and pred_intensities_b is not None:
                    self.metrics_val["spectral_angle_b"] = SimilarityMetrics.spectral_angle(
                        true_intensities_b, pred_intensities_b, 0
                    )
                    self.metrics_val["pearson_corr_b"] = SimilarityMetrics.correlation(
                        true_intensities_b, pred_intensities_b, 0
                    )
                    if all_features:
                        self._calc_additional_metrics(true_intensities_b, pred_intensities_b, key_suffix="_b")

                if true_intensities_a is not None and true_intensities_b is not None:
                    self.metrics_val["spectral_angle"] = (
                        self.metrics_val["spectral_angle_a"] + self.metrics_val["spectral_angle_b"]
                    ) / 2

        else:
            if self.true_intensities is not None and self.pred_intensities is not None:
                self.metrics_val["spectral_angle"] = SimilarityMetrics.spectral_angle(
                    self.true_intensities, self.pred_intensities, 0
                )
                self.metrics_val["pearson_corr"] = SimilarityMetrics.correlation(
                    self.true_intensities, self.pred_intensities, 0, "pearson"
                )
                if all_features:
                    self._calc_additional_metrics(self.true_intensities, self.pred_intensities)

    def _calc_additional_metrics(
        self,
        true_intensities: Union[np.ndarray, scipy.sparse.spmatrix],
        pred_intensities: Union[np.ndarray, scipy.sparse.spmatrix],
        key_suffix: str = "",
    ):
        self.metrics_val[f"spectral_entropy_similarity{key_suffix}"] = SimilarityMetrics.spectral_entropy_similarity(
            true_intensities, pred_intensities
        )
        self.metrics_val[f"cos{key_suffix}"] = SimilarityMetrics.cos(true_intensities, pred_intensities)
        self.metrics_val[f"mean_abs_diff{key_suffix}"] = SimilarityMetrics.abs_diff(
            true_intensities, pred_intensities, "mean"
        )
        self.metrics_val[f"std_abs_diff{key_suffix}"] = SimilarityMetrics.abs_diff(
            true_intensities, pred_intensities, "std"
        )
        self.metrics_val[f"abs_diff_Q3{key_suffix}"] = SimilarityMetrics.abs_diff(
            true_intensities, pred_intensities, "q3"
        )
        self.metrics_val[f"abs_diff_Q2{key_suffix}"] = SimilarityMetrics.abs_diff(
            true_intensities, pred_intensities, "q2"
        )
        self.metrics_val[f"abs_diff_Q1{key_suffix}"] = SimilarityMetrics.abs_diff(
            true_intensities, pred_intensities, "q1"
        )
        self.metrics_val[f"min_abs_diff{key_suffix}"] = SimilarityMetrics.abs_diff(
            true_intensities, pred_intensities, "min"
        )
        self.metrics_val[f"max_abs_diff{key_suffix}"] = SimilarityMetrics.abs_diff(
            true_intensities, pred_intensities, "max"
        )
        self.metrics_val[f"mse{key_suffix}"] = SimilarityMetrics.abs_diff(true_intensities, pred_intensities, "mse")

        col_names_spectral_angle = [
            f"spectral_angle_{amount}_charge{key_suffix}" for amount in ["single", "double", "triple"]
        ] + [f"spectral_angle_b_ions{key_suffix}", f"spectral_angle_y_ions{key_suffix}"]
        col_names_pearson_corr = [
            f"pearson_corr_{amount}_charge{key_suffix}" for amount in ["single", "double", "triple"]
        ] + [
            f"pearson_corr_b_ions{key_suffix}",
            f"pearson_corr_y_ions{key_suffix}",
        ]
        col_names_spearman_corr = [
            f"spearman_corr_{amount}_charge{key_suffix}" for amount in ["single", "double", "triple"]
        ] + [f"spearman_corr_b_ions{key_suffix}", f"spearman_corr_y_ions{key_suffix}"]

        if key_suffix != "":
            # dirty fix, if the key_suffix is not "", that means we have XL mode.
            # TODO: fix self.mz for XL mode

            self.metrics_val[f"spearman_corr{key_suffix}"] = SimilarityMetrics.correlation(
                true_intensities, pred_intensities, 0, "spearman", xl=True
            )

            for i, col_name_spectral_angle in enumerate(col_names_spectral_angle):
                self.metrics_val[col_name_spectral_angle] = SimilarityMetrics.spectral_angle(
                    true_intensities, pred_intensities, i + 1, xl=True
                )

            for i, col_name_pearson_corr in enumerate(col_names_pearson_corr):
                self.metrics_val[col_name_pearson_corr] = SimilarityMetrics.correlation(
                    true_intensities, pred_intensities, i + 1, "pearson", xl=True
                )

            for i, col_name_spearman_corr in enumerate(col_names_spearman_corr):
                self.metrics_val[col_name_spearman_corr] = SimilarityMetrics.correlation(
                    true_intensities, pred_intensities, i + 1, "spearman", xl=True
                )
        else:
            self.metrics_val[f"spearman_corr{key_suffix}"] = SimilarityMetrics.correlation(
                true_intensities, pred_intensities, 0, "spearman"
            )

            for i, col_name_spectral_angle in enumerate(col_names_spectral_angle):
                self.metrics_val[col_name_spectral_angle] = SimilarityMetrics.spectral_angle(
                    true_intensities, pred_intensities, i + 1
                )

            for i, col_name_pearson_corr in enumerate(col_names_pearson_corr):
                self.metrics_val[col_name_pearson_corr] = SimilarityMetrics.correlation(
                    true_intensities, pred_intensities, i + 1, "pearson"
                )

            for i, col_name_spearman_corr in enumerate(col_names_spearman_corr):
                self.metrics_val[col_name_spearman_corr] = SimilarityMetrics.correlation(
                    true_intensities, pred_intensities, i + 1, "spearman"
                )

            self.metrics_val[f"modified_cosine{key_suffix}"] = SimilarityMetrics.modified_cosine(
                true_intensities, pred_intensities, self.mz, self.mz
            )
