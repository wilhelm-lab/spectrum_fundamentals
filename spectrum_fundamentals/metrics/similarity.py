import os

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import scipy.stats
from numpy import absolute, mean, std
from scipy import spatial
from sklearn.metrics import mean_squared_error

from spectrum_fundamentals import constants
from spectrum_fundamentals.metrics.metric import Metric


def _noise_aware_sa_config() -> dict | None:
    """
    Read noise-aware ("detection-aware") spectral-angle settings from the environment.

    Returns ``None`` unless ``DASA_ENABLE`` is truthy, so the default behaviour of the
    package is completely unchanged (no extra feature column is added). When enabled,
    a ``spectral_angle_noise_aware`` feature is emitted alongside ``spectral_angle``.
    See :func:`SimilarityMetrics.spectral_angle_noise_aware` and
    ``mp26_single_cell_proteomics/notes/noise_aware_spectral_angle_proposal.md``.

    Env vars: ``DASA_ENABLE`` (0/1), ``DASA_MODE`` (soft_pred|hard_pred|noise_floor),
    ``DASA_TAU``, ``DASA_S``, ``DASA_BETA``.

    :return: dict of kwargs for ``spectral_angle_noise_aware``, or ``None`` if disabled
    """
    if os.environ.get("DASA_ENABLE", "0").strip().lower() not in ("1", "true", "yes", "on"):
        return None
    return {
        "mode": os.environ.get("DASA_MODE", "soft_pred"),
        "tau": float(os.environ.get("DASA_TAU", "0.05")),
        "s": float(os.environ.get("DASA_S", "0.02")),
        "beta": float(os.environ.get("DASA_BETA", "1.0")),
    }


def _sa_threshold_config() -> float | None:
    """
    Read the predicted-peak intensity floor for a thresholded spectral angle from the environment.

    Predicted intensities are base-peak normalized (max = 1), so the returned value is a fraction
    of the base peak below which a predicted fragment is treated as being within the noise floor and
    is **excluded entirely** from the spectral-angle computation (its position is dropped from both
    the observed and the predicted vector before L2-normalization). This differs from the noise-aware
    SA (:func:`SimilarityMetrics.spectral_angle_noise_aware`), which only discounts *missing* weak
    predicted peaks but keeps matched ones.

    Returns ``None`` unless ``SATHRESH_ENABLE`` is truthy, so the default behaviour of the package is
    completely unchanged (no extra feature column, standard ``spectral_angle`` untouched). When
    enabled, a ``spectral_angle_thresh`` feature is emitted alongside ``spectral_angle``.

    Env vars: ``SATHRESH_ENABLE`` (0/1), ``SATHRESH_TAU`` (default 0.02).

    :return: the predicted-intensity threshold, or ``None`` if disabled
    """
    if os.environ.get("SATHRESH_ENABLE", "0").strip().lower() not in ("1", "true", "yes", "on"):
        return None
    return float(os.environ.get("SATHRESH_TAU", "0.02"))


def _sa_obs_threshold_config() -> float | None:
    """Read the measured (observed) noise floor for a denoised spectral angle from the environment.

    Unlike the predicted-peak threshold (``SATHRESH``, keyed on Prosit's predicted intensity), this
    thresholds the OBSERVED intensity — the physically-motivated "noise floor" axis: a matched
    fragment whose *measured* intensity is present but at/below ``SAOBS_TAU`` (a fraction of the base
    observed peak, since matched intensities are base-peak normalized in annotation.py) is treated as
    noise and dropped from both vectors. Genuinely missing peaks (intensity == ``constants.EPSILON``,
    i.e. no measured peak at all) are left in and remain penalized, so this denoises without trivially
    forgiving absent fragments.

    Returns ``None`` unless ``SAOBS_ENABLE`` is truthy, so the default behaviour is unchanged. When
    enabled, a ``spectral_angle_obsthresh`` feature is emitted alongside ``spectral_angle``.

    Env vars: ``SAOBS_ENABLE`` (0/1), ``SAOBS_TAU`` (default 0.02).

    :return: the observed-intensity noise floor, or ``None`` if disabled
    """
    if os.environ.get("SAOBS_ENABLE", "0").strip().lower() not in ("1", "true", "yes", "on"):
        return None
    return float(os.environ.get("SAOBS_TAU", "0.02"))


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
        observed_intensities: scipy.sparse.csr_matrix | np.ndarray,
        predicted_intensities: scipy.sparse.csr_matrix | np.ndarray,
        charge: int = 0,
        masks: np.ndarray | dict[int, np.ndarray] | None = None,
        predicted_threshold: float = constants.EPSILON,
        observed_threshold: float = 0.0,
    ) -> np.ndarray:
        """
        Calculate spectral angle.

        :param observed_intensities: observed intensities, constants.EPSILON intensity indicates zero intensity peaks, \
                                     0 intensity indicates invalid peaks (charge state > peptide charge state or \
                                     position >= peptide length), array of length 174
        :param predicted_intensities: predicted intensities, see observed_intensities for details, array of length 174
        :param charge: to filter by the peak charges, 0 means everything
        :param masks: masks of array for calculation
        :param predicted_threshold: predicted-intensity floor (fraction of the base peak); predicted peaks at or \
                                    below it are ignored entirely, i.e. dropped from both vectors before \
                                    normalization. Defaults to ``constants.EPSILON`` (only true zeros are dropped), \
                                    which reproduces the standard spectral angle exactly.
        :param observed_threshold: measured (observed) noise floor (fraction of the base peak); a matched fragment \
                                   whose observed intensity is present but at or below it is treated as noise and \
                                   dropped from both vectors, while genuinely missing peaks (== ``constants.EPSILON``) \
                                   stay in and remain penalized. Defaults to ``0.0`` (disabled).
        :raises ValueError: if charge is smaller than 1 or larger than 3
        :return: SA values
        """
        if masks is not None:
            if charge == 0:
                mask = masks
            else:
                if not 1 <= charge <= 3:
                    raise ValueError("Charge must be between 1 to 3.")
                mask = masks[charge]

            mask_csr = scipy.sparse.csr_matrix(mask)

            obs_csr = scipy.sparse.csr_matrix(observed_intensities)
            pred_csr = scipy.sparse.csr_matrix(predicted_intensities)

            observed_intensities = obs_csr.multiply(mask_csr).toarray()
            predicted_intensities = pred_csr.multiply(mask_csr).toarray()
        else:
            # ensure ndarray for downstream boolean ops/multiplication
            if isinstance(observed_intensities, scipy.sparse.csr_matrix):
                observed_intensities = observed_intensities.toarray()
            if isinstance(predicted_intensities, scipy.sparse.csr_matrix):
                predicted_intensities = predicted_intensities.toarray()

        predicted_non_zero_mask = predicted_intensities > predicted_threshold
        keep_mask = predicted_non_zero_mask
        if observed_threshold > 0.0:
            # measured noise floor: drop matched-but-tiny OBSERVED peaks (present but at/below the
            # floor) as noise; genuinely missing peaks (intensity == EPSILON) are NOT > EPSILON, so
            # they stay in and remain penalized -- this denoises without forgiving absent fragments.
            observed_noise = (observed_intensities > constants.EPSILON) & (
                observed_intensities <= observed_threshold
            )
            keep_mask = predicted_non_zero_mask & ~observed_noise
        observed_masked = np.multiply(observed_intensities, keep_mask)
        predicted_masked = np.multiply(predicted_intensities, keep_mask)

        observed_normalized = SimilarityMetrics.unit_normalization(observed_masked)
        predicted_normalized = SimilarityMetrics.unit_normalization(predicted_masked)

        observed_non_zero_mask = observed_intensities > constants.EPSILON
        fragments_in_common = SimilarityMetrics.rowwise_dot_product(observed_non_zero_mask, keep_mask)

        dot_product = SimilarityMetrics.rowwise_dot_product(observed_normalized, predicted_normalized) * (
            fragments_in_common > 0
        )

        arccos = np.arccos(dot_product)
        sa = 1 - 2 * arccos / np.pi
        sa = np.nan_to_num(sa)
        return sa

    @staticmethod
    def spectral_angle_noise_aware(
        observed_intensities: scipy.sparse.csr_matrix | np.ndarray,
        predicted_intensities: scipy.sparse.csr_matrix | np.ndarray,
        tau: float = 0.05,
        s: float = 0.02,
        mode: str = "soft_pred",
        beta: float = 1.0,
    ) -> np.ndarray:
        """
        Calculate a noise-aware ("detection-aware") spectral angle (DA-SA).

        Identical to :func:`spectral_angle` except that a predicted fragment which is
        *not observed* is discounted from the comparison in proportion to its
        detectability ``d in [0, 1]``. In low-input (single-cell) spectra, weak
        predicted peaks are expected to fall within noise, so their absence is barely
        penalised, while strong predicted peaks are penalised as usual. The metric
        reduces exactly to :func:`spectral_angle` when every fragment is detectable
        (``d = 1``). Design notes:
        ``mp26_single_cell_proteomics/notes/noise_aware_spectral_angle_proposal.md``.

        :param observed_intensities: observed intensities, constants.EPSILON indicates a \
                                     missing (valid but unobserved) peak, 0 indicates an \
                                     invalid peak, array of shape (n, 174)
        :param predicted_intensities: predicted intensities, same shape/encoding
        :param tau: detectability threshold on predicted relative intensity (soft_pred/hard_pred)
        :param s: softness of the sigmoid transition (soft_pred)
        :param mode: "soft_pred" (sigmoid in predicted intensity), "hard_pred" (step at tau), \
                     or "noise_floor" (per-spectrum noise floor on the expected intensity)
        :param beta: transition width relative to the noise floor (noise_floor mode)
        :raises ValueError: if mode is unknown
        :return: noise-aware SA values, array of shape (n,)
        """
        if isinstance(observed_intensities, scipy.sparse.csr_matrix):
            observed_intensities = observed_intensities.toarray()
        if isinstance(predicted_intensities, scipy.sparse.csr_matrix):
            predicted_intensities = predicted_intensities.toarray()
        observed = np.asarray(observed_intensities, dtype=float)
        predicted = np.asarray(predicted_intensities, dtype=float)

        pred_pos = predicted > constants.EPSILON  # fragments Prosit predicts
        obs_pos = observed > constants.EPSILON  # fragments actually observed
        matched = pred_pos & obs_pos  # predicted AND observed
        missing = pred_pos & ~obs_pos  # predicted but missing (the relaxed case)

        # Per-fragment detectability; only applied to missing peaks below.
        if mode == "hard_pred":
            d = (predicted >= tau).astype(float)
        elif mode == "soft_pred":
            d = 1.0 / (1.0 + np.exp(-(predicted - tau) / s))
        elif mode == "noise_floor":
            # LS scale A per spectrum so that on matched peaks observed ~= A * predicted.
            num = np.sum(observed * matched * predicted, axis=1)
            den = np.sum((predicted * matched) ** 2, axis=1)
            scale = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
            expected = scale[:, np.newaxis] * predicted
            obs_for_floor = np.where(matched, observed, np.nan)
            has_match = matched.any(axis=1)
            nu = np.full(observed.shape[0], constants.EPSILON)
            if has_match.any():  # nanquantile warns on all-NaN rows; only feed valid rows
                nu[has_match] = np.nanquantile(obs_for_floor[has_match], 0.10, axis=1)
            nu = nu[:, np.newaxis]
            d = 1.0 / (1.0 + np.exp(-(expected - nu) / (beta * nu + 1e-12)))
        else:
            raise ValueError(f"Unknown noise-aware SA mode {mode}")

        # Effective predicted norm: matched peaks count fully, missing peaks count by d.
        p2 = predicted**2
        eff_pred_norm = np.sqrt(np.sum(p2 * matched, axis=1) + np.sum(d * p2 * missing, axis=1))
        obs_norm = np.sqrt(np.sum((observed * matched) ** 2, axis=1))
        dot = np.sum(observed * matched * predicted, axis=1)

        denom = obs_norm * eff_pred_norm
        cos = np.divide(dot, denom, out=np.zeros_like(dot), where=denom > 0)
        cos = np.clip(cos, 0.0, 1.0) * (np.sum(matched, axis=1) > 0)

        sa = 1 - 2 * np.arccos(cos) / np.pi
        return np.nan_to_num(sa)

    @staticmethod
    def l2_norm(matrix) -> np.ndarray:
        """
        Compute the l2-norm (sqrt(sum(x^2) ) for each row of the matrix.

        :param matrix: matrix with intensities, constants.EPSILON intensity indicates zero intensity peaks,
                       0 intensity indicates invalid peaks (charge state > peptide charge state or
                       position >= peptide length), matrix of size (nspectra, 174)
        :return: vector with rowwise norms of the matrix
        """
        # = np.sqrt(np.sum(np.square(matrix), axis=0))
        if scipy.sparse.issparse(matrix):
            return scipy.sparse.linalg.norm(matrix, axis=1)
        else:
            return np.linalg.norm(matrix, axis=1)

    @staticmethod
    def unit_normalization(
        matrix: scipy.sparse.csr_matrix | np.ndarray,
    ) -> scipy.sparse.csr_matrix | np.ndarray:
        """
        Normalize each row of the matrix such that the norm equals 1.0.

        :param matrix: matrix with intensities, constants.EPSILON intensity indicates zero intensity peaks,
                       0 intensity indicates invalid peaks (charge state > peptide charge state or
                       position >= peptide length), matrix of size (nspectra, 174)
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
        observed_intensities: scipy.sparse.csr_matrix | np.ndarray,
        predicted_intensities: scipy.sparse.csr_matrix | np.ndarray,
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
        observed_intensities: scipy.sparse.csr_matrix | np.ndarray,
        predicted_intensities: scipy.sparse.csr_matrix | np.ndarray,
    ) -> list[float]:
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
        for obs, pred in zip(observed_intensities, predicted_intensities, strict=False):
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
        masks: np.ndarray | dict[int, np.ndarray] | None = None,
    ) -> list[float]:
        """
        Calculate correlation between observed and predicted.

        :param observed_intensities: observed intensities, constants.EPSILON intensity indicates zero intensity peaks, \
                                     0 intensity indicates invalid peaks (charge state > peptide charge state or \
                                     position >= peptide length), array of length 174
        :param predicted_intensities: predicted intensities, see observed_intensities for details, array of length 174
        :param charge: to filter by the peak charges, 0 means everything
        :param method: either pearson or spearman
        :param masks: charge mask
        :raises ValueError: if charge is smaller than 1 or larger than 3

        :return: calculated correlations
        """
        observed_intensities_array = observed_intensities.toarray()
        predicted_intensities_array = predicted_intensities.toarray()

        if masks is not None:
            if charge == 0:
                mask = masks  # full mask
            else:
                if not 1 <= charge <= 3:
                    raise ValueError("Charge must be between 1 to 3.")
                mask = masks[charge]  # charge-specific mask

            mask_csr = scipy.sparse.csr_matrix(mask)
            observed_intensities_array = observed_intensities.multiply(mask_csr).toarray()
            predicted_intensities_array = predicted_intensities.multiply(mask_csr).toarray()

        pear_corr = []
        for obs, pred in zip(observed_intensities_array, predicted_intensities_array, strict=False):
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
    ) -> list[float]:
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
        for obs, pred in zip(observed_normalized, predicted_normalized, strict=False):
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
    ) -> list[float]:
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
        for obs, pred in zip(observed_normalized, predicted_normalized, strict=False):
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
        observed_intensities: scipy.sparse.csr_matrix | np.ndarray,
        predicted_intensities: scipy.sparse.csr_matrix | np.ndarray,
        observed_mz: scipy.sparse.csr_matrix | np.ndarray,
        theoretical_mz: scipy.sparse.csr_matrix | np.ndarray,
    ) -> list[float]:
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
        for obs, pred, obs_mz, th_mz in zip(
            observed_normalized, predicted_normalized, observed_mz, theoretical_mz, strict=False
        ):
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

    def calc(self):  # noqa: C901
        """Adds columns with spectral angle feature to metrics_val dataframe."""
        if self.xl:
            if self.true_intensities is not None and self.pred_intensities is not None:
                true_intensities_a = (
                    self.true_intensities[:, : self.max_length]
                    if self.true_intensities.shape[1] >= self.max_length
                    else self.true_intensities
                )
                true_intensities_b = (
                    self.true_intensities[:, self.max_length :]
                    if self.true_intensities.shape[1] >= self.max_length
                    else None
                )
                pred_intensities_a = (
                    self.pred_intensities[:, : self.max_length]
                    if self.pred_intensities.shape[1] >= self.max_length
                    else self.pred_intensities
                )
                pred_intensities_b = (
                    self.pred_intensities[:, self.max_length :]
                    if self.pred_intensities.shape[1] >= self.max_length
                    else None
                )

                if true_intensities_a is not None and pred_intensities_a is not None:
                    self.metrics_val["spectral_angle_a"] = SimilarityMetrics.spectral_angle(
                        true_intensities_a, pred_intensities_a, 0
                    )
                    self.metrics_val["pearson_corr_a"] = SimilarityMetrics.correlation(
                        true_intensities_a, pred_intensities_a, 0
                    )
                    if self.all_features_flag:
                        self._calc_additional_metrics(true_intensities_a, pred_intensities_a, key_suffix="_a")

                if true_intensities_b is not None and pred_intensities_b is not None:
                    self.metrics_val["spectral_angle_b"] = SimilarityMetrics.spectral_angle(
                        true_intensities_b, pred_intensities_b, 0
                    )
                    self.metrics_val["pearson_corr_b"] = SimilarityMetrics.correlation(
                        true_intensities_b, pred_intensities_b, 0
                    )
                    if self.all_features_flag:
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
                dasa_config = _noise_aware_sa_config()
                if dasa_config is not None:
                    self.metrics_val["spectral_angle_noise_aware"] = SimilarityMetrics.spectral_angle_noise_aware(
                        self.true_intensities, self.pred_intensities, **dasa_config
                    )
                # b1 ions are thermodynamically unstable under HCD and rarely observed.
                # Excluding them avoids unfairly penalising PSMs for their absence.
                # Vector layout: for pos in [1..29]: for ion in [y,b]: for charge in [1,2,3]
                # => b1+1=idx3, b1+2=idx4, b1+3=idx5
                b1_exclusion_mask = np.ones((1, self.true_intensities.shape[1]))
                b1_exclusion_mask[:, 3:6] = 0
                self.metrics_val["spectral_angle_no_b1"] = SimilarityMetrics.spectral_angle(
                    self.true_intensities, self.pred_intensities, 0, masks=b1_exclusion_mask
                )
                sa_threshold = _sa_threshold_config()
                if sa_threshold is not None:
                    self.metrics_val["spectral_angle_thresh"] = SimilarityMetrics.spectral_angle(
                        self.true_intensities, self.pred_intensities, 0, predicted_threshold=sa_threshold
                    )
                sa_obs_threshold = _sa_obs_threshold_config()
                if sa_obs_threshold is not None:
                    self.metrics_val["spectral_angle_obsthresh"] = SimilarityMetrics.spectral_angle(
                        self.true_intensities, self.pred_intensities, 0, observed_threshold=sa_obs_threshold
                    )
                self.metrics_val["pearson_corr"] = SimilarityMetrics.correlation(
                    self.true_intensities, self.pred_intensities, 0, "pearson"
                )
                if self.all_features_flag:
                    self._calc_additional_metrics(self.true_intensities, self.pred_intensities)

    def _calc_additional_metrics(
        self,
        true_intensities: np.ndarray | scipy.sparse.spmatrix,
        pred_intensities: np.ndarray | scipy.sparse.spmatrix,
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

        self.metrics_val[f"spearman_corr{key_suffix}"] = SimilarityMetrics.correlation(
            true_intensities, pred_intensities, 0, "spearman"
        )

        amounts = ["single", "double", "triple"]
        for i, amount in enumerate(amounts, start=1):
            self.metrics_val[f"spectral_angle_{amount}_charge{key_suffix}"] = SimilarityMetrics.spectral_angle(
                true_intensities, pred_intensities, i, self.mask_dict
            )

            self.metrics_val[f"pearson_corr_{amount}_charge{key_suffix}"] = SimilarityMetrics.correlation(
                true_intensities, pred_intensities, i, "pearson", self.mask_dict
            )

            self.metrics_val[f"spearman_corr_{amount}_charge{key_suffix}"] = SimilarityMetrics.correlation(
                true_intensities, pred_intensities, i, "spearman", self.mask_dict
            )

        for ion, mask in self.ion_mask.items():
            self.metrics_val[f"spectral_angle_{ion}_ions{key_suffix}"] = SimilarityMetrics.spectral_angle(
                true_intensities, pred_intensities, 0, mask
            )
            self.metrics_val[f"pearson_corr_{ion}_ions{key_suffix}"] = SimilarityMetrics.correlation(
                true_intensities, pred_intensities, 0, "pearson", mask
            )
            self.metrics_val[f"spearman_corr_{ion}_ions{key_suffix}"] = SimilarityMetrics.correlation(
                true_intensities, pred_intensities, 0, "spearman", mask
            )

        if key_suffix == "":
            # TODO: From previous integration of XL
            # dirty fix, if the key_suffix is not "", that means we have XL mode.
            # TODO: fix self.mz for XL mode
            self.metrics_val[f"modified_cosine{key_suffix}"] = SimilarityMetrics.modified_cosine(
                true_intensities, pred_intensities, self.mz, self.mz
            )
