import numpy as np
import scipy.sparse

import spectrum_fundamentals.constants as constants
import spectrum_fundamentals.metrics.similarity as sim


class TestSpectralAngle:
    """Class to test SA."""

    def test_l2_norm(self):
        """Test l2 norm."""
        vector = get_padded_array([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_equal(sim.SimilarityMetrics.l2_norm(vector), np.sqrt(30))

    def test_spectral_angle_equal(self):
        """Test SA."""
        observed = get_padded_array([1.0, 2.0, 3.0, 4.0])
        predicted = get_padded_array([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_almost_equal(sim.SimilarityMetrics.spectral_angle(observed, predicted), 1.0)

    def test_spectral_angle_equal_scaled(self):
        """Test SA."""
        observed = get_padded_array([1.0, 2.0, 3.0, 4.0])
        predicted = get_padded_array([2.0, 4.0, 6.0, 8.0])
        np.testing.assert_almost_equal(sim.SimilarityMetrics.spectral_angle(observed, predicted), 1.0)

    def test_spectral_angle_zero(self):
        """Test SA."""
        z = constants.EPSILON
        observed = get_padded_array([z, 2.0, z, 4.0])
        predicted = get_padded_array([1.0, z, 3.0, z])
        np.testing.assert_almost_equal(sim.SimilarityMetrics.spectral_angle(observed, predicted), 0.0)

    def test_spectral_angle_all_zero(self):
        """Test SA."""
        z = constants.EPSILON
        observed = get_padded_array([z, z, z, z])
        predicted = get_padded_array([z, z, z, z])
        np.testing.assert_almost_equal(sim.SimilarityMetrics.spectral_angle(observed, predicted), 0.0)

    def test_spectral_angle_invalid(self):
        """Test SA."""
        observed = get_padded_array([0.0, 2.0, 0.0, 4.0])
        predicted = get_padded_array([1.0, 0.0, 3.0, 0.0])
        np.testing.assert_almost_equal(sim.SimilarityMetrics.spectral_angle(observed, predicted), 0.0)

    def test_spectral_angle_full(self):
        """Test SA."""
        # 1 - 2*arccos(28/30)/pi
        observed = get_padded_array([1.0, 2.0, 4.0, 3.0])
        predicted = get_padded_array([2.0, 1.0, 3.0, 4.0])
        np.testing.assert_almost_equal(sim.SimilarityMetrics.spectral_angle(observed, predicted), 0.76622811354)

    def test_spectral_angle_full_with_zeros(self):
        """Test SA."""
        # 1 - 2*arccos(24/sqrt(25*29))/pi
        z = constants.EPSILON
        observed = get_padded_array([z, 2.0, 4.0, 3.0])
        predicted = get_padded_array([2.0, z, 3.0, 4.0])
        np.testing.assert_almost_equal(sim.SimilarityMetrics.spectral_angle(observed, predicted), 0.70046462491)

    def test_spectral_angle_full_with_both_zeros(self):
        """Test SA."""
        # 1 - 2*arccos(12/sqrt(16*25))/pi
        z = constants.EPSILON
        observed = get_padded_array([z, 3.0, 4.0, z])
        predicted = get_padded_array([z, z, 3.0, 4.0])
        np.testing.assert_almost_equal(sim.SimilarityMetrics.spectral_angle(observed, predicted), 0.40966552939)


class TestSpectralEntropy:
    """Class to test spectral entropy."""

    def test_spectral_entropy_equal(self):
        """Test spectral entropy."""
        observed = get_padded_array([1.0, 2.0, 3.0, 4.0])
        predicted = get_padded_array([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_almost_equal(sim.SimilarityMetrics.spectral_entropy_similarity(observed, predicted), 1.0)

    def test_spectral_entropy_equal_scaled(self):
        """Test spectral entropy."""
        observed = get_padded_array([1.0, 2.0, 3.0, 4.0])
        predicted = get_padded_array([2.0, 4.0, 6.0, 8.0])
        np.testing.assert_almost_equal(sim.SimilarityMetrics.spectral_entropy_similarity(observed, predicted), 1.0)

    def test_spectral_entropy_all_zero(self):
        """Test spectral entropy."""
        z = constants.EPSILON
        observed = get_padded_array([z, z, z, z])
        predicted = get_padded_array([z, z, z, z])
        np.testing.assert_almost_equal(sim.SimilarityMetrics.spectral_entropy_similarity(observed, predicted), 1.0)

    def test_spectral_entropy_invalid(self):
        """Test spectral entropy."""
        observed = get_padded_array([0.0, 2.0, 0.0, 4.0])
        predicted = get_padded_array([1.0, 0.0, 3.0, 0.0])
        np.testing.assert_almost_equal(sim.SimilarityMetrics.spectral_entropy_similarity(observed, predicted), 0.0)

    def test_spectral_entropy_full(self):
        """Test spectral entropy."""
        # 1 - 2*arccos(28/30)/pi
        observed = get_padded_array([1.0, 2.0, 4.0, 3.0])
        predicted = get_padded_array([2.0, 1.0, 3.0, 4.0])
        np.testing.assert_almost_equal(
            sim.SimilarityMetrics.spectral_entropy_similarity(observed, predicted), 0.96514844
        )

    def test_spectral_entropy_full_with_zeros(self):
        """Test spectral entropy."""
        # 1 - 2*arccos(24/sqrt(25*29))/pi
        z = constants.EPSILON
        observed = get_padded_array([z, 2.0, 4.0, 3.0])
        predicted = get_padded_array([2.0, z, 3.0, 4.0])
        np.testing.assert_almost_equal(
            sim.SimilarityMetrics.spectral_entropy_similarity(observed, predicted), 0.83929633
        )

    def test_spectral_entropy_full_with_both_zeros(self):
        """Test spectral entropy."""
        # 1 - 2*arccos(12/sqrt(16*25))/pi
        z = constants.EPSILON
        observed = get_padded_array([z, 3.0, 4.0, z])
        predicted = get_padded_array([z, z, 3.0, 4.0])
        np.testing.assert_almost_equal(
            sim.SimilarityMetrics.spectral_entropy_similarity(observed, predicted), 0.5469540
        )


class TestModifiedCosine:
    """Class to test modified cosine."""

    def test_modified_cosine_equal(self):
        """Test modified cosine."""
        observed = get_padded_array([1.0, 2.0, 3.0, 4.0])
        predicted = get_padded_array([1.0, 2.0, 3.0, 4.0])
        observed_mz = get_padded_array([100.0, 200.0, 300.0, 400.0])
        theoretical_mz = get_padded_array([100.0, 200.0, 300.0, 400.0])
        np.testing.assert_almost_equal(
            sim.SimilarityMetrics.modified_cosine(observed, predicted, observed_mz, theoretical_mz), 1.0
        )

    def test_modified_cosine_equal_scaled(self):
        """Test modified cosine."""
        observed = get_padded_array([1.0, 2.0, 3.0, 4.0])
        predicted = get_padded_array([2.0, 4.0, 6.0, 8.0])
        observed_mz = get_padded_array([100.0, 200.0, 300.0, 400.0])
        theoretical_mz = get_padded_array([100.0, 200.0, 300.0, 400.0])
        np.testing.assert_almost_equal(
            sim.SimilarityMetrics.modified_cosine(observed, predicted, observed_mz, theoretical_mz), 1.0
        )

    def test_modified_cosine_zero(self):
        """Test modified cosine."""
        z = constants.EPSILON
        observed = get_padded_array([z, 2.0, z, 4.0])
        predicted = get_padded_array([1.0, z, 3.0, z])
        observed_mz = get_padded_array([0.0, 0.0, 0.0, 0.0])
        theoretical_mz = get_padded_array([100.0, 200.0, 300.0, 400.0])
        np.testing.assert_almost_equal(
            sim.SimilarityMetrics.modified_cosine(observed, predicted, observed_mz, theoretical_mz), 0.0
        )

    def test_modified_cosine_all_zero(self):
        """Test modified cosine."""
        z = constants.EPSILON
        observed = get_padded_array([z, z, z, z])
        predicted = get_padded_array([z, z, z, z])
        observed_mz = get_padded_array([0.0, 0.0, 0.0, 0.0])
        theoretical_mz = get_padded_array([100.0, 200.0, 300.0, 400.0])
        np.testing.assert_almost_equal(
            sim.SimilarityMetrics.modified_cosine(observed, predicted, observed_mz, theoretical_mz), 0.0
        )

    def test_modified_cosine_invalid(self):
        """Test modified cosine."""
        observed = get_padded_array([0.0, 2.0, 0.0, 4.0])
        predicted = get_padded_array([1.0, 0.0, 3.0, 0.0])
        observed_mz = get_padded_array([100.0, 200.0, 300.0, 400.0])
        theoretical_mz = get_padded_array([100.0, 200.0, 300.0, 400.0])
        np.testing.assert_almost_equal(
            sim.SimilarityMetrics.modified_cosine(observed, predicted, observed_mz, theoretical_mz), 0.0
        )

    def test_modified_cosine_full(self):
        """Test modified cosine."""
        observed = get_padded_array([1.0, 2.0, 4.0, 3.0])
        predicted = get_padded_array([2.0, 1.0, 3.0, 4.0])
        observed_mz = get_padded_array([100.0, 200.0, 300.0, 400.0])
        theoretical_mz = get_padded_array([100.0, 200.0, 300.0, 400.0])
        np.testing.assert_almost_equal(
            sim.SimilarityMetrics.modified_cosine(observed, predicted, observed_mz, theoretical_mz), 0.990263903
        )

    def test_modified_cosine_full_with_zeros(self):
        """Test modified cosine."""
        z = constants.EPSILON
        observed = get_padded_array([z, 2.0, 4.0, 3.0])
        predicted = get_padded_array([2.0, z, 3.0, 4.0])
        observed_mz = get_padded_array([100.0, 200.0, 300.0, 400.0])
        theoretical_mz = get_padded_array([100.0, 200.0, 300.0, 400.0])
        np.testing.assert_almost_equal(
            sim.SimilarityMetrics.modified_cosine(observed, predicted, observed_mz, theoretical_mz), 0.978272764
        )

    def test_modified_cosine_full_with_both_zeros(self):
        """Test modified cosine."""
        z = constants.EPSILON
        observed = get_padded_array([z, 3.0, 4.0, z])
        predicted = get_padded_array([z, z, 3.0, 4.0])
        observed_mz = get_padded_array([100.0, 200.0, 300.0, 400.0])
        theoretical_mz = get_padded_array([100.0, 200.0, 300.0, 400.0])
        np.testing.assert_almost_equal(
            sim.SimilarityMetrics.modified_cosine(observed, predicted, observed_mz, theoretical_mz), 0.56777188
        )

    def test_modified_cosine_full_different_mz(self):
        """Test modified cosine."""
        observed = get_padded_array([1.0, 2.0, 4.0, 3.0])
        predicted = get_padded_array([2.0, 1.0, 3.0, 4.0])
        observed_mz = get_padded_array([100.0, 200.0, 300.0, 400.0])
        theoretical_mz = get_padded_array([200.0, 100.0, 400.0, 300.0])
        np.testing.assert_almost_equal(
            sim.SimilarityMetrics.modified_cosine(observed, predicted, observed_mz, theoretical_mz), 0.95422659
        )


class TestSpectralAngleMultipleRows:
    """Class to test SA."""

    def test_l2_norm(self):
        """Test l2 norm."""
        vector1 = get_padded_array([1.0, 2.0, 3.0, 4.0])
        vector2 = get_padded_array([1.0, 2.0, 3.0, 5.0])
        matrix = np.concatenate((vector1, vector2), axis=0)
        np.testing.assert_equal(sim.SimilarityMetrics.l2_norm(matrix), np.array([np.sqrt(30), np.sqrt(39)]))

    def test_spectral_angle_full(self):
        """Test SA."""
        z = constants.EPSILON
        vector1 = get_padded_array([1.0, 2.0, 4.0, 3.0])
        vector2 = get_padded_array([z, 3.0, 4.0, z])
        vector3 = get_padded_array([z, z, z, z])
        vector4 = get_padded_array([0.0, 0.0, 0.0, 0.0])
        observed_matrix = np.concatenate((vector1, vector2, vector3, vector4), axis=0)

        vector1 = get_padded_array([2.0, 1.0, 3.0, 4.0])  # 1 - 2*arccos(28/30)/pi
        vector2 = get_padded_array([z, z, 3.0, 4.0])  # 1 - 2*arccos(12/25)/pi
        vector3 = get_padded_array([z, z, z, z])  # 0.0
        vector4 = get_padded_array([0.0, 0.0, 0.0, 0.0])  # 0.0
        predicted_matrix = np.concatenate((vector1, vector2, vector3, vector4), axis=0)

        np.testing.assert_almost_equal(
            sim.SimilarityMetrics.spectral_angle(observed_matrix, predicted_matrix),
            np.array([0.76622811354, 0.40966552939, 0.0, 0.0]),
        )

    def test_spectral_angle_sparse(self):
        """Test SA."""
        z = constants.EPSILON
        vector1 = get_padded_array([1.0, 2.0, 4.0, 3.0])
        vector2 = get_padded_array([z, 3.0, 4.0, z])
        vector3 = get_padded_array([z, z, z, z])
        vector4 = get_padded_array([0.0, 0.0, 0.0, 0.0])
        observed_matrix = np.concatenate((vector1, vector2, vector3, vector4), axis=0)

        vector1 = get_padded_array([2.0, 1.0, 3.0, 4.0])  # 1 - 2*arccos(28/30)/pi
        vector2 = get_padded_array([z, z, 3.0, 4.0])  # 1 - 2*arccos(12/25)/pi
        vector3 = get_padded_array([z, z, z, z])  # 0.0
        vector4 = get_padded_array([0.0, 0.0, 0.0, 0.0])  # 0.0
        predicted_matrix = np.concatenate((vector1, vector2, vector3, vector4), axis=0)

        observed_matrix = scipy.sparse.csr_matrix(observed_matrix)
        predicted_matrix = scipy.sparse.csr_matrix(predicted_matrix)
        np.testing.assert_almost_equal(
            sim.SimilarityMetrics.spectral_angle(observed_matrix, predicted_matrix),
            np.array([0.76622811354, 0.40966552939, 0.0, 0.0]),
        )


class TestSpectralAngleNoiseAware:
    """Tests for the noise-aware / detection-aware spectral angle (DA-SA)."""

    def test_disabled_by_default(self):
        """DA-SA must be off unless explicitly enabled via the environment."""
        assert sim._noise_aware_sa_config() is None

    def test_reduces_to_spectral_angle_when_all_detectable(self):
        """With tau -> 0 every fragment is detectable, so DA-SA == standard SA."""
        z = constants.EPSILON
        observed = get_padded_array([1.0, z, 4.0, 3.0], padding_value=z)
        predicted = get_padded_array([2.0, 1.0, 3.0, 4.0], padding_value=z)
        sa = sim.SimilarityMetrics.spectral_angle(observed, predicted)
        da = sim.SimilarityMetrics.spectral_angle_noise_aware(observed, predicted, tau=0.0, s=1e-6)
        np.testing.assert_almost_equal(da, sa)

    def test_missing_low_peak_relaxed(self):
        """Missing predicted peaks below the detectability threshold are not penalised."""
        z = constants.EPSILON
        # base peak 1.0 observed; four weak predicted peaks (<=0.25, all < tau) missing.
        observed = get_padded_array([1.0, z, z, z, z], padding_value=z)
        predicted = get_padded_array([1.0, 0.25, 0.22, 0.20, 0.18], padding_value=z)
        sa = sim.SimilarityMetrics.spectral_angle(observed, predicted)[0]
        # hard threshold above the weak peaks -> they are dropped -> no penalty at all.
        da_hard = sim.SimilarityMetrics.spectral_angle_noise_aware(observed, predicted, tau=0.30, mode="hard_pred")[0]
        # soft threshold -> penalty strongly reduced but not necessarily zero.
        da_soft = sim.SimilarityMetrics.spectral_angle_noise_aware(observed, predicted, tau=0.30, s=0.06)[0]
        assert sa < 0.8  # standard SA is dragged down by the four "missing" weak peaks
        np.testing.assert_almost_equal(da_hard, 1.0)
        assert da_soft > sa + 0.1

    def test_missing_high_peak_still_punished(self):
        """A missing high-intensity predicted peak is still penalised (≈ standard SA)."""
        z = constants.EPSILON
        observed = get_padded_array([1.0, z, 0.25, 0.22], padding_value=z)
        predicted = get_padded_array([1.0, 0.70, 0.25, 0.22], padding_value=z)
        sa = sim.SimilarityMetrics.spectral_angle(observed, predicted)[0]
        da = sim.SimilarityMetrics.spectral_angle_noise_aware(observed, predicted, tau=0.30, s=0.06)[0]
        np.testing.assert_almost_equal(da, sa, decimal=2)

    def test_no_observed_fragments_is_zero(self):
        """No observed fragments -> SA of 0, like the standard metric."""
        z = constants.EPSILON
        observed = get_padded_array([z, z, z, z], padding_value=z)
        predicted = get_padded_array([1.0, 0.5, 0.3, 0.2], padding_value=z)
        da = sim.SimilarityMetrics.spectral_angle_noise_aware(observed, predicted, tau=0.05)
        np.testing.assert_almost_equal(da, 0.0)


class TestSpectralAngleNoB1:
    """Tests for SA with b1 ions excluded."""

    def test_b1_excluded(self):
        """b1 ions (indices 3,4,5) should not affect the score even if predicted."""
        z = constants.EPSILON
        # b1 predicted but not observed (idx 3) — standard SA would penalise this
        observed = get_padded_array([1.0, 2.0, 3.0, z], padding_value=z)
        predicted = get_padded_array([1.0, 2.0, 3.0, 1.0], padding_value=z)
        sa = sim.SimilarityMetrics.spectral_angle(observed, predicted)[0]
        mask = np.ones((1, constants.VEC_LENGTH))
        mask[:, 3:6] = 0
        sa_no_b1 = sim.SimilarityMetrics.spectral_angle(observed, predicted, masks=mask)[0]
        # without b1 exclusion, missing b1 drags score down
        assert sa_no_b1 > sa

    def test_non_b1_ions_unaffected(self):
        """Ions at positions other than b1 are not affected by the mask."""
        # values only at indices 0,1,2 (y1+1, y1+2, y1+3) — no b1
        observed = get_padded_array([1.0, 2.0, 3.0])
        predicted = get_padded_array([1.0, 2.0, 3.0])
        mask = np.ones((1, constants.VEC_LENGTH))
        mask[:, 3:6] = 0
        sa = sim.SimilarityMetrics.spectral_angle(observed, predicted)[0]
        sa_no_b1 = sim.SimilarityMetrics.spectral_angle(observed, predicted, masks=mask)[0]
        np.testing.assert_almost_equal(sa_no_b1, sa)


class TestSpectralAngleThreshold:
    """Tests for the predicted-peak intensity threshold on the spectral angle."""

    def test_disabled_by_default(self):
        """The thresholded-SA feature must be off unless enabled via the environment."""
        assert sim._sa_threshold_config() is None

    def test_config_reads_env(self, monkeypatch):
        """When enabled, the config returns the configured predicted-intensity floor."""
        monkeypatch.setenv("SATHRESH_ENABLE", "1")
        monkeypatch.setenv("SATHRESH_TAU", "0.03")
        assert sim._sa_threshold_config() == 0.03

    def test_default_threshold_reduces_to_spectral_angle(self):
        """The default EPSILON threshold reproduces the standard spectral angle exactly."""
        z = constants.EPSILON
        observed = get_padded_array([1.0, z, 4.0, 3.0], padding_value=z)
        predicted = get_padded_array([2.0, 1.0, 3.0, 4.0], padding_value=z)
        base = sim.SimilarityMetrics.spectral_angle(observed, predicted)
        same = sim.SimilarityMetrics.spectral_angle(observed, predicted, predicted_threshold=constants.EPSILON)
        np.testing.assert_almost_equal(same, base)

    def test_low_predicted_peak_ignored(self):
        """A predicted peak below the threshold is dropped from BOTH vectors."""
        z = constants.EPSILON
        # base peak agrees perfectly; a weak predicted peak (0.10) coincides with a large,
        # disagreeing observed intensity -> drags the standard SA down.
        observed = get_padded_array([1.0, 0.9], padding_value=z)
        predicted = get_padded_array([1.0, 0.10], padding_value=z)
        sa = sim.SimilarityMetrics.spectral_angle(observed, predicted)[0]
        # threshold above the weak predicted peak -> its position is ignored -> only the
        # perfectly-agreeing base peak remains -> SA == 1.
        sa_thr = sim.SimilarityMetrics.spectral_angle(observed, predicted, predicted_threshold=0.30)[0]
        assert sa < 0.8
        np.testing.assert_almost_equal(sa_thr, 1.0)

    def test_matched_low_peak_also_dropped(self):
        """Unlike DA-SA, the threshold drops matched low peaks too (symmetric masking)."""
        z = constants.EPSILON
        # matched but disagreeing weak peak: obs 0.05 vs pred 0.10, below a 0.30 floor.
        observed = get_padded_array([1.0, 0.05], padding_value=z)
        predicted = get_padded_array([1.0, 0.10], padding_value=z)
        # DA-SA hard_pred keeps the matched low peak (observed contribution stays) -> < 1.
        da_hard = sim.SimilarityMetrics.spectral_angle_noise_aware(
            observed, predicted, tau=0.30, mode="hard_pred"
        )[0]
        # the threshold removes that position entirely -> only the base peak -> SA == 1.
        sa_thr = sim.SimilarityMetrics.spectral_angle(observed, predicted, predicted_threshold=0.30)[0]
        assert da_hard < 1.0
        np.testing.assert_almost_equal(sa_thr, 1.0)

    def test_no_common_fragments_is_zero(self):
        """If the threshold removes every predicted peak that had an observed match, SA -> 0."""
        z = constants.EPSILON
        # the only observed peak coincides with a weak predicted peak; the strong predicted
        # peak has no observed match. Thresholding out the weak one leaves no common fragment.
        observed = get_padded_array([z, 0.5], padding_value=z)
        predicted = get_padded_array([1.0, 0.10], padding_value=z)
        sa_thr = sim.SimilarityMetrics.spectral_angle(observed, predicted, predicted_threshold=0.30)
        np.testing.assert_almost_equal(sa_thr, 0.0)


class TestSpectralAngleObservedThreshold:
    """Tests for the measured (observed) noise floor on the spectral angle."""

    def test_disabled_by_default(self):
        """The observed-floor feature must be off unless enabled via the environment."""
        assert sim._sa_obs_threshold_config() is None

    def test_config_reads_env(self, monkeypatch):
        """When enabled, the config returns the configured observed-intensity floor."""
        monkeypatch.setenv("SAOBS_ENABLE", "1")
        monkeypatch.setenv("SAOBS_TAU", "0.04")
        assert sim._sa_obs_threshold_config() == 0.04

    def test_default_threshold_reduces_to_spectral_angle(self):
        """The default 0.0 observed floor reproduces the standard spectral angle exactly."""
        z = constants.EPSILON
        observed = get_padded_array([1.0, z, 4.0, 3.0], padding_value=z)
        predicted = get_padded_array([2.0, 1.0, 3.0, 4.0], padding_value=z)
        base = sim.SimilarityMetrics.spectral_angle(observed, predicted)
        same = sim.SimilarityMetrics.spectral_angle(observed, predicted, observed_threshold=0.0)
        np.testing.assert_almost_equal(same, base)

    def test_present_but_tiny_observed_dropped(self):
        """A present-but-tiny measured peak is dropped from BOTH vectors as noise."""
        z = constants.EPSILON
        # base peak agrees; the 2nd fragment is predicted 0.3 but only measured at 0.01 (noise).
        observed = get_padded_array([1.0, 0.01], padding_value=z)
        predicted = get_padded_array([1.0, 0.30], padding_value=z)
        sa = sim.SimilarityMetrics.spectral_angle(observed, predicted)[0]
        # floor above the tiny measured peak -> that position is denoised away -> only the base
        # peak remains, which agrees -> SA == 1.
        sa_obs = sim.SimilarityMetrics.spectral_angle(observed, predicted, observed_threshold=0.05)[0]
        assert sa < 0.9
        np.testing.assert_almost_equal(sa_obs, 1.0)

    def test_missing_peak_still_penalised(self):
        """A genuinely missing peak (no measured peak) is NOT forgiven by the observed floor."""
        z = constants.EPSILON
        observed = get_padded_array([1.0, z], padding_value=z)     # 2nd fragment truly absent
        predicted = get_padded_array([1.0, 0.90], padding_value=z)  # ...but strongly predicted
        sa = sim.SimilarityMetrics.spectral_angle(observed, predicted)[0]
        sa_obs = sim.SimilarityMetrics.spectral_angle(observed, predicted, observed_threshold=0.05)[0]
        assert sa < 1.0
        np.testing.assert_almost_equal(sa_obs, sa)  # unchanged: missing stays penalised


def get_padded_array(arr, padding_value: int = 0) -> np.ndarray:
    """Get padded array."""
    return np.array([np.pad(arr, (0, constants.VEC_LENGTH - len(arr)), "constant", constant_values=padding_value)])
