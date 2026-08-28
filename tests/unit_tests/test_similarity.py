import numpy as np
import scipy.sparse

import spectrum_fundamentals.constants as constants
import spectrum_fundamentals.metrics.similarity as sim
from spectrum_fundamentals.metrics.metric import Metric


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


class TestSpectralAngleNoB1:
    """Tests for SA with b1 ions excluded, and for the layout-derived b1 mask."""

    @staticmethod
    def _metric(**kwargs) -> Metric:
        """Build a bare Metric just to reach b1_mask() (calc is abstract but unused here)."""
        return type("_M", (Metric,), {"calc": lambda self: None})(**kwargs)

    def test_b1_excluded(self):
        """A predicted-but-missing b1 must not drag the score down once masked."""
        z = constants.EPSILON
        # b1 predicted but not observed (idx 3) -- standard SA would penalise this
        observed = get_padded_array([1.0, 2.0, 3.0, z], padding_value=z)
        predicted = get_padded_array([1.0, 2.0, 3.0, 1.0], padding_value=z)
        sa = sim.SimilarityMetrics.spectral_angle(observed, predicted)[0]
        mask = self._metric().b1_mask()
        sa_no_b1 = sim.SimilarityMetrics.spectral_angle(observed, predicted, masks=mask)[0]
        assert sa_no_b1 > sa

    def test_non_b1_ions_unaffected(self):
        """Ions at positions other than b1 are not affected by the mask."""
        # values only at indices 0,1,2 (y1+1, y1+2, y1+3) -- no b1
        observed = get_padded_array([1.0, 2.0, 3.0])
        predicted = get_padded_array([1.0, 2.0, 3.0])
        mask = self._metric().b1_mask()
        sa = sim.SimilarityMetrics.spectral_angle(observed, predicted)[0]
        sa_no_b1 = sim.SimilarityMetrics.spectral_angle(observed, predicted, masks=mask)[0]
        np.testing.assert_almost_equal(sa_no_b1, sa)

    def test_default_layout_masks_b1_slots(self):
        """task="default": for pos: for ion in [y, b]: for charge in [1,2,3] -> b1 at 3,4,5."""
        mask = self._metric().b1_mask()
        assert mask.shape == (1, constants.VEC_LENGTH)
        np.testing.assert_array_equal(np.where(mask[0] == 0)[0], [3, 4, 5])

    def test_cms2_layout_masks_both_peptides(self):
        """cms2 doubles the vector, so the second peptide's b1 must be masked as well."""
        mask = self._metric(cms2=True).b1_mask()
        assert mask.shape == (1, 2 * constants.VEC_LENGTH)
        np.testing.assert_array_equal(np.where(mask[0] == 0)[0], [3, 4, 5, 177, 178, 179])

    def test_multifrag_layout_masks_the_real_b1(self):
        """Regression guard: multifrag is laid out by ION_DIC, where index 3:6 is NOT b1.

        ION_DIC is sorted by ion name, so slots 3-5 there are A-ions and b1 sits elsewhere
        entirely -- a hard-coded 3:6 mask silently zeroed the wrong fragments.
        """
        mask = self._metric(task="multifrag", featured_ions=["b", "y"]).b1_mask()
        assert mask.shape == (1, len(constants.ION_DIC))
        masked = [constants.ION_DIC.index[i] for i in np.where(mask[0] == 0)[0]]
        assert masked == ["b1"]


def get_padded_array(arr, padding_value: int = 0) -> np.ndarray:
    """Get padded array."""
    return np.array([np.pad(arr, (0, constants.VEC_LENGTH - len(arr)), "constant", constant_values=padding_value)])
