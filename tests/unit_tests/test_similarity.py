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

class TestModifiedCosine:
    """Class to test modified cosine."""

    def test_modified_cosine_equal(self):
        """Test SA."""
        observed = get_padded_array([1.0, 2.0, 3.0, 4.0])
        predicted = get_padded_array([1.0, 2.0, 3.0, 4.0])
        mz = get_padded_array([100.0, 200.0, 300.0, 400.0])
        np.testing.assert_almost_equal(sim.SimilarityMetrics.modified_cosine(observed, predicted, mz, mz), 1.0)

    def test_modified_cosine_equal_scaled(self):
        """Test SA."""
        observed = get_padded_array([1.0, 2.0, 3.0, 4.0])
        predicted = get_padded_array([2.0, 4.0, 6.0, 8.0])
        mz = get_padded_array([100.0, 200.0, 300.0, 400.0])
        np.testing.assert_almost_equal(sim.SimilarityMetrics.modified_cosine(observed, predicted, mz, mz), 1.0)

    def test_modified_cosine_zero(self):
        """Test SA."""
        z = constants.EPSILON
        observed = get_padded_array([z, 2.0, z, 4.0])
        predicted = get_padded_array([1.0, z, 3.0, z])
        mz = get_padded_array([100.0, 200.0, 300.0, 400.0])
        np.testing.assert_almost_equal(sim.SimilarityMetrics.modified_cosine(observed, predicted, mz, mz), 0.0)

    def test_modified_cosine_all_zero(self):
        """Test SA."""
        z = constants.EPSILON
        observed = get_padded_array([z, z, z, z])
        predicted = get_padded_array([z, z, z, z])
        mz = get_padded_array([100.0, 200.0, 300.0, 400.0])
        np.testing.assert_almost_equal(sim.SimilarityMetrics.modified_cosine(observed, predicted, mz, mz), 0.0)

    def test_modified_cosine_invalid(self):
        """Test SA."""
        observed = get_padded_array([0.0, 2.0, 0.0, 4.0])
        predicted = get_padded_array([1.0, 0.0, 3.0, 0.0])
        mz = get_padded_array([100.0, 200.0, 300.0, 400.0])
        np.testing.assert_almost_equal(sim.SimilarityMetrics.modified_cosine(observed, predicted, mz, mz), 0.0)

    def test_modified_cosine_full(self):
        """Test SA."""
        # 1 - 2*arccos(28/30)/pi
        observed = get_padded_array([1.0, 2.0, 4.0, 3.0])
        predicted = get_padded_array([2.0, 1.0, 3.0, 4.0])
        mz = get_padded_array([100.0, 200.0, 300.0, 400.0])
        np.testing.assert_almost_equal(sim.SimilarityMetrics.modified_cosine(observed, predicted, mz, mz), 0.76622811354)

    def test_modified_cosine_full_with_zeros(self):
        """Test SA."""
        # 1 - 2*arccos(24/sqrt(25*29))/pi
        z = constants.EPSILON
        observed = get_padded_array([z, 2.0, 4.0, 3.0])
        predicted = get_padded_array([2.0, z, 3.0, 4.0])
        mz = get_padded_array([100.0, 200.0, 300.0, 400.0])
        np.testing.assert_almost_equal(sim.SimilarityMetrics.modified_cosine(observed, predicted, mz, mz), 0.70046462491)

    def test_modified_cosine_full_with_both_zeros(self):
        """Test SA."""
        # 1 - 2*arccos(12/sqrt(16*25))/pi
        z = constants.EPSILON
        observed = get_padded_array([z, 3.0, 4.0, z])
        predicted = get_padded_array([z, z, 3.0, 4.0])
        mz = get_padded_array([100.0, 200.0, 300.0, 400.0])
        np.testing.assert_almost_equal(sim.SimilarityMetrics.modified_cosine(observed, predicted, mz, mz), 0.40966552939)


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


def get_padded_array(arr, padding_value: int = 0) -> np.ndarray:
    """Get padded array."""
    return np.array([np.pad(arr, (0, constants.VEC_LENGTH - len(arr)), "constant", constant_values=padding_value)])
