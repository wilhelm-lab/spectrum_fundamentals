import pytest
import numpy as np
import scipy.sparse

import fundamentals.metrics.similarity as sim
import fundamentals.constants as constants


class TestSpectralAngle:
    
    def test_l2_norm(self):
        vector = get_padded_array([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_equal(sim.SimilarityMetrics.l2_norm(vector), np.sqrt(30))
    
    def test_spectral_angle_equal(self):
        observed  = get_padded_array([1.0, 2.0, 3.0, 4.0])
        predicted = get_padded_array([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_almost_equal(sim.SimilarityMetrics.spectral_angle(observed, predicted), 1.0)
    
    def test_spectral_angle_equal_scaled(self):
        observed  = get_padded_array([1.0, 2.0, 3.0, 4.0])
        predicted = get_padded_array([2.0, 4.0, 6.0, 8.0])
        np.testing.assert_almost_equal(sim.SimilarityMetrics.spectral_angle(observed, predicted), 1.0)
    
    def test_spectral_angle_zero(self):
        z = constants.EPSILON
        observed  = get_padded_array([  z, 2.0,   z, 4.0])
        predicted = get_padded_array([1.0,   z, 3.0,   z])
        np.testing.assert_almost_equal(sim.SimilarityMetrics.spectral_angle(observed, predicted), 0.0)
    
    def test_spectral_angle_all_zero(self):
        z = constants.EPSILON
        observed  = get_padded_array([z,z,z,z])
        predicted = get_padded_array([z,z,z,z])
        np.testing.assert_almost_equal(sim.SimilarityMetrics.spectral_angle(observed, predicted), 0.0)
        
    def test_spectral_angle_invalid(self):
        observed  = get_padded_array([0.0, 2.0, 0.0, 4.0])
        predicted = get_padded_array([1.0, 0.0, 3.0, 0.0])
        np.testing.assert_almost_equal(sim.SimilarityMetrics.spectral_angle(observed, predicted), 0.0)
    
    def test_spectral_angle_full(self):
        # 1 - 2*arccos(28/30)/pi
        observed  = get_padded_array([1.0, 2.0, 4.0, 3.0])
        predicted = get_padded_array([2.0, 1.0, 3.0, 4.0])
        np.testing.assert_almost_equal(sim.SimilarityMetrics.spectral_angle(observed, predicted), 0.76622811354)
    
    def test_spectral_angle_full_with_zeros(self):
        # 1 - 2*arccos(24/sqrt(25*29))/pi
        z = constants.EPSILON
        observed  = get_padded_array([  z, 2.0, 4.0, 3.0])
        predicted = get_padded_array([2.0,   z, 3.0, 4.0])
        np.testing.assert_almost_equal(sim.SimilarityMetrics.spectral_angle(observed, predicted), 0.70046462491)

    def test_spectral_angle_full_with_both_zeros(self):
        # 1 - 2*arccos(12/sqrt(16*25))/pi
        z = constants.EPSILON
        observed  = get_padded_array([  z, 3.0, 4.0,   z])
        predicted = get_padded_array([  z,   z, 3.0, 4.0])
        np.testing.assert_almost_equal(sim.SimilarityMetrics.spectral_angle(observed, predicted), 0.40966552939)


class TestSpectralAngleMultipleRows:
    
    def test_l2_norm(self):
        vector1 = get_padded_array([1.0, 2.0, 3.0, 4.0])
        vector2 = get_padded_array([1.0, 2.0, 3.0, 5.0])
        matrix = np.concatenate((vector1, vector2), axis = 0)
        np.testing.assert_equal(sim.SimilarityMetrics.l2_norm(matrix), np.array([np.sqrt(30), np.sqrt(39)]))
    
    def test_spectral_angle_full(self):
        z = constants.EPSILON
        vector1 = get_padded_array([1.0, 2.0, 4.0, 3.0])
        vector2 = get_padded_array([  z, 3.0, 4.0,   z])
        vector3 = get_padded_array([  z,   z,   z,   z])
        vector4 = get_padded_array([0.0, 0.0, 0.0, 0.0])
        observed_matrix = np.concatenate((vector1, vector2, vector3, vector4), axis = 0)
        
        vector1 = get_padded_array([2.0, 1.0, 3.0, 4.0]) # 1 - 2*arccos(28/30)/pi
        vector2 = get_padded_array([  z,   z, 3.0, 4.0]) # 1 - 2*arccos(12/25)/pi
        vector3 = get_padded_array([  z,   z,   z,   z]) # 0.0
        vector4 = get_padded_array([0.0, 0.0, 0.0, 0.0]) # 0.0
        predicted_matrix = np.concatenate((vector1, vector2, vector3, vector4), axis = 0)
        
        np.testing.assert_almost_equal(sim.SimilarityMetrics.spectral_angle(observed_matrix, predicted_matrix), np.array([0.76622811354, 0.40966552939, 0.0, 0.0]))
    
    def test_spectral_angle_sparse(self):
        z = constants.EPSILON
        vector1 = get_padded_array([1.0, 2.0, 4.0, 3.0])
        vector2 = get_padded_array([  z, 3.0, 4.0,   z])
        vector3 = get_padded_array([  z,   z,   z,   z])
        vector4 = get_padded_array([0.0, 0.0, 0.0, 0.0])
        observed_matrix = np.concatenate((vector1, vector2, vector3, vector4), axis = 0)
        
        vector1 = get_padded_array([2.0, 1.0, 3.0, 4.0]) # 1 - 2*arccos(28/30)/pi
        vector2 = get_padded_array([  z,   z, 3.0, 4.0]) # 1 - 2*arccos(12/25)/pi
        vector3 = get_padded_array([  z,   z,   z,   z]) # 0.0
        vector4 = get_padded_array([0.0, 0.0, 0.0, 0.0]) # 0.0
        predicted_matrix = np.concatenate((vector1, vector2, vector3, vector4), axis = 0)
        
        observed_matrix = scipy.sparse.csr_matrix(observed_matrix)
        predicted_matrix = scipy.sparse.csr_matrix(predicted_matrix)
        np.testing.assert_almost_equal(sim.SimilarityMetrics.spectral_angle(observed_matrix, predicted_matrix), np.array([0.76622811354, 0.40966552939, 0.0, 0.0]))
    
    
        
def get_padded_array(l, padding_value = 0):
    return np.array([np.pad(l, (0, constants.VEC_LENGTH - len(l)), 'constant', constant_values=padding_value)])
