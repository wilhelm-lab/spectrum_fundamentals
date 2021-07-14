import pytest
import numpy as np

import fundamentals.metrics.similarity as sim
import fundamentals.constants as constants


class TestSpectralAngle:
    
    def test_l2_norm(self):
        vector = get_padded_array([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_equal(sim.SimilarityMetrics.l2_norm(vector), np.sqrt(30))
    
    def test_spectral_angle_equal(self):
        vector1 = get_padded_array([1.0, 2.0, 3.0, 4.0])
        vector2 = get_padded_array([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_almost_equal(sim.SimilarityMetrics.spectral_angle(vector1, vector2), 1.0)
    
    def test_spectral_angle_equal_scaled(self):
        vector1 = get_padded_array([1.0, 2.0, 3.0, 4.0])
        vector2 = get_padded_array([2.0, 4.0, 6.0, 8.0])
        np.testing.assert_almost_equal(sim.SimilarityMetrics.spectral_angle(vector1, vector2), 1.0)
    
    def test_spectral_angle_zero(self):
        z = constants.EPSILON
        vector1 = get_padded_array([  z, 2.0,   z, 4.0])
        vector2 = get_padded_array([1.0,   z, 3.0,   z])
        np.testing.assert_almost_equal(sim.SimilarityMetrics.spectral_angle(vector1, vector2), 0.0)
    
    def test_spectral_angle_all_zero(self):
        z = constants.EPSILON
        vector1 = get_padded_array([z,z,z,z])
        vector2 = get_padded_array([z,z,z,z])
        np.testing.assert_almost_equal(sim.SimilarityMetrics.spectral_angle(vector1, vector2), 0.0)
        
    def test_spectral_angle_invalid(self):
        vector1 = get_padded_array([0.0, 2.0, 0.0, 4.0])
        vector2 = get_padded_array([1.0, 0.0, 3.0, 0.0])
        np.testing.assert_almost_equal(sim.SimilarityMetrics.spectral_angle(vector1, vector2), 0.0)
    
    def test_spectral_angle_full(self):
        # 1 - 2*arccos(28/30)/pi
        vector1 = get_padded_array([1.0, 2.0, 4.0, 3.0])
        vector2 = get_padded_array([2.0, 1.0, 3.0, 4.0])
        np.testing.assert_almost_equal(sim.SimilarityMetrics.spectral_angle(vector1, vector2), 0.76622811354)
    
    def test_spectral_angle_full_with_zeros(self):
        # 1 - 2*arccos(24/29)/pi
        z = constants.EPSILON
        vector1 = get_padded_array([  z, 2.0, 4.0, 3.0])
        vector2 = get_padded_array([2.0,   z, 3.0, 4.0])
        np.testing.assert_almost_equal(sim.SimilarityMetrics.spectral_angle(vector1, vector2), 0.62057306283)

    def test_spectral_angle_full_with_both_zeros(self):
        # 1 - 2*arccos(12/25)/pi
        z = constants.EPSILON
        vector1 = get_padded_array([  z, 3.0, 4.0,   z])
        vector2 = get_padded_array([  z,   z, 3.0, 4.0])
        np.testing.assert_almost_equal(sim.SimilarityMetrics.spectral_angle(vector1, vector2), 0.31872668904)


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
        matrix1 = np.concatenate((vector1, vector2, vector3, vector4), axis = 0)
        
        vector1 = get_padded_array([2.0, 1.0, 3.0, 4.0]) # 1 - 2*arccos(28/30)/pi
        vector2 = get_padded_array([  z,   z, 3.0, 4.0]) # 1 - 2*arccos(12/25)/pi
        vector3 = get_padded_array([  z,   z,   z,   z]) # 0.0
        vector4 = get_padded_array([0.0, 0.0, 0.0, 0.0]) # 0.0
        matrix2 = np.concatenate((vector1, vector2, vector3, vector4), axis = 0)
        
        np.testing.assert_almost_equal(sim.SimilarityMetrics.spectral_angle(matrix1, matrix2), np.array([0.76622811354, 0.31872668904, 0.0, 0.0]))
    
        
def get_padded_array(l, padding_value = 0):
    return np.array([np.pad(l, (0, constants.NUM_IONS - len(l)), 'constant', constant_values=padding_value)])
