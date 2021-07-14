import pytest
import numpy as np

import fundamentals.metrics.similarity as sim
import fundamentals.constants as constants


class TestSpectralAngle:
    
    def test_l2_norm(self):
        vector = np.array([1,2,3,4], dtype = 'float64')
        np.testing.assert_equal(sim.SimilarityMetrics.l2_norm(vector), np.sqrt(30))
    
    def test_spectral_angle_equal(self):
        vector1 = np.array([1,2,3,4], dtype = 'float64')
        vector2 = np.array([1,2,3,4], dtype = 'float64')
        np.testing.assert_almost_equal(sim.SimilarityMetrics.spectral_angle(vector1, vector2), 1.0)
    
    def test_spectral_angle_equal_scaled(self):
        vector1 = np.array([1,2,3,4], dtype = 'float64')
        vector2 = np.array([2,4,6,8], dtype = 'float64')
        np.testing.assert_almost_equal(sim.SimilarityMetrics.spectral_angle(vector1, vector2), 1.0)
    
    def test_spectral_angle_zero(self):
        z = constants.EPSILON
        vector1 = np.array([z,2,z,4], dtype = 'float64')
        vector2 = np.array([1,z,3,z], dtype = 'float64')
        np.testing.assert_almost_equal(sim.SimilarityMetrics.spectral_angle(vector1, vector2), 0.0)
    
    def test_spectral_angle_all_zero(self):
        z = constants.EPSILON
        vector1 = np.array([z,z,z,z], dtype = 'float64')
        vector2 = np.array([z,z,z,z], dtype = 'float64')
        np.testing.assert_almost_equal(sim.SimilarityMetrics.spectral_angle(vector1, vector2), 0.0)
        
    def test_spectral_angle_invalid(self):
        vector1 = np.array([0,2,0,4], dtype = 'float64')
        vector2 = np.array([1,0,3,0], dtype = 'float64')
        np.testing.assert_almost_equal(sim.SimilarityMetrics.spectral_angle(vector1, vector2), 0.0)
    
    def test_spectral_angle_full(self):
        # 1 - 2*arccos(28/30)/pi
        vector1 = np.array([1,2,4,3], dtype = 'float64')
        vector2 = np.array([2,1,3,4], dtype = 'float64')
        np.testing.assert_almost_equal(sim.SimilarityMetrics.spectral_angle(vector1, vector2), 0.76622811354)
    
    def test_spectral_angle_full_with_zeros(self):
        # 1 - 2*arccos(24/29)/pi
        z = constants.EPSILON
        vector1 = np.array([z,2,4,3], dtype = 'float64')
        vector2 = np.array([2,z,3,4], dtype = 'float64')
        np.testing.assert_almost_equal(sim.SimilarityMetrics.spectral_angle(vector1, vector2), 0.62057306283)

    def test_spectral_angle_full_with_both_zeros(self):
        # 1 - 2*arccos(12/25)/pi
        z = constants.EPSILON
        vector1 = np.array([z,3,4,z], dtype = 'float64')
        vector2 = np.array([z,z,3,4], dtype = 'float64')
        np.testing.assert_almost_equal(sim.SimilarityMetrics.spectral_angle(vector1, vector2), 0.31872668904)
