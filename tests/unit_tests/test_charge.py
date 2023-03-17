import spectrum_fundamentals.charge as charge
import numpy as np

class TestCharge:
    """Class to test charge."""

    def test_indices_to_one_hot(self):
        """Test get_mask_observed_valid."""

    def test_indices_to_one_hot_with_classes(self):
        """Test indices_to_one_hot."""
        labels = np.array([1, 2, 3])
        classes = 4
        expected_output = np.array([[1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 1, 0]])
        np.testing.assert_equal(charge.indices_to_one_hot(labels, classes), expected_output)

    def test_indices_to_one_hot_without_classes(self):
        """Test indices_to_one_hot."""
        labels = np.array([1, 2, 3, 4])
        expected_output = np.array([[1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]])
        np.testing.assert_equal(charge.indices_to_one_hot(labels), expected_output)

