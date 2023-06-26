import typing
import unittest

import numpy as np

import spectrum_fundamentals.charge as charge


class TestCharge(unittest.TestCase):
    """Class to test charge."""

    def test_indices_to_one_hot(self):
        """Test get_mask_observed_valid."""

    def test_indices_to_one_hot_with_classes(self):
        """Test indices_to_one_hot with given number of classes."""
        labels = np.array([1, 2, 3])
        classes = 4
        expected_output = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
        np.testing.assert_equal(charge.indices_to_one_hot(labels, classes), expected_output)

    def test_indices_to_one_hot_without_classes(self):
        """Test indices_to_one_hot."""
        labels = np.array([1, 2, 3, 4])
        expected_output = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        np.testing.assert_equal(charge.indices_to_one_hot(labels), expected_output)

    def test_indices_to_one_hot_with_int_and_class(self):
        """Test indices_to_one_hot with a single integer and given number of classes."""
        labels = 1
        classes = 4
        expected_output = np.array([[1, 0, 0, 0]])
        np.testing.assert_equal(charge.indices_to_one_hot(labels, classes), expected_output)

    def test_indices_to_one_hot_with_list_and_class(self):
        """Test indices_to_one_hot with a list and given number of classes."""
        labels = [1, 2, 3]
        classes = 4
        expected_output = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
        np.testing.assert_equal(charge.indices_to_one_hot(labels, classes), expected_output)

    def test_indices_to_one_hot_with_wrong_input_type(self):
        """Test indices_to_one_hot correctly raises TypeError on wrong input type."""
        typing.TYPE_CHECKING = False
        labels = None
        self.assertRaises(TypeError, charge.indices_to_one_hot, labels)
        typing.TYPE_CHECKING = True

    def test_indices_to_one_hot_with_incompatible_classes(self):
        """Test indices_to_one_hot correctly raises TypeError on wrong input type."""
        labels = [1, 2, 3]
        classes = 2
        self.assertRaises(ValueError, charge.indices_to_one_hot, labels, classes)
