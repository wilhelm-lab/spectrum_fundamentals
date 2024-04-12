import unittest
from typing import List, Optional, Union

import numpy as np

import spectrum_fundamentals.charge as charge


class TestCharge(unittest.TestCase):
    """Class to test charge."""

    def test_indices_to_one_hot_with_classes(self):
        """Test indices_to_one_hot with given number of classes."""
        labels = np.array([1, 2, 3])
        classes = 4
        expected_output = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
        np.testing.assert_equal(charge.indices_to_one_hot(labels, classes), expected_output)

    # Inside charge.py


def indices_to_one_hot(labels: Union[int, List[int], np.ndarray], classes: Optional[int] = None) -> np.ndarray:
    """
    Convert a single or a list of labels to one-hot encoding.

    :param labels: The labels to be one-hot encoding. Must be one-based.
    :param classes: The number of classes, i.e. the length of the encoding. If omitted, set to the max label + 1.
    :raises TypeError: If the type of labels is not understood
    :raises ValueError: If the highest label in labels is larger or equal to the number of classes.
    :return: np.ndarray with the one-hot encoded labels.
    """
    if isinstance(labels, int):
        labels = np.array([labels])
    elif isinstance(labels, (list, np.ndarray)):
        labels = np.array(labels)
    else:
        raise TypeError(
            f"Type of labels not understood. Only int, List[int] and np.ndarray are supported. Given: {type(labels)}."
        )

    if classes is None:
        classes = np.max(labels) + 1
    elif classes < np.max(labels) + 1:
        raise ValueError(
            f"Number of classes must be greater than or equal to the maximum label in labels. "
            f"Given classes: {classes}, maximum label: {np.max(labels)}."
        )

    one_hot = np.zeros((labels.size, classes), dtype=int)
    one_hot[np.arange(labels.size), labels - 1] = 1
    return one_hot

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

    def test_indices_to_one_hot_with_incompatible_classes(self):
        """Test indices_to_one_hot correctly raises TypeError on wrong input type."""
        labels = [1, 2, 3]
        classes = 2
        self.assertRaises(ValueError, charge.indices_to_one_hot, labels, classes)
