from typing import List, Optional, Union

import numpy as np


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

    max_label = labels.max()
    if classes is None:
        classes = max_label
    if max_label > classes:
        raise ValueError(
            f"All labels must be smaller or equal to the number of classes. max_label: {max_label}, classes: {classes}"
        )

    one_hot = np.zeros((labels.size, classes), dtype=int)
    one_hot[np.arange(labels.size), labels - 1] = 1
    return one_hot
