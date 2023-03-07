from ast import List

import numpy as np


def indices_to_one_hot(data: int, nb_classes: int) -> List[int]:
    """
    Convert an iterable of indices to one-hot encoded labels.

    :param data: charge, int between 1 and 6
    :param nb_classes: int
    :return: a list representing the one-hot encoded label
    :raises IndexError: if the value of `data` is less than 1 or greater than 6
    """
    targets = np.array([data])
    targets = targets.astype(np.uint8)
    targets = targets - 1  # -1 for 0 indexing
    try:
        return np.int_(np.eye(nb_classes)[targets]).tolist()[0]
    except IndexError as ie:
        raise IndexError("Please validate the precursor charge values are between 1 and 6") from ie
