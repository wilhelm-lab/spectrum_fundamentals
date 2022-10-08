import numpy as np

def indices_to_one_hot(data, nb_classes):
    """
    Convert an iterable of indices to one-hot encoded labels.
    :param data: charge, int between 1 and 6
    """
    #print(data)
    targets = np.array([data])
    targets = targets.astype(np.uint8)
    targets = targets-1  # -1 for 0 indexing
    try:
        return np.int_((np.eye(nb_classes)[targets])).tolist()[0]
    except IndexError:
        raise IndexError('Please validate the precursor charge values are between 1 and 6')
