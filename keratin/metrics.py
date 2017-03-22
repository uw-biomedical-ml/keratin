from keras import backend as K


def jaccard():
    """
    """
    raise NotImplementedError


def dice(y_true, y_pred, smooth=1.0):
    """
    The Dice coefficient, defined as ::

        \frac{2 |X \intersect Y|}{|X| + |Y|}

    Parameters
    ----------
    y_true, y_pred : tensors
        The predicted and binary classification in an image

    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return ((2. * intersection + smooth) /
            (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))


def dice_loss(y_true, y_pred):
    """
    The negative Dice coefficient

    Parameters
    ----------
    y_true, y_pred : tensors
        The predicted and binary classification in an image

    """
    return -dice(y_true, y_pred)
