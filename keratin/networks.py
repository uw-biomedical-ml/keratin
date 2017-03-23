import keras.models as km
from keras.layers import (Input, merge, MaxPooling2D,
                          UpSampling2D, ZeroPadding2D, Dropout)
import keras.layers.convolutional as klc
from keras.layers.merge import Concatenate


def segnet(img_rows, img_cols, n_channels=1):
    """
    This is a segnet (u-shaped) architecture, such as that described in
    [1]_

    Parameters
    ----------
    img_rows, img_cols : int
        Number of rows and columns in each image to be fed in as inputs.
    n_channels : int, optional
        The number of channels in the image.
        Default: 1.

    Returns
    -------
    model : a uncompiled :class:`km.Model` class instance.

    References
    ----------
    .. [1] Vijay Badrinarayanan, Alex Kendall and Roberto Cipolla "SegNet: A
    Deep Convolutional Encoder-Decoder Architecture for Image Segmentation."
    `arXiv:1511.00561, 2015. <https://arxiv.org/abs/1511.00561>`_.
    """
    inputs = Input((img_rows, img_cols, n_channels))
    conv1 = klc.Conv2D(32, (3, 3), activation='relu',
                       padding='same')(inputs)
    conv1 = klc.Conv2D(32, (3, 3), activation='relu',
                       padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = klc.Conv2D(64, (3, 3), activation='relu',
                       padding='same')(pool1)
    conv2 = klc.Conv2D(64, (3, 3), activation='relu',
                       padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = klc.Conv2D(128, (3, 3), activation='relu',
                       padding='same')(pool2)
    conv3 = klc.Conv2D(128, (3, 3), activation='relu',
                       padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = klc.Conv2D(256, (3, 3), activation='relu',
                       padding='same')(pool3)
    conv4 = klc.Conv2D(256, (3, 3), activation='relu',
                       padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = klc.Conv2D(512, (3, 3), activation='relu',
                       padding='same')(pool4)
    conv5 = klc.Conv2D(512, (3, 3), activation='relu',
                       padding='same')(conv5)

    up6 = Concatenate()(
            [UpSampling2D(size=(2, 2))(conv5), conv4])

    conv6 = klc.Conv2D(256, (3, 3), activation='relu',
                       padding='same')(up6)
    conv6 = klc.Conv2D(256, (3, 3), activation='relu',
                       padding='same')(conv6)

    up7 = Concatenate()(
            [UpSampling2D(size=(2, 2))(conv6), conv3])

    conv7 = klc.Conv2D(128, (3, 3), activation='relu',
                       padding='same')(up7)
    conv7 = klc.Conv2D(128, (3, 3), activation='relu',
                       padding='same')(conv7)

    up8 = Concatenate()(
            [UpSampling2D(size=(2, 2))(conv7), conv2])

    conv8 = klc.Conv2D(64, (3, 3), activation='relu',
                       padding='same')(up8)
    conv8 = klc.Conv2D(64, (3, 3), activation='relu',
                       padding='same')(conv8)

    up9 = Concatenate()(
            [UpSampling2D(size=(2, 2))(conv8), conv1])
    conv9 = klc.Conv2D(32, (3, 3), activation='relu',
                       padding='same')(up9)
    conv9 = klc.Conv2D(32, (3, 3), activation='relu',
                       padding='same')(conv9)

    conv10 = klc.Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    return km.Model(input=inputs, output=conv10)
