import keras.models as km
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Input, merge, UpSampling2D, ZeroPadding2D, Dropout
from keras.layers.merge import Concatenate


def unet(img_rows, img_cols, n_channels=1):
    """
    This is a unet architecture, described in

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
    .. [1] Olaf Ronneberger, Philipp Fischer, Thomas Brox "U-Net: Convolutional
       Networks for Biomedical Image Segmentation". In: Navab N., Hornegger J.,
       Wells W., Frangi A. (eds) Medical Image Computing and Computer-Assisted
       Intervention – MICCAI 2015. Lecture Notes in Computer Science, vol 9351.
    """
    inputs = Input(shape=(img_rows, img_cols, n_channels))
    conv1 = Convolution2D(32, (3, 3), activation='relu',
                          padding='same')(inputs)
    conv1 = Convolution2D(32, (3, 3), activation='relu',
                          padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, (3, 3), activation='relu',
                          padding='same')(pool1)
    conv2 = Convolution2D(64, (3, 3), activation='relu',
                          padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, (3, 3), activation='relu',
                          padding='same')(pool2)
    conv3 = Convolution2D(128, (3, 3), activation='relu',
                          padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, (3, 3), activation='relu',
                          padding='same')(pool3)
    conv4 = Convolution2D(256, (3, 3), activation='relu',
                          padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, (3, 3), activation='relu',
                          padding='same')(pool4)
    conv5 = Convolution2D(512, (3, 3), activation='relu',
                          padding='same')(conv5)

    up6 = Concatenate()(
            [UpSampling2D(size=(2, 2))(conv5), conv4])

    conv6 = Convolution2D(256, (3, 3), activation='relu',
                          padding='same')(up6)
    conv6 = Convolution2D(256, (3, 3), activation='relu',
                          padding='same')(conv6)

    up7 = Concatenate()(
            [UpSampling2D(size=(2, 2))(conv6), conv3])

    conv7 = Convolution2D(128, (3, 3), activation='relu',
                          padding='same')(up7)
    conv7 = Convolution2D(128, (3, 3), activation='relu',
                          padding='same')(conv7)

    up8 = Concatenate()(
            [UpSampling2D(size=(2, 2))(conv7), conv2])

    conv8 = Convolution2D(64, (3, 3), activation='relu',
                          padding='same')(up8)
    conv8 = Convolution2D(64, (3, 3), activation='relu',
                          padding='same')(conv8)

    up9 = Concatenate()(
            [UpSampling2D(size=(2, 2))(conv8), conv1])
    conv9 = Convolution2D(32, (3, 3), activation='relu',
                          padding='same')(up9)
    conv9 = Convolution2D(32, (3, 3), activation='relu',
                          padding='same')(conv9)

    conv10 = Convolution2D(1, (1, 1), activation='sigmoid')(conv9)
    return km.Model(input=inputs, outputs=conv10)
