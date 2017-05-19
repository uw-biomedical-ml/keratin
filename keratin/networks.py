import keras.models as km
from keras.layers import (Convolution2D, MaxPooling2D, Convolution3D,
                          MaxPooling3D, Flatten, Dense, Input, merge,
                          UpSampling2D, Dropout)
from keras.layers.merge import Concatenate


def unet(img_x, img_y, n_channels=1):
    """
    This is a unet architecture, described in [1]_.

    Parameters
    ----------
    img_x, img_y : int
        Number of rows and columns in each image to be fed in as inputs.
    n_channels : int, optional
        The number of channels in the image.
        Default: 1.

    Returns
    -------
    model : an uncompiled :class:`km.Model` class instance.

    References
    ----------
    .. [1] Olaf Ronneberger, Philipp Fischer, Thomas Brox "U-Net: Convolutional
       Networks for Biomedical Image Segmentation". In: Navab N., Hornegger J.,
       Wells W., Frangi A. (eds) Medical Image Computing and Computer-Assisted
       Intervention – MICCAI 2015. Lecture Notes in Computer Science, vol 9351.
    """
    inputs = Input(shape=(img_x, img_y, n_channels))
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


def vgg16(img_x, img_y, n_classes, img_z=None, n_channels=1,
          pool_dim=2, kernel_dim=3, stride=2):
    """
    The VGG16 architecture described in [1]_.


    Parameters
    ----------
    img_x, img_y : int
        Number of rows and columns in each image to be fed in as inputs.

    n_classes : int
        How many classes do we want to distinguish.

    img_z : int, optional
        Number of images in a stack, for 3D images. Default: 2D images

    n_channels: int, optional.
        Number of channels in the images. Default: 1.


    Returns
    -------
    model : an uncompiled :class:`km.Model` class instance.


    """
    if img_z is None:
        inputs = Input(shape=(img_x, img_y, n_channels))
        kernel_dims = (kernel_dim, kernel_dim)
        pool_dims = (pool_dim, pool_dim)
        strides = (stride, stride)
        conv = Convolution2D
        max_pool = MaxPooling2D
    else:
        inputs = Input(shape=(img_x, img_y, img_z, n_channels))
        kernel_dims = (kernel_dim, kernel_dim, kernel_dim)
        pool_dims = (pool_dim, pool_dim, pool_dim)
        strides = (stride, stride, stride)
        conv = Convolution3D
        max_pool = MaxPooling3D

    # Block 1
    conv1_1 = conv(64, kernel_dims, activation='relu', padding='same',
                   name='block1_conv1')(inputs)
    conv1_2 = conv(64, kernel_dims, activation='relu', padding='same',
                   name='block1_conv2')(conv1_1)
    maxpool1 = max_pool(pool_dims, strides=strides,
                        name='block1_pool')(conv1_2)

    # Block 2
    conv2_1 = conv(128, kernel_dims, activation='relu', padding='same',
                   name='block2_conv1')(maxpool1)
    conv2_2 = conv(128, kernel_dims, activation='relu', padding='same',
                   name='block2_conv2')(conv2_1)
    maxpool2 = max_pool(pool_dims, strides=strides,
                        name='block2_pool')(conv2_2)

    # Block 3
    conv3_1 = conv(256, kernel_dims, activation='relu', padding='same',
                   name='block3_conv1')(maxpool2)
    conv3_2 = conv(256, kernel_dims, activation='relu', padding='same',
                   name='block3_conv2')(conv3_1)
    conv3_3 = conv(256, kernel_dims, activation='relu', padding='same',
                   name='block3_conv3')(conv3_2)
    maxpool3 = max_pool(pool_dims, strides=strides,
                        name='block3_pool')(conv3_3)

    # Block 4
    conv4_1 = conv(512, kernel_dims, activation='relu', padding='same',
                   name='block4_conv1')(maxpool3)
    conv4_2 = conv(512, kernel_dims, activation='relu', padding='same',
                   name='block4_conv2')(conv4_1)
    conv4_3 = conv(512, kernel_dims, activation='relu', padding='same',
                   name='block4_conv3')(conv4_2)
    maxpool4 = max_pool(pool_dims, strides=strides,
                        name='block4_pool')(conv4_3)

    # Block 5
    conv5_1 = conv(512, kernel_dims, activation='relu', padding='same',
                   name='block5_conv1')(maxpool4)
    conv5_2 = conv(512, kernel_dims, activation='relu', padding='same',
                   name='block5_conv2')(conv5_1)
    conv5_3 = conv(512, kernel_dims, activation='relu', padding='same',
                   name='block5_conv3')(conv5_2)
    maxpool5 = max_pool(pool_dims, strides=strides,
                        name='block5_pool')(conv5_3)

    # Classification block
    flatten = Flatten(name='flatten')(maxpool5)
    fc1 = Dense(4096, activation='relu', name='fc1')(flatten)
    fc2 = Dense(4096, activation='relu', name='fc2')(fc1)
    out = Dense(n_classes, activation='softmax', name='predictions')(fc2)
    return km.Model(input=inputs, outputs=out)
