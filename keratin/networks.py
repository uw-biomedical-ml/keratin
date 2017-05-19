import keras.models as km
from keras.layers import (Convolution2D, MaxPooling2D, Convolution3D,
                          MaxPooling3D, Flatten, Dense, Input, UpSampling2D,
                          UpSampling3D)
from keras.layers.merge import Concatenate


def unet(img_x, img_y, img_z=None, n_channels=1,
         pool_dim=2, upsamp_dim=2, kernel_dim=3, stride=2):
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
    if img_z is None:
        inputs = Input(shape=(img_x, img_y, n_channels))
        kernel_dims = (kernel_dim, kernel_dim)
        final_kernel_dims = (1, 1)
        pool_dims = (pool_dim, pool_dim)
        upsamp_size = (upsamp_dim, upsamp_dim)
        strides = (stride, stride)
        conv = Convolution2D
        max_pool = MaxPooling2D
        upsamp = UpSampling2D
    else:
        inputs = Input(shape=(img_x, img_y, img_z, n_channels))
        kernel_dims = (kernel_dim, kernel_dim, kernel_dim)
        final_kernel_dims = (1, 1, 1)
        pool_dims = (pool_dim, pool_dim, pool_dim)
        upsamp_size = (upsamp_dim, upsamp_dim, upsamp_dim)
        strides = (stride, stride, stride)
        conv = Convolution3D
        max_pool = MaxPooling3D
        upsamp = UpSampling3D

    cnv1_1 = conv(32, kernel_dims, activation='relu', padding='same')(inputs)
    cnv1_2 = conv(32, kernel_dims, activation='relu', padding='same')(cnv1_1)
    pool1 = max_pool(pool_size=pool_dims)(cnv1_2)
    cnv2_1 = conv(64, kernel_dims, activation='relu', padding='same')(pool1)
    cnv2_2 = conv(64, kernel_dims, activation='relu', padding='same')(cnv2_1)
    pool2 = max_pool(pool_size=pool_dims)(cnv2_2)

    cnv3_1 = conv(128, kernel_dims, activation='relu', padding='same')(pool2)
    cnv3_2 = conv(128, kernel_dims, activation='relu', padding='same')(cnv3_1)
    pool3 = max_pool(pool_size=pool_dims)(cnv3_2)
    cnv4_1 = conv(256, kernel_dims, activation='relu', padding='same')(pool3)
    cnv4_2 = conv(256, kernel_dims, activation='relu', padding='same')(cnv4_1)
    pool4 = max_pool(pool_size=pool_dims)(cnv4_2)

    cnv5_1 = conv(512, kernel_dims, activation='relu', padding='same')(pool4)
    cnv5_2 = conv(512, kernel_dims, activation='relu', padding='same')(cnv5_1)

    up6 = Concatenate()([upsamp(size=upsamp_size)(cnv5_2), cnv4_2])

    cnv6_1 = conv(256, kernel_dims, activation='relu', padding='same')(up6)
    cnv6_2 = conv(256, kernel_dims, activation='relu', padding='same')(cnv6_1)

    up7 = Concatenate()([upsamp(size=upsamp_size)(cnv6_2), cnv3_2])

    cnv7_1 = conv(128, kernel_dims, activation='relu', padding='same')(up7)
    cnv7_2 = conv(128, kernel_dims, activation='relu', padding='same')(cnv7_1)

    up8 = Concatenate()([upsamp(size=upsamp_size)(cnv7_2), cnv2_2])

    cnv8_1 = conv(64, kernel_dims, activation='relu', padding='same')(up8)
    cnv8_2 = conv(64, kernel_dims, activation='relu', padding='same')(cnv8_1)

    up9 = Concatenate()([upsamp(size=upsamp_size)(cnv8_2), cnv1_2])
    cnv9_1 = conv(32, kernel_dims, activation='relu', padding='same')(up9)
    cnv9_2 = conv(32, kernel_dims, activation='relu', padding='same')(cnv9_1)

    outputs = conv(1, final_kernel_dims, activation='sigmoid')(cnv9_2)
    return km.Model(input=inputs, outputs=outputs)


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
    outputs = Dense(n_classes, activation='softmax', name='predictions')(fc2)
    return km.Model(input=inputs, outputs=outputs)
