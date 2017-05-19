import numpy as np
import keratin.networks as kn
from keras.optimizers import Adam
from keratin.metrics import dice, dice_loss

def test_unet():
    img_rows = 48
    img_cols = 48
    model = kn.unet(img_rows, img_cols)
    model.compile(optimizer=Adam(lr=10e-5),
                  loss=dice_loss,
                  metrics=[dice])
    img = np.random.randn(2, img_rows, img_cols, 1)
    seg = img > 0
    model.fit(img, seg)


def test_vgg16():
    img_rows = 48
    img_cols = 48
    model = kn.vgg16(img_rows, img_cols, 2)
    model.compile(optimizer=Adam(lr=10e-5),
                  loss=dice_loss,
                  metrics=['accuracy'])
    img = np.random.randn(2, img_rows, img_cols, 1)
    classes = np.array([[0, 1], [1, 0]])
    model.fit(img, classes)

    # Test with 3D image:
    img_rows = 48
    img_cols = 48
    img_z = 48
    model = kn.vgg16(img_rows, img_cols, 2, img_z=img_z)
    model.compile(optimizer=Adam(lr=10e-5),
                  loss=dice_loss,
                  metrics=['accuracy'])
    img = np.random.randn(2, img_rows, img_cols, img_z, 1)
    classes = np.array([[0, 1], [1, 0]])
    model.fit(img, classes)
