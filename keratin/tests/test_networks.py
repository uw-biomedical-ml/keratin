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
