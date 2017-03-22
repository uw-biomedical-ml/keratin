import keratin.networks as kn
from keras.optimizers import Adam
from keratin.metrics import dice, dice_loss

def test_unet():
    img_rows = 432
    img_cols = 32
    u = kn.unet(img_rows, img_cols)
    u.compile(optimizer=Adam(lr=10e-5),
                  loss=dice_loss,
                  metrics=[dice])
