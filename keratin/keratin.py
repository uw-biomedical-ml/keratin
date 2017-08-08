from __future__ import absolute_import, division, print_function

import keras

from .due import due, Doi  # noqa

# Use duecredit (duecredit.org) to provide a citation to relevant work to
# be cited. This does nothing, unless the user has duecredit installed,
# And calls this with duecredit (as in `python -m duecredit script.py`):
# due.cite(Doi("XXX"),
#         description="",
#         tags=["", ""],
#         path='XXX')


class LossHistory(keras.callbacks.Callback):
    def __init__(self, log_path, loss_name, valid_name):
        keras.callbacks.Callback.__init__(self)
        self.log_path = log_path
        self.loss_name = loss_name
        self.valid_name = valid_name

    def on_train_begin(self, logs={}):
        self.losses = []
        self.valid = []
        self.lastiter = 0

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get(self.loss_name))
        self.lastiter = len(self.losses) - 1
        self.valid.append([self.lastiter, logs.get(self.valid_name)])

        with open(op.join(self.log_path, "/%d/history.txt" % pid), "w") as fo:
                strings = ["train\t%d\t%.4f" % (i, x) for i, x in
                           enumerate(self.losses)]
                fo.write("\n".join(strings) + "\n")
                strings = ["valid\t%d\t%.4f" % (i, x) for i, x in self.valid]
                fo.write("\n".join(strings) + "\n")

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get(self.loss_name))
        self.lastiter = len(self.losses) - 1
        with open(op.join(self.log_path, "/%d/history.txt" % pid), "w") as fo:
            strings = ["train\t%d\t%.4f" % (i, x) for i, x in
                       enumerate(self.losses)]
            fo.write("\n".join(strings) + "\n")
            strings = ["valid\t%d\t%.4f" % (i, x) for i, x in self.valid]
            fo.write("\n".join(strings) + "\n")


def generate_arrays(files, batch_size, mean=0, std=1):
    """
    Generate arrays for training
    """
    maxvolid = find_max_vol_id()
    while True:
        vols = range(0, maxvolid)
        random.shuffle(vols)
        for volid in vols:
            imgs_train, imgs_mask_train = load_train_data(volid)
            imgs_train -= mean
            imgs_train /= std
            for i in range(0, imgs_train.shape[0], batchsize):
                batch_x = imgs_train[i:i+batchsize]
                batch_y = imgs_mask_train[i:i+batchsize]
                yield (batch_x, batch_y)
