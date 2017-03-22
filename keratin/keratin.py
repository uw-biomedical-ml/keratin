from __future__ import absolute_import, division, print_function

import keras

from .due import due, Doi

# Use duecredit (duecredit.org) to provide a citation to relevant work to
# be cited. This does nothing, unless the user has duecredit installed,
# And calls this with duecredit (as in `python -m duecredit script.py`):
due.cite(Doi("10.1167/13.9.30"),
         description="Template project for small scientific Python projects",
         tags=["reference-implementation"],
         path='shablona')


class LossHistory(keras.callbacks.Callback):
    def __init__(self, log_path, ):
        keras.callbacks.Callback.__init__(self)
        self.log_path = log_path

    def on_train_begin(self, logs={}):
        self.losses = []
        self.valid = []
        self.lastiter = 0

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('dice_coef'))
        self.lastiter = len(self.losses) - 1

        self.valid.append([self.lastiter, logs.get('val_dice_coef')])

        with open("/data/octirfseg/runs/%d/history.txt" % pid, "w") as fout:
                strings = ["train\t%d\t%.4f" % (i, x) for i ,x in enumerate(self.losses)]
                fout.write("\n".join(strings) + "\n")
                strings = ["valid\t%d\t%.4f" % (i, x) for i ,x in self.valid]
                fout.write("\n".join(strings) + "\n")



    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('dice_coef'))
        self.lastiter = len(self.losses) - 1
        with open("/data/octirfseg/runs/%d/history.txt" % pid, "w") as fout:
                strings = ["train\t%d\t%.4f" % (i, x) for i ,x in enumerate(self.losses)]
                fout.write("\n".join(strings) + "\n")
                strings = ["valid\t%d\t%.4f" % (i, x) for i ,x in self.valid]
                fout.write("\n".join(strings) + "\n")
