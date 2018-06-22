"""
Script to train the model on Tensorflow hot word detection dataset
"""

from __future__ import print_function
import numpy as np
from model import resemul
from datetime import datetime
from keras.models import Model
from keras.layers import Input
from keras import optimizers, callbacks
from keras import utils
from utils import data_generator
from kaggle_utils import score_generator
import sys
import os
try:
    import cPickle as pickle
except:
    import pickle

# network arch
input_tensor = Input(shape=(16000,))
output_tensor = resemul(
                        input_tensor, block_type="rese", init_features=128,
                        amplifying_ratio=16, drop_rate=0.5, weight_decay=0.0,
                        num_classes=31
                    )
model = Model(input_tensor, output_tensor)
print(model.summary())
sgd = optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
model.compile(sgd, "categorical_crossentropy", metrics=["accuracy"])
print("Model compiled.")


# scoring
if sys.argv[1] == "score":
    model.load_weights("/home/tejaswin.p/Can-You-Hear-Me/checkpoints/2018-06-22_08:40:45.299850/weights.04-0.25.hdf5")
    print("Weights loaded.")

    score_dgen = score_generator(sys.argv[2], batch_size=250)
    with open("/home/tejaswin.p/.kaggle/competitions/tensorflow-speech-recognition-challenge/train/audio/DICT_ix_class.cpkl", "rb") as fp:
        ix_label = {ix:label for label, ix in pickle.load(fp).items()}


    allowed = "yes, no, up, down, left, right, on, off, stop, go".strip().split(",")
    allowed = [token.strip() for token in allowed]

    for ix,label in ix_label.items():
        if label not in allowed:
            ix_label[ix] = "unknown"

    score_csv = ["fname,label\n"]

    _scount = 0
    for _sdata, _stargets in score_dgen:
        score_preds = np.argmax(model.predict_on_batch(_sdata), axis=1)
        score_labels = [ix_label[v] for v in score_preds]

        score_csv.extend([a+","+b+"\n" for a,b in zip(_stargets, score_labels)])
        print("Completed", _scount+1, "batch.")
        _scount += 1

    print("\nScoring done...")
    with open("/home/tejaswin.p/.kaggle/competitions/tensorflow-speech-recognition-challenge/submit.csv", "w") as fp:
        fp.writelines(score_csv)
    print("Written to file.")
    sys.exit()


# callbacks
_expdt = str(datetime.now()).replace(' ', '_')
os.makedirs("./checkpoints/%s"%_expdt)
os.makedirs("./logs/%s"%_expdt)

checkpoint = callbacks.ModelCheckpoint(
                    filepath="./checkpoints/%s/weights.{epoch:02d}-{val_loss:.2f}.hdf5"%_expdt,
                    save_weights_only=True,
                    verbose=1
                )
lrschedule = callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=3,
                    verbose=1,
                    min_lr=1e-07
                )
tboard     = callbacks.TensorBoard(log_dir="./logs/%s"%_expdt)

# Train the model
batch_size = 32
train_dgen = data_generator(sys.argv[1], batch_size)
valid_dgen  = data_generator(sys.argv[2], batch_size)
train_steps_per_epoch = 51384 // batch_size + 1
valid_steps_per_epoch = 13718 // batch_size + 1
print("Train steps per epoch: ", train_steps_per_epoch)
print("valid steps per epoch: ", valid_steps_per_epoch)

history = model.fit_generator(
                    train_dgen,
                    steps_per_epoch=train_steps_per_epoch,
                    epochs=50,
                    validation_data=valid_dgen,
                    validation_steps=valid_steps_per_epoch,
                    callbacks=[checkpoint, lrschedule, tboard]
                )

import ipdb; ipdb.set_trace()
