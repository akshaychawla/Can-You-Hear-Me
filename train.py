"""
Script to train the model on Tensorflow hot word detection dataset
"""

from __future__ import print_function
import numpy as np
from model import resemul
from keras.models import Model
from keras.layers import Input
from keras import optimizers, callbacks
from keras import utils
from utils import data_generator

# network arch
input_tensor = Input(shape=(16000,))
output_tensor = resemul(
                        input_tensor, block_type="basic", init_features=128,
                        amplifying_ratio=16, drop_rate=0.5, weight_decay=0.0,
                        num_classes=30
                    )
model = Model(input_tensor, output_tensor)
print(model.summary())
sgd = optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
model.compile(sgd, "categorical_crossentropy")
print("..Model Compiled")

# callbacks
checkpoint = callbacks.ModelCheckpoint(
                    filepath="./checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5",
                    save_weights_only=True,
                    verbose=1
                )
lrschedule = callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.2,
                    patience=3,
                    verbose=1,
                    min_lr=1e-07
                )
tboard     = callbacks.TensorBoard(log_dir="./logs/")

# Train the model
dgen = data_generator("/Users/tejaswin.p/Downloads/audio-experiments/speech_commands_v0.01/training.h5py")

history = model.fit_generator(
                    dgen,
                    steps_per_epoch=1597,
                    epochs=10,
                    callbacks=[checkpoint, tboard]
                )

import ipdb; ipdb.set_trace()
