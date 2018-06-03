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
import sys
import os

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
model.compile(sgd, "categorical_crossentropy", metrics=["accuracy"])
print("..Model Compiled")

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
                    factor=0.2,
                    patience=3,
                    verbose=1,
                    min_lr=1e-07
                )
tboard     = callbacks.TensorBoard(log_dir="./logs/%s"%_expdt)

# Train the model
batch_size = 32
train_dgen = data_generator(sys.argv[1], batch_size)
valid_dgen  = data_generator(sys.argv[2], batch_size)
train_steps_per_epoch = 51087 // batch_size + 1
valid_steps_per_epoch = 13633 // batch_size + 1
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
