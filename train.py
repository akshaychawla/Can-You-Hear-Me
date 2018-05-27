"""
Script to train the model on Tensorflow hot word detection dataset 
""" 
import numpy as np 
from model import resemul 
from keras.models import Model 
from keras.layers import Input 
from keras import optimizers, callbacks 
from keras import utils 

# network arch 
input_tensor = Input(shape=(16000,)) 
output_tensor = resemul(
                        input_tensor, block_type="rese", init_features=128,
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
# dummy data 
x_train, x_valid = np.random.randn(20, 16000), np.random.randn(5, 16000) 
y_train, y_valid = np.random.randint(0, 30, size=(20,)), np.random.randint(0, 30, size=(5,))
y_train, y_valid = utils.to_categorical(y_train, num_classes=30), utils.to_categorical(y_valid, num_classes=30) 

history = model.fit(
                    x = x_train, 
                    y = y_train, 
                    batch_size=2, 
                    epochs=10, 
                    callbacks=[checkpoint, lrschedule, tboard],
                    validation_data = (x_valid, y_valid)
                ) 

import ipdb; ipdb.set_trace()



