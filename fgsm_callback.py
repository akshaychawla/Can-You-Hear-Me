"""
Callback for fgsm. 
This will attack the model after each epoch
"""
import numpy as np
import matplotlib.pyplot as plt
import os, sys, time, pickle
from keras.callbacks import Callback, ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
import keras.backend as K
from keras import losses
from tqdm import *
from utils import data_generator

def create_gradient_function(model, input_idx, output_idx):
    gt_tensor     = K.placeholder(shape=(None, 31))
    input_tensor  = model.inputs[input_idx]
    output_tensor = model.outputs[output_idx]
    assert "softmax" in output_tensor.name.lower(), "[ERROR] output tensor name is {}".format(output_tensor.name)
    loss_tensor   = losses.categorical_crossentropy(gt_tensor, output_tensor)
    grads_tensor  = K.gradients(loss_tensor, input_tensor)[0]

    get_gradients_function = K.function(
                                inputs=[input_tensor, gt_tensor],
                                outputs=[loss_tensor, grads_tensor])
    return get_gradients_function

class fgsm_callback(Callback):

    def __init__(self, test_h5, eta=0.05):
        super(fgsm_callback, self).__init__() 
        self.eta = eta
        self.test_h5 = test_h5

        # generators
        batch_size = 32
        test_dgen = data_generator(self.test_h5, batch_size)
        test_steps_per_epoch = 13718 // batch_size + 1
        print("test steps per epoch: ", test_steps_per_epoch)

        # normalize test set
        print("[FGSM] Generating normed version of test set..")
        testX_normed = [] 
        self.testY = []
        for _ in range(test_steps_per_epoch):
            x_batch, y_batch = next(test_dgen)
            testX_normed.append(x_batch)
            self.testY.append(y_batch)
        testX_normed = np.concatenate(testX_normed, axis=0)
        self.testY = np.concatenate(self.testY, axis=0)
        del x_batch, y_batch
        self.testX_normed = testX_normed
        print("[FGSM] Done")
        print("[FGSM] shapes {} {}".format(self.testX_normed.shape, self.testY.shape))

    def on_epoch_end(self, epoch, logs = {}):
        """
        Attack!
        """

        # Performance before attack
        preds_pre_attack = self.model.predict(
                                x = self.testX_normed,
                                batch_size=50, 
                                verbose=1
                            )
        performance_pre_attack = np.count_nonzero(
                                np.argmax(preds_pre_attack, axis=1) == 
                                np.argmax(self.testY, axis=1)
                            )
        print("[FGSM]Accuracy before attack is: ", performance_pre_attack/len(preds_pre_attack))

        ### Perform attack
        # Calculate grad w.r.t input for all test images
        print("[FGSM]Calculating gradient w.r.t input..")
        calc_grads = create_gradient_function(self.model, 0, -1)
        grads_X_test = []
        for start_idx in tqdm(range(0, len(self.testX_normed), 50)):
            end_idx = min(len(self.testX_normed), start_idx+50)
            x_batch = self.testX_normed[start_idx: end_idx]
            y_batch = self.testY[start_idx: end_idx]
            _, grads_batch = calc_grads([x_batch, y_batch])
            grads_X_test.append(grads_batch)
        grads_X_test = np.concatenate(grads_X_test, axis=0)

        # attacked = orig + eta*sign(grad)
        assert grads_X_test.shape == self.testX_normed.shape
        attacked_testX = self.testX_normed + self.eta*np.sign(grads_X_test)
        print("[FGSM] Attacked..")

        # Performance after attack
        preds_post_attack = self.model.predict(
                                x = attacked_testX,
                                batch_size=32, 
                                verbose=1
                            )
        performance_post_attack = np.count_nonzero(
                                np.argmax(preds_post_attack, axis=1) == 
                                np.argmax(self.testY, axis=1)
                            )
        print("[FGSM]Accuracy after attack is: ", performance_post_attack/len(preds_pre_attack))

        # Clean up 
        del calc_grads



