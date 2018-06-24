"""
Stand-Alone utility for FGSM 
Input: checkpoint for 'rese' model, wavefile, eta, output_folder
Output: (DICT -> orig_numpy, grads_numpy, perturbed_sample_numpy) + wavefile perturbed sample

"""
import numpy as np
import matplotlib.pyplot as plt
import os, sys, time, pickle
from keras.callbacks import Callback, ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras.layers import Input
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers 
import keras.backend as K
from keras.models import Model, Sequential
from keras import losses
from tqdm import *
from utils import data_generator, load_audio_16k
import argparse
from model import resemul
import librosa

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

def get_model(weights):
    """
    Define the model arch + load weights
    """
    # network arch
    print("Creating model..")
    input_tensor = Input(shape=(16000,))
    output_tensor = resemul(
                            input_tensor, block_type="rese", init_features=128,
                            amplifying_ratio=16, drop_rate=0.5, weight_decay=0.0,
                            num_classes=31
                        )
    model = Model(input_tensor, output_tensor)
    # sgd = optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
    # model.compile(sgd, "categorical_crossentropy", metrics=["accuracy"])
    print("Model compiled.")
    model.load_weights(weights)
    print("weights loaded..")

    return model 

def attack(wavefile, checkpoint, eta, dict_index):

    # infer gt 
    gt_string = os.path.dirname(wavefile).split("/")[-1]
    assert gt_string in dict_index.keys(), "Could not find {} in dict".format(gt_string)
    gt_index = dict_index[gt_string] 
    # print("GT String: {} | GT index: {}".format(gt_string, gt_index))
    gt_index = to_categorical(gt_index, num_classes=31)
    gt_index = gt_index.reshape(1,-1)

    # Load audio file + Model with checkpoint
    x = load_audio_16k(wavefile) 
    x = x.reshape(1,-1)
    model = get_model(checkpoint) 

    # predict before attack 
    preds_pre = model.predict(x, batch_size=1) 

    # Calculate gradient 
    calc_grads = create_gradient_function(model, 0, -1)
    _, grads_x = calc_grads([x, gt_index])

    # Perturb + predict
    attacked_x = x + eta*np.sign(grads_x)
    preds_post = model.predict(attacked_x, batch_size=1) 

    argmax_pre, argmax_post = np.argmax(preds_pre), np.argmax(preds_post)
    
    return x, attacked_x, grads_x, argmax_pre, argmax_post 



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",type=str,help="weights/checkpoint h5")
    parser.add_argument("--wavefile",type=str, help="wavefile to attack")
    parser.add_argument("--eta", type=float, help="eta for attack") 
    parser.add_argument("--output_folder", type=str, help="directory to store output")
    parser.add_argument("--dict_location", type=str, help="location of DICT_ix_pickle file")
    args = parser.parse_args() 

    # checks 
    assert os.path.isfile(args.checkpoint), "{} is not a valid file".format(args.checkpoint)
    assert os.path.isfile(args.wavefile), "{} is not a valid file".format(args.wavefile) 
    assert os.path.isdir(args.output_folder), "{} is not a valid folder".format(args.output_folder)
    assert os.path.isfile(args.dict_location), "{} is not a valid dictionary".format(args.dict_location)
    assert isinstance(args.eta, float), "eta is not a valid float"

    # Load index dictionary 
    with open(args.dict_location, "rb") as f: 
        dict_index = pickle.load(f) 
    orig_np, attacked_np, grads_np, pre_idx, post_idx = attack(
                                                                args.wavefile, 
                                                                args.checkpoint,
                                                                args.eta, 
                                                                dict_index
                                                        )
    dict_index_reverse = {value:key for key,value in dict_index.items()}
    print("Before attack: ",dict_index_reverse[pre_idx])
    print("After attack: ",dict_index_reverse[post_idx])
    print("saving files in folder: ", args.output_folder) 
    np.save(os.path.join(args.output_folder,"origs.npy"), orig_np)
    np.save(os.path.join(args.output_folder,"attacked.npy"), attacked_np)
    np.save(os.path.join(args.output_folder,"grads.npy"), grads_np)
    librosa.output.write_wav(os.path.join(args.output_folder,os.path.basename(args.wavefile)), orig_np.flatten(), 16000)
    librosa.output.write_wav(os.path.join(args.output_folder,"attacked_"+os.path.basename(args.wavefile)), attacked_np.flatten(), 16000)
    print("..Finished")




    # def on_epoch_end(self, epoch, logs = {}):
        # """
        # Attack!
        # """

        # # Performance before attack
        # preds_pre_attack = self.model.predict(
                                # x = self.testX_normed,
                                # batch_size=50, 
                                # verbose=1
                            # )
        # performance_pre_attack = np.count_nonzero(
                                # np.argmax(preds_pre_attack, axis=1) == 
                                # np.argmax(self.testY, axis=1)
                            # )
        # print("[FGSM]Accuracy before attack is: ", performance_pre_attack/len(preds_pre_attack))

        # ### Perform attack
        # # Calculate grad w.r.t input for all test images
        # print("[FGSM]Calculating gradient w.r.t input..")
        # calc_grads = create_gradient_function(self.model, 0, -1)
        # grads_X_test = []
        # for start_idx in tqdm(range(0, len(self.testX_normed), 50)):
            # end_idx = min(len(self.testX_normed), start_idx+50)
            # x_batch = self.testX_normed[start_idx: end_idx]
            # y_batch = self.testY[start_idx: end_idx]
            # _, grads_batch = calc_grads([x_batch, y_batch])
            # grads_X_test.append(grads_batch)
        # grads_X_test = np.concatenate(grads_X_test, axis=0)

        # # attacked = orig + eta*sign(grad)
        # assert grads_X_test.shape == self.testX_normed.shape
        # attacked_testX = self.testX_normed + self.eta*np.sign(grads_X_test)
        # print("[FGSM] Attacked..")

        # # Performance after attack
        # preds_post_attack = self.model.predict(
                                # x = attacked_testX,
                                # batch_size=32, 
                                # verbose=1
                            # )
        # performance_post_attack = np.count_nonzero(
                                # np.argmax(preds_post_attack, axis=1) == 
                                # np.argmax(self.testY, axis=1)
                            # )
        # print("[FGSM]Accuracy after attack is: ", performance_post_attack/len(preds_pre_attack))

        # # Clean up 
        # del calc_grads



