"""
Utility codes for data, eval, plotting, etc.
"""

from __future__ import print_function
import os
import re
import hashlib
import sys
import librosa
import librosa.display
import h5py
import numpy as np
import multiprocessing as mp
try:
    import cPickle as pickle
except:
    import pickle
from keras.utils import to_categorical
from joblib import Parallel, delayed
from functools import partial
from scipy.fftpack import fft
import matplotlib.pyplot as plt


SRATE = 16000
def librosa_load(path, srate):
    audio, _ = librosa.load(path, sr=srate)
    if audio.size < srate:
        return np.pad(audio, (srate - audio.size, 0), mode='constant')
    else:
        return audio[:srate]
load_audio_16k = partial(librosa_load, srate=SRATE)


def plot_fft(y, fs=SRATE):
    """
    Plot the frequency domain visualisation.
    Inputs: audio array, sampling rate
    """
    T = 1.0 / fs
    N = y.shape[0]
    yf = fft(y)
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
    vals = 2.0/N * np.abs(yf[0:N//2])
    # FFT is simmetrical, so we take just the first half
    # FFT is also complex, to we take just the real part (abs)
    plt.figure(figsize=(14, 5))
    plt.title('FFT of recording sampled with ' + str(SRATE) + ' Hz')
    plt.plot(xf, vals)
    plt.xlabel('Frequency')
    plt.grid()
    plt.show()
    return


def plot_spectogram(y, sr=SRATE):
    """
    Plots spectogram of input signal.
    Inputs: audio array, sampling rate
    """
    X = librosa.stft(y)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.show()
    return


def data_generator(h5path, batch_size=32):
    """
    Custom generator for loading from h5 files.
    """
    assert os.path.isfile(h5path), "%s does not exist"
    with h5py.File(h5path, 'r') as f:
        data = f["subgroup"]["data"]
        targets = f["subgroup"]["targets"]
        assert len(data) == len(targets), "data, target lengths mismatch"
        print("File", h5path, len(targets))
        while True:
            data_indices = np.arange(len(data)).astype(np.int32)
            np.random.shuffle(data_indices)
            for _index in range(0, len(data_indices), batch_size):
                indices_for_h5 = data_indices[ _index: min(len(data),_index+batch_size) ]
                indices_for_h5 = np.sort(indices_for_h5).tolist()
                x_batch, y_batch = data[indices_for_h5], targets[indices_for_h5]
                yield x_batch, y_batch


def make_training_rest_list(data_root, exclude_dirs = ["_background_noise_"]):
    """
    Download v0.01 from http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz
    This function will generate and store the training_list.txt and rest_list.txt
    TO BE RUN ONLY AFTER DOWNLOADING FRESH DATA.
    """

    assert os.path.isdir(data_root), "directory %s does not exist"%data_root
    train_path = os.path.join(data_root, "training_list.txt")
    rest_path = os.path.join(data_root, "rest_list.txt")
    assert not os.path.isfile(train_path), "training_list.txt already exists in %s"%data_root
    assert not os.path.isfile(rest_path), "rest_list.txt already exists in %s"%data_root

    validation_path = os.path.join(data_root, "validation_list.txt")
    test_path = os.path.join(data_root, "testing_list.txt")
    train_files = []
    rest_files = []

    # print("\nLoading files to exclude from training...")
    # with open(test_path) as fp:
    #     rest_files.extend(fp.readlines())
    # with open(validation_path) as fp:
    #     rest_files.extend(fp.readlines())
    ## The rest files are generated using the hashing function!!!!

    # exclude_files = set([f.rstrip() for f in rest_files])
    classes = set()

    for dname, _, files in os.walk(data_root, topdown=False):
        cmd = os.path.basename(os.path.normpath(dname))
        if (cmd not in exclude_dirs) and len(_)==0:
            print("Reading", cmd, "...")
            classes.add(cmd)
            for fname in files:
                addfile = os.path.join(cmd, fname)
                if fname.endswith(".wav") and which_set(fname, 10, 10)=="training":
                    train_files.append(addfile)
                else:
                    rest_files.append(addfile)

    print("\nWriting training %s ..."%train_path)
    with open(train_path, 'w') as fp:
        fp.writelines("\n".join(train_files))

    print("\nWriting rest %s ..."%rest_path)
    with open(rest_path, 'w') as fp:
        fp.writelines("\n".join(rest_files))

    class_ix = {c:ix for ix,c in enumerate(sorted(classes))}
    class_path = os.path.join(data_root, "DICT_ix_class.cpkl")
    print("\nWriting %s ..."%class_path)
    with open(class_path, 'wb') as fp:
        pickle.dump(class_ix, fp)

    print("\nDone.")


MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M
def which_set(filename, validation_percentage, testing_percentage):
    """Determines which data partition the file should belong to.

    We want to keep files in the same training, validation, or testing sets even
    if new ones are added over time. This makes it less likely that testing
    samples will accidentally be reused in training when long runs are restarted
    for example. To keep this stability, a hash of the filename is taken and used
    to determine which set it should belong to. This determination only depends on
    the name and the set proportions, so it won't change as other files are added.

    It's also useful to associate particular files as related (for example words
    spoken by the same person), so anything after '_nohash_' in a filename is
    ignored for set determination. This ensures that 'bobby_nohash_0.wav' and
    'bobby_nohash_1.wav' are always in the same set, for example.

    Args:
      filename: File path of the data sample.
      validation_percentage: How much of the data set to use for validation.
      testing_percentage: How much of the data set to use for testing.

    Returns:
      String, one of 'training', 'validation', or 'testing'.
    """
    base_name = os.path.basename(filename)
    # We want to ignore anything after '_nohash_' in the file name when
    # deciding which set to put a wav in, so the data set creator has a way of
    # grouping wavs that are close variations of each other.
    hash_name = re.sub(r'_nohash_.*$', '', base_name)
    # This looks a bit magical, but we need to decide whether this file should
    # go into the training, testing, or validation sets, and we want to keep
    # existing files in the same set even if more files are subsequently
    # added.
    # To do that, we need a stable way of deciding based on just the file name
    # itself, so we do a hash of that and then use that to generate a
    # probability value that we use to assign it.
    hash_name_hashed = hashlib.sha1(hash_name.encode('utf-8')).hexdigest()
    percentage_hash = ((int(hash_name_hashed, 16) %
                        (MAX_NUM_WAVS_PER_CLASS + 1)) *
                       (100.0 / MAX_NUM_WAVS_PER_CLASS))
    if percentage_hash < validation_percentage:
      result = 'validation'
    elif percentage_hash < (testing_percentage + validation_percentage):
      result = 'testing'
    else:
      result = 'training'
    return result


def create_data_hdf5(root):
    """
    Expects training_list.txt, rest_list.txt and DICT_ix_class.cpkl in the root path.
    It will generate training.h5py and rest.h5py files.
    Currently uses librosa.
    """

    train_read_path = os.path.join(root, "training_list.txt")
    rest_read_path = os.path.join(root, "rest_list.txt")
    train_write_path = os.path.join(root, "training.h5py")
    rest_write_path = os.path.join(root, "rest.h5py")
    class_path = os.path.join(root, "DICT_ix_class.cpkl")

    assert os.path.isfile(train_read_path), "training_list.txt does not exist in %s"%root
    assert os.path.isfile(rest_read_path), "rest_list.txt does not exist in %s"%root
    assert os.path.isfile(class_path), "DICT_ix_class.cpkl does not exist in %s"%root
    assert not os.path.isfile(train_write_path), "training.h5py already exists in %s"%root
    assert not os.path.isfile(rest_write_path), "rest.h5py already exists in %s"%root

    with open(class_path, "rb") as fp:
        ix_class = pickle.load(fp)

    def stupid(fname):
        print("Processing", fname)
        with open(fname) as fp:
            wavfiles = [line.rstrip() for line in fp.readlines()]

        classes = [line.split('/')[0] for line in wavfiles]
        classes = [ix_class[c] for c in classes]
        wavpaths = [os.path.join(root, line) for line in wavfiles]

        results = [load_audio_16k(p) for p in wavpaths]
        return np.vstack(results), to_categorical(classes, len(ix_class))

    def write_h5py(path, data, targets):
        with h5py.File(path, 'w') as f:
            grp = f.create_group("subgroup")
            grp.create_dataset("data", (len(data), SRATE), chunks=True, data=data)
            grp.create_dataset("targets", (len(data), len(ix_class)), data=targets)

        print("Created", path)

    train_wavs, train_targets = stupid(train_read_path)
    write_h5py(train_write_path, train_wavs, train_targets)

    rest_wavs, rest_targets = stupid(rest_read_path)
    write_h5py(rest_write_path, rest_wavs, rest_targets)

    print("\nDone.")


if __name__ == '__main__':
    data_root = sys.argv[1]
    print("\nRoot provided:", data_root)
    import time
    _st = time.time()

    make_training_rest_list(data_root)
    create_data_hdf5(data_root)

    print("\n\nTime taken:", (time.time() - _st)/60, "mins.")
