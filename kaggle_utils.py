"""
Utility codes for data, eval, plotting, etc.
"""

from __future__ import print_function
import os
import re
import hashlib
import sys
import librosa
import scipy
import shutil
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

SRATE = 16000
def librosa_load(path, srate):
    audio, _ = librosa.load(path, sr=srate)
    if audio.size < srate:
        return np.pad(audio, (srate - audio.size, 0), mode='constant')
    else:
        return audio[:srate]
load_audio_16k = partial(librosa_load, srate=SRATE)


def data_generator(h5path, batch_size=32):
    """
    Custom generator for loading from h5 files.
    """
    assert os.path.isfile(h5path), "%s does not exist"
    with h5py.File(h5path, 'r') as f:
        data = f["subgroup"]["data"]
        targets = f["subgroup"]["targets"]
        assert len(data) == len(targets), "data, target lengths mismatch"
        while True:
            data_indices = np.arange(len(data)).astype(np.int32)
            np.random.shuffle(data_indices)
            for _index in range(0, len(data_indices), batch_size):
                indices_for_h5 = data_indices[ _index: min(len(data),_index+batch_size) ]
                indices_for_h5 = np.sort(indices_for_h5).tolist()
                x_batch, y_batch = data[indices_for_h5], targets[indices_for_h5]
                yield x_batch, y_batch


def score_generator(test_samples_folder, batch_size=32):
    print("""
    Custom SCORING for KAGGLE! Loading from raw wave files.
    """)
    file_paths, file_names = [], []
    for fname in os.listdir(test_samples_folder):
        fpath = os.path.join(test_samples_folder, fname)
        if os.path.isfile(fpath) and fname.endswith(".wav"):
            file_paths.append(fpath)
            file_names.append(fname)

    s = len(file_names)
    lowers = list(range(0, s, batch_size))
    print("Total test files found...", s)

    for ix in lowers:
        if (ix+batch_size)>s:
            ux = s
        else:
            ux = ix + batch_size

        batch_data = np.vstack([load_audio_16k(fp) for fp in file_paths[ix:ux]])
        batch_targets = file_names[ix:ux]
        yield batch_data, batch_targets


def make_training_list(data_root, exclude_dirs = ["_background_noise_"]):
    """
    Download v0.01 from http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz
    This function will generate and store the training_list.txt .
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
    #
    # exclude_files = set([f.rstrip() for f in rest_files])
    exclude_files = set()
    classes = set()

    for dname, _, files in os.walk(data_root, topdown=False):
        cmd = os.path.basename(os.path.normpath(dname))
        if (cmd not in exclude_dirs) and len(_)==0:
            print("Reading", cmd, "...")
            classes.add(cmd)
            for fname in files:
                addfile = os.path.join(cmd, fname)
                if fname.endswith(".wav") and (addfile not in exclude_files):
                    train_files.append(addfile)

    print("\nWriting %s ..."%train_path)
    with open(train_path, 'w') as fp:
        fp.writelines("\n".join(train_files))

    class_ix = {c:ix for ix,c in enumerate(sorted(classes))}
    class_path = os.path.join(data_root, "DICT_ix_class.cpkl")
    print("\nWriting %s ..."%class_path)
    with open(class_path, 'wb') as fp:
        pickle.dump(class_ix, fp)

    print("\nDone.")


def create_data_hdf5(root):
    """
    Expects training_list.txt, rest_list.txt and DICT_ix_class.cpkl in the root path.
    It will generate training.h5py files.
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

    # rest_wavs, rest_targets = stupid(rest_read_path)
    # write_h5py(rest_write_path, rest_wavs, rest_targets)

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


def add_silence(data_root):
    silence_dir = os.path.join(data_root, "silence")
    bnoise_dir = os.path.join(data_root, "_background_noise_")
    cat_file = os.path.join(bnoise_dir, "dude_miaowing.wav")

    assert os.path.isdir(silence_dir) is False, "ERROR. silence dir already exists."
    assert os.path.isdir(bnoise_dir), "ERROR. _background_noise_ dir does not exist."
    assert os.path.isfile(cat_file), "ERROR. dude_miaowing.wav does not exist."

    os.makedirs(silence_dir)

    ## clean the cat file.
    fs1, y1 = scipy.io.wavfile.read(cat_file)
    drop_points = [(0, 4), (20, 23), (39, 46), (50, 52)]
    use_points = [(4, 20), (23, 39), (46, 50), (52, 62)]

    new_audio = []
    for start, stop in use_points:
        new_audio.append(y1[start*fs1:stop*fs1])
    new_audio = np.hstack(new_audio)

    print("Saving new cat audio...")
    shutil.move(cat_file, os.path.join(bnoise_dir, "new_dude_miaowing.wav.orig"))
    scipy.io.wavfile.write(cat_file, fs1, new_audio)
    print("Done.")

    for fname in os.listdir(bnoise_dir):
        if fname.endswith(".wav"):
            path = os.path.join(bnoise_dir, fname)
            audio, _ = librosa.load(path, sr=SRATE)
            alen = len(audio)
            print(fname, alen, alen//SRATE)

            for count, ix in enumerate(range(0, len(audio), SRATE)):
                up = ix + SRATE
                if up > alen:
                    up = alen

                temp_audio = audio[ix:up]
                temp_path = os.path.join(silence_dir, fname[:2]+"%d_nohash_%d.wav"%(count, count))

                if len(temp_audio) >= 10000:
                    if len(temp_audio) < SRATE:
                        temp_audio = np.pad(temp_audio, (SRATE - len(temp_audio), 0), mode='constant')
                    else:
                        temp_audio = temp_audio[:SRATE]

                    librosa.output.write_wav(temp_path, temp_audio, SRATE)

            assert up==alen, "YO! Your indexing limits are not matching!!!"


if __name__ == '__main__':
    data_root = sys.argv[1]
    what = sys.argv[2]
    print("\nRoot provided:", data_root)
    print("Mode:", what)

    import time
    _st = time.time()

    if what == "data":
        make_training_list(data_root)
        create_data_hdf5(data_root)
        print("\n\nTime taken:", (time.time() - _st)/60, "mins.")

    elif what == "silence":
        add_silence(data_root)
        print("\n\nTime taken:", (time.time() - _st)/60, "mins.")

    else:
        print("Valid options are 'data' or 'silence'")
