## Utility codes for data, eval, plotting, etc.

from __future__ import print_function
import os
import sys
import librosa

def _make_training_rest_list(data_root, exclude_dirs = ["_background_noise_"]):
    assert os.path.isdir(data_root), "directory %s does not exist"%data_root
    train_path = os.path.join(data_root, "training_list.txt")
    rest_path = os.path.join(data_root, "rest_list.txt")
    assert not os.path.isfile(train_path), "training_list.txt already exists in %s"%data_root
    assert not os.path.isfile(rest_path), "rest_list.txt already exists in %s"%data_root

    validation_path = os.path.join(data_root, "validation_list.txt")
    test_path = os.path.join(data_root, "testing_list.txt")
    train_files = []
    rest_files = []

    print("\nLoading files to exclude from training...")
    with open(test_path) as fp:
        rest_files.extend(fp.readlines())
    with open(validation_path) as fp:
        rest_files.extend(fp.readlines())

    exclude_files = set([f.rstrip() for f in rest_files])

    for dname, _, files in os.walk(data_root, topdown=False):
        cmd = os.path.basename(os.path.normpath(dname))
        if (cmd not in exclude_dirs) and len(_)==0:
            print("Reading", cmd, "...")
            for fname in files:
                addfile = os.path.join(cmd, fname)
                if fname.endswith(".wav") and (addfile not in exclude_files):
                    train_files.append(addfile)

    print("\nWriting %s ..."%train_path)
    with open(train_path, 'w') as fp:
        fp.writelines("\n".join(train_files))

    print("\nWriting %s : test + validation ..."%rest_path)
    with open(rest_path, 'w') as fp:
        fp.writelines(rest_files)

    print("\nDone.")

if __name__ == '__main__':
    data_root = sys.argv[1]
    print("\nRoot provided:", data_root)
    _make_training_rest_list(data_root)
