Audio Adversarial Examples for Hot Word Detection

1. Install the kaggle cli api using `pip install kaggle`
2. Create and copy your kaggle key (from your profile) to a `.kaggle` dir in your home.
3. Download the competition data using `kaggle competitions download -c tensorflow-speech-recognition-challenge`
4. Check if data has been downloaded in `~/.kaggle/competitions/tensorflow-speech-recognition-challenge/`
5. `cd ~/.kaggle/competitions/tensorflow-speech-recognition-challenge/` and extract train, test data using `7za x train.7z`
6. Run `make` after cloning this repo to see valid recipes.
```

COMMANDS:
  clean          Remove all generated files.
  download       Create data dir, download and extract the training data.
  setup          Create checkpoint, log directories.
  vanilla        Assume that data in DATA_PATH is ready for splitting and h5.
  kaggle         Add silence clips. Then split and dump to h5.

VARIABLES:
  DATA_PATH      Root for all data and extracted files.
  /home/tejaswin.p/.kaggle/competitions/tensorflow-speech-recognition-challenge/train/audio

```
7. `make clean`
8. `make setup`
9. `make kaggle`
10. `python train.py ~/.kaggle/competitions/tensorflow-speech-recognition-challenge/train/audio/training.h5py ~/.kaggle/competitions/tensorflow-speech-recognition-challenge/train/audio/rest.h5py`
