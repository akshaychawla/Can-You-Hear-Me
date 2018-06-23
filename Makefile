DATA_PATH=$(HOME)/.kaggle/competitions/tensorflow-speech-recognition-challenge/train/audio

.DEFAULT_GOAL := help
help:
	@echo ""
	@echo "COMMANDS:"
	@echo "  clean          Remove all generated files."
	@echo "  download       Create data dir, download and extract the training data."
	@echo "  setup          Create checkpoint, log directories."
	@echo "  vanilla        Assume that data in DATA_PATH is ready for splitting and h5."
	@echo "  kaggle         Add silence clips. Then split and dump to h5."
	@echo ""
	@echo "VARIABLES:"
	@echo "  DATA_PATH      Root for all data and extracted files."
	@echo "  $(DATA_PATH)"
	@echo ""

clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	@echo ""

download:
	mkdir $(DATA_PATH)
	wget -P $(DATA_PATH) http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz
	tar -xvzf $(DATA_PATH)/speech_commands_v0.01.tar.gz --directory $(DATA_PATH)
	@echo ""

setup:
	mkdir logs
	mkdir checkpoints
	@echo ""

vanilla:
	python utils.py $(DATA_PATH)
	@echo ""

kaggle:
	python kaggle_utils.py $(DATA_PATH) silence
	python utils.py $(DATA_PATH)
	@echo ""
