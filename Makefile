DATA_PATH=/home/sviper/audio-data

.DEFAULT_GOAL := help
help:
	@echo "COMMANDS:"
	@echo "  clean          Remove all generated files."
	@echo "  download       Create data dir, download and extract the training data."
	@echo "  setup          Create log directories and prepare data."
	@echo ""
	@echo "VARIABLES:"
	@echo "  DATA_PATH      Root for all data and extracted files."
	@echo ""

clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

download:
	mkdir $(DATA_PATH)
	wget -P $(DATA_PATH) http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz
	tar -xvzf $(DATA_PATH)/speech_commands_v0.01.tar.gz --directory $(DATA_PATH)

setup:
	mkdir logs
	mkdir checkpoints

vanilla:
	python utils.py $(DATA_PATH)

kaggle:
	python kaggle_utils.py $(DATA_PATH) silence
