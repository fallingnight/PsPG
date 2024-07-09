# modify this if needed
EXTRA_FLAGS = 
EXTRA_VAL_FLAGS = 
EXTRA_TRAIN_FLAGS =

ifdef CONDA_EXE
  CONDA_EXE := $(shell cygpath $(CONDA_EXE))
endif

ifndef CONDA_EXE
  CONDA_EXE = $(shell which conda)
endif

# find conda
ifeq ($(CONDA_EXE), )
  $(error CONDA_EXE not set or conda executable not in PATH!)
endif

CONDA_DIR = $(shell dirname $(CONDA_EXE))
CONDA = $(CONDA_DIR)/conda

ifdef MSYSTEM
FIND = /bin/find 
PYTHON = `cygpath $$CONDA_PREFIX`/python
else
FIND = find
PYTHON = $$CONDA_PREFIX/python
endif

# not clean only
ifneq ($(MAKECMDGOALS), clean)

# check conda's presence
conda_exit_v = $(shell $(CONDA) --version >/dev/null; echo $$?)
ifeq ($(conda_exit_v), 127)
  $(error conda not found! It seems that $$CONDA_EXE is not properly set.)
else 
  ifeq ($(conda_exit_v), 0)
    $(info Detected $(shell $(CONDA) --version))
  endif
endif

endif

# configs

OUTPUT_DIR = output
OUTPUT_FLAG = --output_dir $(OUTPUT_DIR)
$(shell mkdir -p $(OUTPUT_DIR))

# mostly no need to change
CONFIG_DIR = configs
DATASETS_DIR = datasets

# use example dataset and rn50
DATASET = example
MODEL = RN50

CONFIG_FLAG = -nc $(CONFIG_DIR)/model/$(MODEL).yaml -dc $(CONFIG_DIR)/dataset/$(DATASET).yaml

# dataset

DATASET_ROOT = $(DATASETS_DIR)/$(DATASET)
DATASET_ROOT_FLAG = --dataset_dir $(DATASET_ROOT)

DATAMAP_FILE_TEST = test.jsonl
DATAMAP_FILE_VAL = val.jsonl
DATAMAP_FILE_TRAIN = train.jsonl

DATAMAP_FILE_FLAGS_ALL = $(DATAMAP_FILE_FLAGS_TEST) --val_file_path $(DATAMAP_FILE_VAL) --train_file_path $(DATAMAP_FILE_TRAIN)

DATAMAP_FILE_FLAGS_TEST = --test_file_path $(DATAMAP_FILE_TEST)

# misc

DECODER_HIDDEN = 512
DECODER_HIDDEN_FLAG = --decoder_hidden $(DECODER_HIDDEN)

# checkpoint

# no user-defined checkpoint 
ifndef CHECKPOINT
  # no last checkpoint
  ifeq ($(shell test -e $(OUTPUT_DIR)/last.pth; echo $$?), 1)
    # no applicable ckpt, left CHECKPOINT undefined
    ifeq ($(shell test -e weights/PsPG.pth; echo $$?), 1)
      # test with no ckpt
      ifneq ($(filter val, $(MAKECMDGOALS)), )
        $(error val with no checkpoint!)
      endif
    else
      # fall back to PsPG checkpoint
      CHECKPOINT = weights/PsPG.pth
    endif
  else
    # fallback to last checkpoint
    CHECKPOINT = $(OUTPUT_DIR)/last.pth
  endif
endif

ifdef CHECKPOINT
CHECKPOINT_FLAG = --checkpoint $(CHECKPOINT)
endif

# epoch

EPOCH = 50
EPOCH_FLAG = --max_epochs $(EPOCH)

# model_name

ifdef CHECKPOINT
MODEL_NAME = $(DATASET)-$(MODEL)-$(basename $(shell basename $(CHECKPOINT)))
MODEL_NAME_FLAG = --model_name $(MODEL_NAME)
endif

# val and train flags

GENERAL_FLAGS = $(CONFIG_FLAG) $(DATASET_ROOT_FLAG) $(CHECKPOINT_FLAG) $(DECODER_HIDDEN_FLAG) $(EPOCH_FLAG) $(OUTPUT_FLAG) $(EXTRA_FLAGS)

VAL_FLAGS = $(GENERAL_FLAGS) $(DATAMAP_FILE_FLAGS_TEST) $(MODEL_NAME_FLAG) --save_pred $(EXTRA_VAL_FALGS)
TRAIN_FLAGS = $(GENERAL_FLAGS) $(DATAMAP_FILE_FLAGS_ALL) --start_afresh $(EXTRA_TRAIN_FLAGS)

init:
	@$(CONDA) init
	@$(CONDA) env create -f environment.yaml

train:
	@eval "$$($(CONDA) shell.posix activate pspg)"; \
	$(PYTHON) train.py $(TRAIN_FLAGS)

val:
	@eval "$$($(CONDA) shell.posix activate pspg)"; \
	$(PYTHON) val.py $(VAL_FLAGS)

clean:
	-@$(FIND) . -name "__pycache__" | xargs rm -rf
	-@rm -rf $(OUTPUT_DIR)

.PHONY: ALL clean init val train