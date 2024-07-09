# PsPG - Pseudo Prompt Generating

<br> [![Awesome Screenshot](https://img.shields.io/badge/Paper-PDF-red "Awesome Website")](https://arxiv.org/abs/2405.06468) [![Pretrained models](https://img.shields.io/badge/Pretrained-Models-blue)](https://drive.google.com/drive/folders/1aGHNXcWmaMrBnLr8a4AJpOlnwA6RhSP9?usp=sharing) ![GitHub Repo stars](https://img.shields.io/github/stars/fallingnight/PsPG)  ![GitHub Repo forks](https://img.shields.io/github/forks/fallingnight/PsPG)


Here is the official implementation of [Pseudo Prompt Generating (PsPG)](https://arxiv.org/abs/2405.06468). 

> Yaoqin Ye, Junjie Zhang and Hongwei Shi. 2024. Pseudo-Prompt Generating in Pre-trained Vision-Language Models for Multi-Label Medical Image Classification. In Pattern Recognition and Computer Vision: 7th Chinese Conference, PRCV 2024

## Intro
 Pseudo-Prompt Generating **(PsPG)** is designed to address _Multi-Label Medical Image Classification_ task, capitalizing on the priori knowledge of multi-modal features. Featuring a RNN-based decoder, **PsPG** autoregressively generates class-tailored embedding vectors, i.e., pseudo-prompts.

## Via `python` 

We recommend you use conda and provide the `environment.yaml`. You can import and activate the environment by:

```shell
conda env create -f environment.yaml
conda activate pspg
```

### Get Started

You can use `python train.py --help` or `python val.py --help` to learn the usage of the python scripts for training and validating. Next section we'll introduce some details.

For reference, we provide two typical commands for training and validating, respectively:

```shell
# train
python train.py -nc configs/model/RN50.yaml -dc configs/dataset/example.yaml \
  --dataset_dir datasets/example --checkpoint weights/PsPG.pth --decoder_hidden 512 \
  --max_epochs 50 --output_dir output  --test_file_path test.jsonl \
  --val_file_path val.jsonl --train_file_path train.jsonl --start_afresh 
# validate
python val.py -nc configs/model/RN50.yaml -dc configs/dataset/example.yaml \
  --dataset_dir datasets/example --checkpoint weights/PsPG.pth --decoder_hidden 512 \
  --output_dir output  --test_file_path test.jsonl \
  --model_name example-RN50 --save_pred
```



## Via `make` (recommended)

We provide makefile to simplify the process of training and validating the PsPG model, and we **recommend** you use make instead of directly executing the python script.

To use make, you must have linux, msys2 (if in Windows) or OSX (not tested for OSX) environment with make and conda installed. And make sure that conda is in your `PATH`, or `CONDA_EXE` is set to the path of conda executable (usually `path/to/Conda/Scripts/conda`). Otherwise, the make will report that it cannot find conda.

### Get started

Upon the first usage, you can establish the conda environment of running the PsPG model by this command:

```shell
make init
```

Next, you can train the model with the default setting with this command:

```shell
make train
```

Then, you can validate the model with the weights you just trained with this command:

```shell
make val
```

The weights and predicting results will output to the `output` folder by default, use this command to clear the output folder (it will clear the \_\_pycache\_\_ as well):

```shell
make clean
```

### Plug in Dataset

We provide an example dataset so that user can run for the first time quickly. For other datasets, please refer to [Datasets](#datasets). 

To plug in a dataset, you should create a dataset folder named after the dataset in the `datasets` folder, create two (or three) *datamap files*, and place the datamap files (will be explained afterwards) and the pictures into the folder.

A datamap file consists of lines of jsons (jsonl), each decribing one data entry of a dataset. A data entry consists of:

+ id (optional)
+ path to the image (relative to the dataset root directory)
+ text (optional)
+ ground truth labels

The datamap files required by a dataset should be named as `test.jsonl`, `val.jsonl` (if need) and `train.jsonl` (or you could modify `Makefile` to use custom setting).

You can refer to `datasets/example` to further learn the format of a dataset.

Append `DATASET=[dataset]` to your *make* command to specify the [dataset], and the variable is set to `example` by default.

### Select The Weights

By default, *make* will check if there is any file of checkpoint output during training (to be exact if there is `last.pth` in the output folder). If so, *make* will select it as the checkpoint for training and validating.

Otherwise, make will next check if there is `weights/PsPG.pth`. If so, this will be used, or, make will try to train the model from scratch in case of `train` target is specified or report an error in case of `val` is specified.

If you want to specify a target checkpoint, append `CHECKPOINT=[yourCheckpointPath]` to your *make* command. 

### Change the Image Encoder of CLIP

You can change the backbone type via appending `MODEL=model` to the *make* command. We use `model=RN50` by default. The provided weights are also for resnet-50.

You can use other pre-trained clip provided by openai like RN50, RN101, ViT-B.

### Advanced usage

There are other variables in makefile for more flexible usage. **If you want to modify more detailedly, please refer to `utils/cfg_builder.py`.**

#### `EPOCH`

You can control the max train epochs by specifying the `EPOCH` variable.

#### `MODEL_NAME`

The model name is used only when saving the validating result. Results will be output to `output/$(MODEL_NAME)` by default.

#### `OUTPUT_DIR`

You can change the name of the output folder by appending `OUTPUT_DIR=output_dir`. This variable is useful when you want to execute two or more processes simultaneously.


#### `EXTRA_FLAGS`

You can add extra flags to the python script by specifying the `EXTRA_FLAGS` variable in `Makefile` (See `Makefile`for details).

Or you can append `EXTRAFLAGS="--option1 arg1 --option2 arg2..."` **(do not add extra spaces!!)** to the *make* command.

If you wonder what flags are provided by the python script, execute:

+ `make val VAL_FLAGS=--help` (for `val.py`)
+ `make train TRAIN_FLAGS=--help` (for `train.py`)

If you want to override the flag specified by make, use `VAL_FLAG="..."` or `TRAIN_FLAGS="..."`.

We recommend you to refer to the source code. 

## Data and Processing
### Datasets

Datasets for training or testing can be accessed via the following links: 

+ [Chestxray](https://nihcc.app.box.com/v/ChestXray-NIHCC)
+ [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/)
+ [MIMIC-CXR](https://physionet.org/content/mimic-cxr-jpg/2.1.0/)
+ [PadChest](http://bimcv.cipf.es/bimcv-projects/padchest/)
+ [VinDr-CXR](https://physionet.org/content/vindr-cxr/1.0.0/)

In order to access to some of these datasets, you should be credentialed.

Notice that due to restriction, Private-CXR is not provided. For any research requests, please contact the corresponding author.

### Processing Scripts
We provide some scripts can be run independently to preprocess datasets or calculate metrics. For more information, please refer to `utils/independ_metrics` and `utils/preprocess`.

## Pre-trained Weights

We provide one pre-trained weight: [PsPG](https://drive.google.com/file/d/1Z6EBzMmRZNBhIXWsomtwCtFdNnwuJxRJ/view?usp=drive_link). This weight was trained on CheXpert, use RN50 as backbone, with hidden size 512, seq length 16.

In addition, we provide another without parameters of the prompt learner: [CLIP-finetuned](https://drive.google.com/file/d/1jpAMN09j4p3VtyCWbYtNprRi6gRhp41B/view?usp=drive_link). This weight was finetuned on MIMIC-CXR.



