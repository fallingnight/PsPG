# PSPG - PSeudo Prompt Generating

Here is the official implementation of PSeudo Prompt Generating. The paper is presented at [arxiv](https://arxiv.org/abs/2405.06468).

## Usage

### Via `make` (recommended)

We provide makefile to simplify the process of training and validating the PsPG model, and we **recommend** you use make instead of directly executing the python script.

To use make, you must have linux, msys2 (if in Windows) or OSX (not tested for OSX) environment with make and conda installed. And make sure that conda is in your `PATH`, or `CONDA_EXE` is set to the path of conda executable (usually `path/to/Conda/Scripts/conda`). Otherwise, the make will report that it cannot find conda.

#### Get started

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

#### Plug in Dataset

We provide an example dataset with only one picture for testing and do not provide any other dataset. This is because the datasets used by the PsPG model all have limited copyrights and cannot be distributed freely. So you must get these datasets yourself and plug them into the project. 

The supported datasets are:

+ Chestxray
+ Chexpert
+ CXR
+ MIMIC
+ PadChest
+ Vindr

To plug in a dataset listed above, you should create a dataset folder named after the dataset in the `datasets` folder, create two *datamap files*, and place the datamap files (will be explained afterwards) and the pictures into the folder.

A datamap file consists of lines of jsons (jsonl), each decribing one data entry of a dataset. A data entry consists of:

+ entry id
+ path to the image (relative to the dataset root directory)
+ text (optional)
+ ground truth labels

You can refer to `datasets/example/test.jsonl` to better understand the format of datamap files.

The two datamap files required by a dataset should be named as `test.jsonl` and `train.jsonl`. The former will be used for validating while the latter for training.

Similarly, you can refer to `datasets/example` to further learn the format of a dataset.

Append `DATASET=dataset` to the make command to specify the dataset to be used by the PsPG model, and the variable is set to `example` by default.

#### Select The Weights/Checkpoint

By default, make will check if there is any file of weights (or checkpoint, we will call the weights as checkpoint afterward) output during training (to be exact if there is `last.pth` in the output folder). If so, make will select it as the checkpoint for training and validating.

Otherwise, make will next check if there is `weights/PsPG.pth`. If so, this checkpoint will be used, or, make will try to train the model from scratch in case of `train` target is specified or report an error in case of `val` is specified (Validating a model with random weights is meaningless)

If you want to specify a target checkpoint, append `CHECKPOINT=/path/to/checkpoint` to your make command. 

#### Change the model

You can change the vision-language model used by PsPG via appending `MODEL=model` to the make command.

Supported models are:

+ RN50
+ RN101
+ VIT

We use `model=RN50` by default.

#### Advanced usage

There are other variables in makefile for more flexible usage.

##### `EPOCH`

You can control the epochs by specifying the `EPOCH` variable.

##### `MODEL_NAME`

The model name is used only when validating. And the predicting results will be output to `output/$(MODEL_NAME)` by default.

##### `OUTPUT_DIR`

You can change the name of the output folder by appending `OUTPUT_DIR=output_dir`. This variable is useful when you want to execute two or more processes simultaneously so that the outputs will not get mixed in the same folder (particularly, the `last.pth`).

Note that if you once specify this variable, you should do so during cleaning as well.

##### `EXTRA_FLAGS`

You can add extra flags to the python script by specifying the `EXTRA_FLAGS` variable in makefile (See makefile for details).

Or you can append `EXTRAFLAGS="--option1 arg1 --option2 arg2..."` (do not add extra spaces) to the make command, while we recommend you modify the makefile directly.

Note that you cannot add a flag that has been controlled by make.

If you wonder what flags are provided by the python script, execute:

+ `make val VAL_FLAGS=--help` (for `val.py`)
+ `make train TRAIN_FLAGS=--help` (for `train.py`)

Similarly, if you want to override the flag specified by make, use `VAL_FLAG="..."` or `TRAIN_FLAGS="..."`.

If you still do not understand the meaning of some flags, please refer to the source code. 

### Via `python` (advanced)

If you do not want to use make, you should configure the python environment yourself and handcraft the proper flags to invoke the python script.

We recommend you use conda and provide the `environment.yaml` that describes the environment exported by `conda env export`. You can import and activate the environment by:

```shell
conda env create -f environment.yaml
conda activate pspg
```

#### Get Started

You can use `python train.py --help` or `python val.py --help` to learn the usage of the python scripts for training and validating.

For reference, we provide two typical commands for training and validating, respectively:

```shell
# train
python train.py -nc configs/model/RN50.yaml -dc configs/dataset/example.yaml --dataset_dir datasets/example --checkpoint output/last.pth --decoder_hidden 512 --max_epochs 50 --output_dir output  --test_file_path test.jsonl --val_file_path test.jsonl --train_file_path train.jsonl --start_afresh 
# validate
python val.py -nc configs/model/RN50.yaml -dc configs/dataset/example.yaml --dataset_dir datasets/example --checkpoint output/last.pth --decoder_hidden 512 --max_epochs 50 --output_dir output  --test_file_path test.jsonl --model_name example-RN50-last --save_pred
```

## Resources

### Pre-trained Weights

We provide a file of pre-trained weights for PsPG [here](https://drive.google.com/file/d/1Z6EBzMmRZNBhIXWsomtwCtFdNnwuJxRJ/view?usp=drive_link) and another without prompt learner [here](https://drive.google.com/file/d/1jpAMN09j4p3VtyCWbYtNprRi6gRhp41B/view?usp=drive_link).

