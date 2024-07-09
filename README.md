# PSPG - PSeudo Prompt Generating

Here is the official implementation of PSeudo Prompt Generating. The paper is present at [arxiv](https://google.com).

本仓库是 PsPG 的官方实现。对应论文可见于 arxiv。

## Usage

### Via `make` (recommendded)

We provide makefile to simplify the process of training and validating the PsPG model, and we **recommend** you to use make instead of directly executing the python script.

我们提供了 Makefile 脚本来简化训练和测试 PsPG 模型的操作流程，并且推荐您使用 make 而不是直接调用 python 程序。

To use make, you must have linux, msys2 (if in Windows) or OSX (not tested for OSX) environment with make and conda installed. And make sure that conda is in your `PATH`, or `CONDA_EXE` is set to the path of conda executable (usually `path/to/Conda/Scripts/conda`). Otherwise, the make will report that it cannot find conda.

要使用 make，您必须使用装有 make 的 linux，msys2（如果在 windows 下）或 OSX 环境（未针对 OSX 进行测试）。并且确保 conda 在您的 `PATH` 中或 `CONDA_EXE` 被设置成 conda 可执行程序的路径（它通常是 `path/to/Conda/Scripts/conda`）。否则，make 会报告它找不到 conda。

#### Get started

Upon first usage, you can establish the conda environment of running PsPG model by this command:

第一次使用时，您可以通过以下命令来建立可供 PsPG 模型运行的 conda 环境：

```shell
make init
```

Next, you can train the model with default setting with this command:

接着，您可以执行以下命令来以默认设置训练模型：

```shell
make train
```

Then, you can validate the model with the weights you just trained with this command:

然后，您可以通过以下命令测试刚刚训练出的模型：

```shell
make val
```

The weights and predicting results will output to the `output` folder by default, use this command to clear the output folder (it will clear the \_\_pycache\_\_ as well):

模型的权重和预测结果会被默认输出到 `output` 文件夹中，用以下命令清空 `output` 文件夹（同时也会清空 \_\_pycache\_\_）：

```shell
make clean
```

#### Plug in Dataset

We provide an example dataset with only one picture for testing and do not provide any other dataset. This is because the datasets used by PsPG model are all have limitative copyrights and cannot be distributed freely. So you must get these datasets yourself and plug them into the project. 

我们提供了一个只有一张图片的样例数据集并且不提供任何其他数据集，这是因为 PsPG 模型使用的数据集都有限制性版权，不能够被自由分发。所以您必须自己获取这些数据集并且将它们插入工程。

The supported datasets are:

支援的数据集有：

+ Chestxray
+ Chexpert
+ CXR
+ MIMIC
+ PadChest
+ Vindr

To plug in a dataset listed above, you should create a dataset folder named after the dataset in `datasets` folder, create two *datamap files*, and place the datamap files (will be explained afterwards) and the pictures into the folder.

要插入一个上面列出的数据集，您需要在 `datasets` 文件夹中创建一个以数据集名称命名的数据集文件夹，创建两个数据映射文件（后文我们会解释），并将数据映射文件和数据图片都放入这个文件夹。

A datamap file is consist of lines of jsons (jsonl), each line of which decribes one data entry of dataset. A data entry is consist of:

一个数据映射文件由若干行 json 构成（即 jsonl）。每行 json 描述了一个数据集中的条目。一个数据条目由如下成分组成：

+ entry id
+ path to the image (relative to the dataset root directory)
+ text (optional)
+ ground truth labels

You can refer to `datasets/example/test.jsonl` to better understand the format of datamap files.

您可以参考 `datasets/example/test.jsonl` 来更好理解数据映射文件的格式。

The two datamap files required by a dataset should be named as `test.jsonl` and `train.jsonl`. The former will be used for validating while the latter will be used for trainning.

每个数据集需要的两个数据映射文件需要被命名为 `test.jsonl` 和 `train.jsonl`，前者用于测试而后者用于训练。

Similarly, you can refer to `datasets/example` to further learn the format of a dataset.

类似地，您也可以参考 `datasets/example` 来进一步了解数据集的格式。

Append `DATASET=dataset` to the make command to specify the dataset to be used by the PsPG model, and the variable is set to `example` by default.

在 make 命令后添加 `DATASET=dataset` 来指定 PsPG 模型使用的数据集，这个变量被默认设置为 `example`。

#### Select The Weights/Checkpoint

By default, make will check if there is any file of weights (or checkpoint, we will call the weights as checkpoint afterwards) output during trainning (to be exact, if there is `last.pth` in the output folder). And if so, make will select it as the checkpoint for training and validating.

默认状况下，make 会查看是否有训练阶段输出的权重文件（即检查点，下同），具体来说，是查看输出文件夹是否有 `last.pth`。如果是的话，make 就会选择它作为训练和测试的检查点。

Otherwise, make will next check if there is `weights/PsPG.pth`. If so, this checkpoint will be used, or, make will try to train the model from scratch in case of `train` target is specified or report error in case of `val` is specified (Validating a model with random weights is meaningless)

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

You can control the epoches by specifying the `EPOCH` variable.

##### `MODEL_NAME`

The model name is used only when validating. And the predicting results will be output to `output/$(MODEL_NAME)` by default.

##### `OUTPUT_DIR`

You can change the name of output folder by appending `OUTPUT_DIR=output_dir`. This variable is useful when you want to execute two or more processes simultaneously so that the outputs will not get mixed in the same folder (particularly, the `last.pth`).

Note that if you once specify this variable, you should do so during cleanning as well.

##### `EXTRA_FLAGS`

You can add extra flags to the python script by specifying `EXTRA_FLAGS` variable in makefile (See makefile for details).

Or you can append `EXTRAFLAGS="--option1 arg1 --option2 arg2..."` (do not add extra spaces) to the make command, while we recommend you to modify the makefile directly.

Note that you cannot add a flag that has been controlled by make.

If you wonder what flags are provided by the python script, execute:

+ `make val VAL_FLAGS=--help` (for `val.py`)
+ `make train TRAIN_FLAGS=--help` (for `train.py`)

Similarly, if you want to override the flag specified by make, use `VAL_FLAG="..."` or `TRAIN_FLAGS="..."`.

If you still do not understand the meaning of some flags, please refer to the the next section or read the source code. 

### Via `python` (advanced)

If you do not want to use make, you should configure the python environment yourself and handcraft the proper flags to invoke python script.

We recommend you to use conda and provide the `environment.yaml` that describes the environment exported by `conda env export`. You can import and activate the environment by:

```shell
conda env create -f environment.yaml
conda activate pspg
```

#### Get Started

You can use `python train.py --help` or `python val.py --help` to learn the usage of the python scripts for trainning and validating.

For reference, we provide two typical command for trainning and validating, respectively:

```shell
# train
python train.py -nc configs/model/RN50.yaml -dc configs/dataset/example.yaml --dataset_dir datasets/example --checkpoint output/last.pth --decoder_hidden 512 --max_epochs 50 --output_dir output  --test_file_path test.jsonl --val_file_path test.jsonl --train_file_path train.jsonl --start_afresh 
# validate
python val.py -nc configs/model/RN50.yaml -dc configs/dataset/example.yaml --dataset_dir datasets/example --checkpoint output/last.pth --decoder_hidden 512 --max_epochs 50 --output_dir output  --test_file_path test.jsonl --model_name example-RN50-last --save_pred
```

## Resources

### Pretrained Weights

We provide a file of pretrained weights for PsPG [here](https://drive.google.com/file/d/1Z6EBzMmRZNBhIXWsomtwCtFdNnwuJxRJ/view?usp=drive_link) and another without prompt learner [here](https://drive.google.com/file/d/1jpAMN09j4p3VtyCWbYtNprRi6gRhp41B/view?usp=drive_link).

