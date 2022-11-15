# 一个简单的语音识别模型训练和部署示例

**这个代码库用于在 2022 年 11 月面向本科生和低年级研究生的课程上演示，该课程是小米集团 AIoT 创新大赛的培训模块的一部分。** 

## 简介

本代码库演示了使用 [PyTorch](https://pytorch.org/) 在 [Yesno](https://www.openslr.org/1/) 玩具数据集上训练一个语音识别模型，使能够识别 "YES" 和 "NO" 两个单词。

为了部署模型，本代码库也演示了使用 C++ 语言基于 libtorch 封装一个动态链接库，并使用该库制作了一个简单的命令行程序。

部署代码可以使用目标平台的工具链编译，以便在小米 AIoT 开发实验箱等设备上运行。

为了使代码简单，这里的训练流程做了一些简化，比如训练时固定 batchsize 为 1 以避免补零，以及解码时采用了粗糙但易于实现的 greedy search 等。

## 如何运行

本代码库分训练和部署两部分，训练的代码在 `train` 目录下，部署的代码在 `deploy` 目录下。训练部分生成一个 [TorchScript](https://pytorch.org/docs/stable/jit.html) 格式的模型 `model.pt`，部署部分读取该模型进行识别。

### 1. 环境设置

一个被较为普遍认可的 Python 最佳实践是始终在虚拟环境中开发。常用的虚拟环境包括 [Virtualenv](https://virtualenv.pypa.io) 和 [Conda](https://docs.conda.io/en/latest/)。
当然，如果是在 [Docker](https://www.docker.com) 或 [Codespaces](https://github.com/codespaces)、[Gitpod](https://gitpod.io/) 等容器类环境中开发，则不必使用 Python 虚拟环境。

这里我们用 Virtualenv 建立一个虚拟环境：

```bash
$ python3 -m venv venv
$ . venv/bin/activate
```

在虚拟环境下，安装所需要的 Python 依赖包：

```bash
(venv)$ pip install -r requirements.txt
```

安装 [Jupyter](https://jupyter.org/) 用于交互地运行训练代码：
```bash
(venv)$ pip install jupyter
```

安装基本构建工具用于编译部署代码。对于如 [Ubuntu](https://ubuntu.com/) 等基于 [Debian](https://www.debian.org/) 的发行版，可以使用 apt 安装必要的软件包：
```plain
# apt install build-essential cmake
```

### 2. 模型训练

模型训练的全部代码在 [`train.ipynb`](https://github.com/jimbozhang/yesno-example-for-undergraduates/blob/main/train/train.ipynb) 中。执行整个流程即可完成模型训练。

```bash
(venv)$ cd train
(venv)$ jupyter-notebook train.ipynb
```

上述操作会生成一个 notebook 链接，用浏览器打开即可。如果希望直接运行训练流程而不是使用 notebook，也可以把 `ipynb` 文件转为 `py` 文件：

```bash
(venv)$ jupyter-nbconvert --to python train.ipynb
(venv)$ python train.py
```

上述两种方式是等价的，都会训练一个模型，并以 TorchScript 形式保存到文件 `train/model.pt`。

这个模型的输入是声音波形的采样点序列，提取 [MFCC](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum) 声学特征的操作在模型内部完成。输出是每帧语音在各 token 上的后验概率，这里的 token 指 "Y", "E", "S", "N", "O"、空格以及 *blank* 这七种符号其中之一。*blank* 是 [CTC](http://www.cs.toronto.edu/~graves/icml_2006.pdf) 损失函数中的概念，[Sequence Modeling With CTC](https://distill.pub/2017/ctc/) 是一个非常好的 CTC 教程。

为了从模型的输出后验概率得到识别结果的字符串，最简单的方法是 greedy search，即直接把每帧得分最高的 token 作为该帧的识别结果，然后相邻去重即得到整句识别结果，这个过程在 [`train.ipynb`](https://github.com/jimbozhang/yesno-example-for-undergraduates/blob/main/train/train.ipynb) 中有详细演示。

由于 Yesno 集极小的数据量，整个模型训练流程在普通的个人计算机上用两分钟左右即可完成，且不需要 GPU 参与计算。

### 3. 部署

为了在实际设备上进行识别，我们把识别过程封装成 C++ 动态链接库，其核心是神经网络推理。
推理的方案有非常多种，为了实现简单，这里我们选用了 PyTorch 自带的 libtorch。

首先下载 libtorch 的代码到 `deploy` 目录下：
```bash
$ cd deploy
$ ./install_libtorch.sh
```

在 `deploy` 目录下，使用 [CMake](https://cmake.org/) 进行编译：
```bash
$ mkdir build
$ cd build

$ cmake -DCMAKE_PREFIX_PATH=`realpath ../libtorch` ..
$ make
```

如果顺利，会在 `deploy/build` 目录下生成动态链接库文件 `libyesno.so` 和可执行文件 `example-app`。
`example-app` 读入模型文件和一个声音文件，在屏幕上打印识别结果。我们尝试运行一下：

```bash
$ ./example-app
usage: example-app <model-path> <wav-path>

$ ./example-app ../../train/model.pt ../../train/waves_yesno/0_0_0_0_1_1_1_1.wav
NO NO NO NO YES YES YES YES

$ ./example-app ../../train/model.pt ../../train/waves_yesno/0_0_1_0_1_0_1_1.wav
NO NO YES NO YES NO YES YES

$ ./example-app ../../train/model.pt ../../train/waves_yesno/1_1_1_1_1_1_1_1.wav
YES YES YES YES YES YES YES YES
```

如果要基于识别算法开发新的应用程序，使用 `yesno.h` 和 `libyesno.so` 这两个文件即可。
当然，`libyesno.so` 需要使用目标平台上的构建环境进行编译。