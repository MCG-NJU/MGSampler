## Overview
This repo is the implementation of [MGSampler](https://arxiv.org/abs/2104.09952). The code is based on [mmaction2](https://github.com/open-mmlab/mmaction2)
## Dependencies
* GPU: TITAN Xp
* GCC: 7.3
* Python: 3.7.4
* Pytorch: 1.6.0+cu101
* TorchVision: 0.7.0+cu101
* OpenCV: 4.4.0
* MMCV: 1.3.6
* MMaction2: 0.15.0+
## Installation:
a. Create a conda virtual environment and activate it.

```shell
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab
```

b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,

```shell
conda install pytorch torchvision -c pytorch
```

Note: Make sure that your compilation CUDA version and runtime CUDA version match.
You can check the supported CUDA version for precompiled packages on the [PyTorch website](https://pytorch.org/).

c. Install mmcv, we recommend you to install the pre-build mmcv as below.

```shell
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
```
d. Clone the MGSampler repository.

```shell
git clone https://github.com/MCG-NJU/MGSampler.git
```

e. Install build requirements and then install MMAction2.

```shell
pip install -r requirements/build.txt
pip install -v -e .  # or "python setup.py develop"
```

## Data Preparation:
Pleasr refer to the default [MMAction2 dataset setup](https://github.com/open-mmlab/mmaction2/blob/master/docs/data_preparation.md) to set datasets correctly.
## Training 
MGSampler is a sampling strategy to guide the model to choose motion-salient frames. It is very easy to be inserted into the original codes. Here we explain three places we have mainly changed in mmaction2.
* motion represention is generated/saved in .json file.
* in [MGSampler/mmaction/datasets/rawframe_dataset.py ](https://github.com/castle971005/MGSampler/blob/main/mmaction/datasets/rawframe_dataset.py), we add **video_name** and **results["img_diff]**.
* in [MGSampler/mmaction/datasets/pipelines/loading.py](https://github.com/castle971005/MGSampler/blob/main/mmaction/datasets/pipelines/loading.py), we change the function **SampleFrames(object)** to guide the sampling.

For training the model, run the following the code(take sthv1 dataset and tsm model as an example):
```shell
bash tools/dist_train.sh configs/recognition/tsm/tsm_r50_1x1x8_50e_sthv1_rgb.py 8 --validate
```
## License
See [Apache-2.0 License](https://github.com/castle971005/MGSampler/blob/main/LICENSE
)
## Acknowledgement
In addition to the MMAction2 codebase, this repo contains modified codes from:
* [PAN-Pytorch](https://github.com/zhang-can/PAN-PyTorch): for implement of generating feature-level differences.




