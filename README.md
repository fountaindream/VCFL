**Related Projects:**
- [Strong Triplet Loss Baseline](https://github.com/huanghoujing/person-reid-triplet-loss-baseline)
- [Strong Identification Loss Baseline](https://github.com/huanghoujing/beyond-part-models)
- [AlignedReID: Surpassing Human-Level Performance in Person Re-Identification](https://arxiv.org/abs/1711.08184) using [pytorch](https://github.com/pytorch/pytorch).

If you adopt AlignedReID in your research, please cite the paper
```
@InProceedings{Liu_2019_ICCV,
author = {Liu, Fangyi and Zhang, Lei},
title = {View Confusion Feature Learning for Person Re-Identification},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {October},
year = {2019}
}
```

# TODO List

- Model
  - [x] ResNet-50
- Loss
  - [x] Triplet Global Loss
  - [x] Center Loss
  - [x] Sift Loss
  - [x] Confusion Loss
- Testing
  - [x] Re-Ranking

# Installation

**NOTE: This project now does NOT require installing, so if you has installed `aligned_reid` before, now you have to uninstall it by `pip uninstall aligned-reid`.**

It's recommended that you create and enter a python virtual environment, if versions of the packages required here conflict with yours.

I use Python 2.7 and Pytorch 0.3. For installing Pytorch, follow the [official guide](http://pytorch.org/). Other packages are specified in `requirements.txt`.

```bash
pip install -r requirements.txt
```

Our code is partly based on AlignedReID-Re-Production-Pytorch, you can clone the repository:

```bash
git clone https://github.com/huanghoujing/AlignedReID-Re-Production-Pytorch.git
cd AlignedReID-Re-Production-Pytorch
```

# Dataset Preparation

Inspired by Tong Xiao's [open-reid](https://github.com/Cysu/open-reid) project, dataset directories are refactored to support a unified dataset interface.

Transformed dataset has following features
- All used images, including training and testing images, are inside the same folder named `images`
- Images are renamed, with the name mapping from original images to new ones provided in a file named `ori_to_new_im_name.pkl`. The mapping may be needed in some cases.
- The train/val/test partitions are recorded in a file named `partitions.pkl` which is a dict with the following keys
  - `'trainval_im_names'`
  - `'trainval_ids2labels'`
  - `'train_im_names'`
  - `'train_ids2labels'`
  - `'val_im_names'`
  - `'val_marks'`
  - `'test_im_names'`
  - `'test_marks'`
- Validation set consists of 100 persons (configurable during transforming dataset) unseen in training set, and validation follows the same ranking protocol of testing.
- Each val or test image is accompanied by a mark denoting whether it is from
  - query (`mark == 0`), or
  - gallery (`mark == 1`), or
  - multi query (`mark == 2`) set

## Market1501

You can download what I have transformed for the project from [Google Drive](https://drive.google.com/open?id=1CaWH7_csm9aDyTVgjs7_3dlZIWqoBlv4) or [BaiduYun](https://pan.baidu.com/s/1nvOhpot). Otherwise, you can download the original dataset and transform it using my script, described below.

Download the Market1501 dataset from [here](http://www.liangzheng.org/Project/project_reid.html). Run the following script to transform the dataset, replacing the paths with yours.

```bash
python script/dataset/transform_market1501.py \
--zip_file ~/Dataset/market1501/Market-1501-v15.09.15.zip \
--save_dir ~/Dataset/market1501
```

## CUHK03

We follow the new training/testing protocol proposed in paper
```
@article{zhong2017re,
  title={Re-ranking Person Re-identification with k-reciprocal Encoding},
  author={Zhong, Zhun and Zheng, Liang and Cao, Donglin and Li, Shaozi},
  booktitle={CVPR},
  year={2017}
}
```
Details of the new protocol can be found [here](https://github.com/zhunzhong07/person-re-ranking).

You can download what I have transformed for the project from [Google Drive](https://drive.google.com/open?id=1Ssp9r4g8UbGveX-9JvHmjpcesvw90xIF) or [BaiduYun](https://pan.baidu.com/s/1hsB0pIc). Otherwise, you can download the original dataset and transform it using my script, described below.

Download the CUHK03 dataset from [here](http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html). Then download the training/testing partition file from [Google Drive](https://drive.google.com/open?id=14lEiUlQDdsoroo8XJvQ3nLZDIDeEizlP) or [BaiduYun](https://pan.baidu.com/s/1miuxl3q). This partition file specifies which images are in training, query or gallery set. Finally run the following script to transform the dataset, replacing the paths with yours.

```bash
python script/dataset/transform_cuhk03.py \
--zip_file ~/Dataset/cuhk03/cuhk03_release.zip \
--train_test_partition_file ~/Dataset/cuhk03/re_ranking_train_test_split.pkl \
--save_dir ~/Dataset/cuhk03
```

## DukeMTMC-reID

You can download what I have transformed for the project from [Google Drive](https://drive.google.com/open?id=1P9Jr0en0HBu_cZ7txrb2ZA_dI36wzXbS) or [BaiduYun](https://pan.baidu.com/s/1miIdEek). Otherwise, you can download the original dataset and transform it using my script, described below.

Download the DukeMTMC-reID dataset from [here](https://github.com/layumi/DukeMTMC-reID_evaluation). Run the following script to transform the dataset, replacing the paths with yours.

```bash
python script/dataset/transform_duke.py \
--zip_file ~/Dataset/duke/DukeMTMC-reID.zip \
--save_dir ~/Dataset/duke
```

## Configure Dataset Path

The project requires you to configure the dataset paths. In `aligned_reid/dataset/__init__.py`, modify the following snippet according to your saving paths used in preparing datasets.

```python
# In file aligned_reid/dataset/__init__.py

########################################
# Specify Directory and Partition File #
########################################

if name == 'market1501':
  im_dir = ospeu('~/Dataset/market1501/images')
  partition_file = ospeu('~/Dataset/market1501/partitions.pkl')

elif name == 'cuhk03':
  im_type = ['detected', 'labeled'][0]
  im_dir = ospeu(ospj('~/Dataset/cuhk03', im_type, 'images'))
  partition_file = ospeu(ospj('~/Dataset/cuhk03', im_type, 'partitions.pkl'))

elif name == 'duke':
  im_dir = ospeu('~/Dataset/duke/images')
  partition_file = ospeu('~/Dataset/duke/partitions.pkl')

elif name == 'combined':
  assert part in ['trainval'], \
    "Only trainval part of the combined dataset is available now."
  im_dir = ospeu('~/Dataset/market1501_cuhk03_duke/trainval_images')
  partition_file = ospeu('~/Dataset/market1501_cuhk03_duke/partitions.pkl')
```

## Evaluation Protocol

Datasets used in this project all follow the standard evaluation protocol of Market1501, using CMC and mAP metric. According to [open-reid](https://github.com/Cysu/open-reid), the setting of CMC is as follows

```python
# In file aligned_reid/dataset/__init__.py

cmc_kwargs = dict(separate_camera_set=False,
                  single_gallery_shot=False,
                  first_match_break=True)
```

To play with [different CMC options](https://cysu.github.io/open-reid/notes/evaluation_metrics.html), you can [modify it accordingly](https://github.com/Cysu/open-reid/blob/3293ca79a07ebee7f995ce647aafa7df755207b8/reid/evaluators.py#L85-L95).

```python
# In open-reid's reid/evaluators.py

# Compute all kinds of CMC scores
cmc_configs = {
  'allshots': dict(separate_camera_set=False,
                   single_gallery_shot=False,
                   first_match_break=False),
  'cuhk03': dict(separate_camera_set=True,
                 single_gallery_shot=True,
                 first_match_break=False),
  'market1501': dict(separate_camera_set=False,
                     single_gallery_shot=False,
                     first_match_break=True)}
```


# Examples

### `ResNet-50` on Market1501
You can train with different scripts. Train_confusion（only include confusion loss）,Train_sift(only include sift loss),Train_cen(only include center loss)

--------------------------------Training------------------------------------
```bash
python script/experiment/train_whole.py \
-d '(0,)' \
-r 1 \
--dataset market1501 \
--ids_per_batch 32 \
--ims_per_id 4 \
--normalize_feature false \
-gm 0.3 \
-glw 1 \
-llw 0 \
-idlw 0 \
-slw 0.1 \
-clw 0.0001 \
-dglw 0 \
--base_lr 3e-4 \
--lr_decay_type exp \
--exp_decay_at_epoch 151 \
--total_epochs 300
```

----------------------------testing with trained model-----------------------

```bash
python script/experiment/train_whole.py \
-d '(0,)' \
--dataset market1501 \
--normalize_feature false \
-glw 1 \
-llw 0 \
-idlw 0 \
--only_test true \
--exp_dir SPECIFY_AN_EXPERIMENT_DIRECTORY_HERE \
--model_weight_file THE_DOWNLOADED_MODEL_WEIGHT_FILE
```

### Log

During training, you can run the [TensorBoard](https://github.com/lanpa/tensorboard-pytorch) and access port `6006` to watch the loss curves etc. E.g.

```bash
# Modify the path for `--logdir` accordingly.
tensorboard --logdir YOUR_EXPERIMENT_DIRECTORY/tensorboard
```

For more usage of TensorBoard, see the website and the help:

```bash
tensorboard --help
```

# References & Credits

- [AlignedReID](https://arxiv.org/abs/1711.08184)
- [open-reid](https://github.com/Cysu/open-reid)
- [In Defense of the Triplet Loss for Person Re-Identification](https://arxiv.org/abs/1703.07737)
- [Re-ranking Person Re-identification with k-reciprocal Encoding](https://github.com/zhunzhong07/person-re-ranking)
- [Market1501](http://www.liangzheng.org/Project/project_reid.html)
- [CUHK03](http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html)
- [DukeMTMC-reID](https://github.com/layumi/DukeMTMC-reID_evaluation)
