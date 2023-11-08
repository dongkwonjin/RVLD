# [ICCV 2023] Recursive Video Lane Detection

### Dongkwon Jin, Dahyun Kim, and Chang-Su Kim


Official implementation for **"Recursive Video Lane Detection"** [[arxiv]](https://arxiv.org/abs/2308.11106) [[paper]](https://openaccess.thecvf.com/content/ICCV2023/html/Jin_Recursive_Video_Lane_Detection_ICCV_2023_paper.html)

<!--
[[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Jin_Harmonious_Semantic_Line_Detection_via_Maximal_Weight_Clique_Selection_CVPR_2021_paper.pdf) [[supp]](http://mcl.korea.ac.kr/~dkjin/cvpr2021/04585-supp.pdf) [[video]](https://www.youtube.com/watch?v=CFQ168_6jw8).
-->

A new dataset called **"OpenLane-V"** is available at [here](https://drive.google.com/file/d/1Jf7g1EG2oL9uVi9a1Fk80Iqtd1Bvb0V7/view?usp=sharing).



<img src="https://github.com/dongkwonjin/RVLD/blob/main/Overview.PNG" alt="overview" width="50%" height="50%" border="0"/>



### Video
<a href="https://www.youtube.com/watch?v=Z0FaOqVrN5w" target="_blank"><img src="https://img.youtube.com/vi/Z0FaOqVrN5w/0.jpg" alt="Video" width="30%" height="20%" /></a>     <a href="https://www.youtube.com/watch?v=rTeyZDXBW1I" target="_blank"><img src="https://img.youtube.com/vi/rTeyZDXBW1I/0.jpg" alt="Video" width="30%" height="20%" /></a>


### Requirements
- PyTorch >= 1.10
- CUDA >= 10.0
- CuDNN >= 7.6.5
- python >= 3.6

### Installation
1. Download repository. We call this directory as `ROOT`:
```
$ git clone https://github.com/dongkwonjin/RVLD.git
```

2. Download [pre-trained model](https://drive.google.com/file/d/1PjbN2uQ-7DgFJjApH1vRx81vEo_Tvn-9/view?usp=sharing) parameters and [preprocessed data](https://drive.google.com/file/d/14JI2BIwJ677_rCBLGQiHvl6IF-n0LIwH/view?usp=sharing) in `ROOT`:
```
$ cd ROOT
$ unzip pretrained.zip
$ unzip preprocessing.zip
```
4. Create conda environment:
```
$ conda create -n RVLD python=3.7 anaconda
$ conda activate RVLD
```
4. Install dependencies:
```
$ conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=10.2 -c pytorch
$ pip install -r requirements.txt
```
Pytorch can be installed on [here](https://pytorch.org/get-started/previous-versions/). Other versions might be available as well.

### Dataset
Download [OpenLane-V](https://drive.google.com/file/d/1Jf7g1EG2oL9uVi9a1Fk80Iqtd1Bvb0V7/view?usp=sharing) into the original OpenLane dataset directory. VIL-100 can be downloaded in [here](https://github.com/yujun0-0/MMA-Net).
    
### Directory structure
    .                           # ROOT
    ├── Preprocessing           # directory for data preprocessing
    │   ├── VIL-100             # dataset name (VIL-100, OpenLane-V)
    |   |   ├── P00             # preprocessing step 1
    |   |   |   ├── code
    |   |   ├── P01             # preprocessing step 2
    |   |   |   ├── code
    |   │   └── ...
    │   └── ...                 # etc.
    ├── Modeling                # directory for modeling
    │   ├── VIL-100             # dataset name (VIL-100, OpenLane-V)
    |   |   ├── ILD_seg         # a part of ILD for predicting lane probability maps
    |   |   |   ├── code
    |   |   ├── ILD_coeff       # a part of ILD for predicting lane coefficient maps
    |   |   |   ├── code
    |   |   ├── PLD             # PLD
    |   |   |   ├── code
    │   ├── OpenLane-V           
    |   |   ├── ...             # etc.
    ├── pretrained              # pretrained model parameters 
    │   ├── VIL-100              
    │   ├── OpenLane-V            
    │   └── ...                 # etc.
    ├── preprocessed            # preprocessed data
    │   ├── VIL-100             # dataset name (VIL-100, OpenLane-V)
    |   |   ├── P00             
    |   |   |   ├── output
    |   |   ├── P02             
    |   |   |   ├── output
    |   │   └── ...
    │   └── ...
    .
    .                           
    ├── OpenLane                # dataset directory
    │   ├── images              # Original images
    │   ├── lane3d_1000         # We do not use this directory
    │   ├── OpenLane-V
    |   |   ├── label           # lane labels formatted into pickle files
    |   |   ├── list            # training/test video datalists
    ├── VIL-100
    │   ├── JPEGImages          # Original images
    │   ├── Annotations         # We do not use this directory
    |   └── ...
    
### Evaluation (for VIL-100)
To test on VIL-100, you need to install official CULane evaluation tools. The official metric implementation is available [here](https://github.com/yujun0-0/MMA-Net/blob/main/INSTALL.md). Please downloads the tools into `ROOT/Modeling/VIL-100/MODEL_NAME/code/evaluation/culane/`. Then, you compile the evaluation tools. We recommend to see an [installation guideline](https://github.com/yujun0-0/MMA-Net/blob/main/INSTALL.md).
```
$ cd ROOT/Modeling/VIL-100/MODEL_NAME/code/evaluation/culane/
$ make
```

### Train
1. Set the dataset you want to train on (`DATASET_NAME`). Also, set the model (ILD or PLD) you want to train (`MODEL_NAME`).
2. Parse your dataset path into the `-dataset_dir` argument.
3. Edit `config.py` if you want to control the training process in detail
```
$ cd ROOT/Modeling/DATASET_NAME/MODEL_NAME/code/
$ python main.y --run_mode train --pre_dir ROOT/preprocessed/DATASET_NAME/ --dataset_dir /where/is/your/dataset/path 
```
 
### Test
1. Set the dataset you want to train on (`DATASET_NAME`). Also, set the model (ILD or PLD) you want to train (`MODEL_NAME`).
2. Parse your dataset path into the `-dataset_dir` argument.
3. If you want to get the performances of our work,
```
$ cd ROOT/Modeling/DATASET_NAME/MODEL_NAME/code/
$ python main.y --run_mode test_paper --pre_dir ROOT/preprocessed/DATASET_NAME/ --paper_weight_dir ROOT/pretrained/DATASET_NAME/ --dataset_dir /where/is/your/dataset/path
```
4. If you want to evaluate a model you trained,
```
$ cd ROOT/Modeling/DATASET_NAME/MODEL_NAME/code/
$ python main.y --run_mode test --pre_dir ROOT/preprocessed/DATASET_NAME/ --dataset_dir /where/is/your/dataset/path
```
5. (optional) If you set `disp_test_result=True` in code/options/config.py file, you can visualize the detection results.

### Preprocessing
You can obtain the preprocessed data, by running the codes in Preprocessing directories. Data preprocessing is divided into several steps. Below we describe each step in detail.
1. In P00, the type of ground-truth lanes in a dataset is converted to pickle format. (only for VIL-100)
2. In P01, each lane in a training set is represented by 2D points sampled uniformly in the vertical direction.
3. In P02, a lane matrix is constructed and SVD is performed. Then, each lane is transformed into its coefficient vector.
4. In P03, video-based datalists are generated for training and test sets.

```
$ cd ROOT/Preprocessing/DATASET_NAME/PXX_each_preprocessing_step/code/
$ python main.py --dataset_dir /where/is/your/dataset/path
```

### Reference
```
@Inproceedings{
    Jin2023rvld,
    title={Recursive Video Lane Detection},
    author={Jin, Dongkwon and Kim, Dahyun and Kim, Chang-Su},
    booktitle={ICCV},
    year={2023}
}
```
