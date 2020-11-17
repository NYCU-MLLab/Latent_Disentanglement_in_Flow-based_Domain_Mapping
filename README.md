# Introduction
This is the source code for the Master thesis **Latent Disentanglement in Flow-based Domain Mapping** from National Chiao Tung University, Taiwan. There have two models in this research. Attribute Decomposition Flow (AD-flow) and Mutual Information Flow (MI-flow). 

## Attribute Decomposition Flow

Attribute decomposition flow used feature encoders to disentangle the latent vectors derived from the Glow model into attribute-relevant and attribute-irrelevant latent vectors. By using the $\textit{structural-perceptual}$ loss and $\textit{cycle consistency}$ loss, we could guide the feature encoders to extract the attribute-irrelevant latent vectors according to the image structural information. Since there is no paired data for domain mapping, we proposed the $\textit{sequential random-pair reconstruction}$ loss to create a self-supervised learning which could fully maintain the structural information and learn the attribute and the additional the $\textit{sequential classification}$ loss provided the attribute information from the sequence data.
![](https://i.imgur.com/6sOP42h.png)

## Mutual Information Flow

Mutual information flow introduced the information theory to the flow-based generative model formed an end-to-end training. We provided the relationship between the mutual information and latent disentanglement. Condition-relevant and condition-irrelevant mutual information help to strengthen the correlation between the separating latent variable and specific attribute. We also proved that minimizing the mutual information between observed inputs and latent outputs is equivalent to train the flow model to a certain extent. 

<img src="https://i.imgur.com/Ko3Ij6b.png" width="400">

# Getting start
## Environment
The developed environment is listed in below
* OS : Ubuntu 16.04
* CUDA : 10.0
* Nvidia Driver : 410.78
* Python 3.6
* Pytorch 1.2.0

The related python packages are listed in `requirements.txt`.

## Preprocess
### Dataset

The dataset in the research we used is [Lip Reading in the Wild (LRW)](http://www.robots.ox.ac.uk/~vgg/data/lip_reading/). We only used the images for the training. For the dataset preprocessing, please reference this [repo](https://github.com/voletiv/lipreading-in-the-wild-experiments).

The dataset file directory format must be in the following way:
```
train
  ├── ABOUT
  |     ├── ABOUT_00001_01.jpg
  |     ├── ABOUT_00001_02.jpg
  |     |        ⋮
  |     ├── ABOUT_01000_18.jpg
  |     └── ABOUT_01000_19.jpg
  ├── ABSOLUTELY
  |     ⋮
  └── YOUNG      
```
It need to separate different classes in different folder. Each folder has all the frame images with the formate of 

> [class]\_[ID]\_[frame_ID].jpg


### Query image
The input size of the query image need to be 256x256 RGB image. The script `preprocess.py` is used to preprocess your own original query image. Note that the original image should be clear and high resolution.
```bash
$ python preprocess.py [original image dir]
```
If your original image file name is `ABC.jpg`, the output image would be named as `ABC_256.jpg` and save in the directory `./image/ref_img/ABC_256.jpg`

# Training
Both model AD-flow and MI-flow could use the same method for training and testing. Before training the model, you should change the current directory to the corresponding model.
* The source code of **Attribute Decomposition Flow** is in the folder `./ad-flow/`.
```
$ cd ad-flow
```
* The source code of **Mutual Information Flow** is in the folder `./mi-flow/`.
```
$ cd mi-flow
```


The script `train.py` in each folder is used for training. The training dataset directory and the image file formate need to follow the previous showed.

```bash
$ python train.py -n <name>  [dataset directory]
```
The argument `-n` control the saving folder name for the training result. The result would be saved in `./result/sample_<name>`

# Testing
## Pretrain model
* Download the pretrained model [checkpoint](https://drive.google.com/drive/folders/1by8REbZTezl9nxsqYolNgzL7sjGoqXWM?usp=sharing).
* Place all the checkpoint files in the `ad-flow` into the `./ad-flow/checkpoint/`
* Place all the checkpoint files in the `mi-flow` into the `./mi-flow/checkpoint/`
```
ad-flow
   ├── checkpoint
   |     ├── ad-flow_total_checkpoint.tar
   |     ├── glow_checkpoint.pt
   |     └── shape_predictor_68_face_landmarks.dat
mi-flow
   └── checkpoint
         └── mi-flow_total_checkpoint.pt 

```

The script `test.py` is used for generate the sample result. 

The source image is in the directory of `./image/ref_img/` which contains two choices of input souce image [`trump` / `ellen`]. 
The source image would be choosen `trump` as default. The argument `--soucre` controls the source image. 

The directory of the query image should place in `./image/ref_img/[query_image]_256.jpg`
```bash
$ python test.py --source ellen [query_image]
```
The output would be saved in the directory `./result/target_img/ellen/[query_image]/`

# Results
Here has the [demo](https://www.youtube.com/watch?v=SuxqpKWr6BQ&feature=youtu.be) video for both AD-flow and MI-flow in the task of talking face generation.

# References
The source code of the [Glow](https://arxiv.org/abs/1807.03039) model in this research reference from the PyTorch version with this [repo](https://github.com/rosinality/glow-pytorch).