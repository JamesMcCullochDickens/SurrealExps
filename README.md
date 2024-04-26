# SurrealExps

This repository contains PyTorch code to train and evaluate the built-in TorchVision models for segmentation on the task of human parsing/part segmentation
on the Surreal dataset, for the models LR_ASPP, DeepLabv3 with Resnet50 and Resnet101 backbones. 

# Installation
Run install.sh, tested on PyTorch version 2.2.2, torchvision version 0.17.2, Cuda 12.1, and python 3.10. Further ultralytics is only
necessary for in the wild inference.

# Data Pre-Processing
The data can be downloaded here: https://www.di.ens.fr/willow/research/surreal/
Write the path of the original data in the first line of Dataset_Location.txt.
The second line of that file should be where you want to save individual frames
for training/validation/testing. 

I do not use the entire dataset, but roughly about 1/15 of the total number of frames.
Rather than decode videos during training/inference you can write your own vid_frame pairs using get_vid_frame_pairs(...) in data_preprocess.py, 
or just use the included video frame pairs. Then you should run write_surreal_dataset_frames() in Dataloaders/data_preprocess.py to write
the frames needed for training.


# Training and Inference:
Run train_rgb.sh, adjusting gpus used with the --gpu_override flag. I use 2 gpus with 24 GB of VRAM, and a batch size of 128. For single
GPU training, I recommend a batch size of 64 images and 16-bit mixed precision training.  

# Eval Results




# Citations
Please cite the Surreal dataset if you end up using any of this in a paper. 


@INPROCEEDINGS{varol17_surreal,
  title     = {Learning from Synthetic Humans},
  author    = {Varol, G{\"u}l and Romero, Javier and Martin, Xavier and Mahmood, Naureen and Black, Michael J. and Laptev, Ivan and Schmid, Cordelia},
  booktitle = {CVPR},
  year      = {2017}
}

