# PET-Reconstruction
This is the implementation of the **Contrastive Diffusion Model with Auxiliary
Guidance for Coarse-to-Fine PET Reconstruction** **(early acccpeted by MICCAI 2023)**, 
which is the **first work** that applies Diffusion to Pet Reconstruction.

[//]: # (codebase: https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement)

## 1. Project Overview
<p align="center">
    <img src="assets/model.svg" width="550">

## 2. Environment Setup
The environment can be set up following the instructions below.

```
conda create --name diffpet python=3.8
source activate diffpet
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
git clone https://github.com/wyhlovecpp/PET-Reconstruction.git
cd PET-Reconstruction
pip install -r requirements.txt
```

## 3. Dataset
We conducted most of our low-dose brain PET
image reconstruction experiments on a public brain dataset, which is obtained
from the Ultra-low Dose PET Imaging Challenge 2022.
Out of the 206 18F-FDG brain PET subjects acquired using a Siemens Biograph Vision Quadra, 170
were utilized for training and 36 for evaluation. Each subject has a resolution of
128 × 128 × 128, and 2D slices along the z-coordinate were used for training and
evaluation.