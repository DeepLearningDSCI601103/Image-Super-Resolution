# Single-Image Super-Resolution with Deep Learning

Enhance the resolution and visual quality of low-resolution images using a deep learning-based Single-Image Super-Resolution system. This project leverages Convolutional Neural Networks (CNNs) to intelligently upscale images, providing a high-resolution output that preserves fine details and improves overall image quality.

## Table of Contents
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [File Structure](#FileStructure)
- [Output](#Output)

## Getting Started
Data sets and softwares to be installed are provided below to set up and deploy the deep learning-based Super-Resolution system on your local machine.

### Prerequisites
Download datasets from below links:  
DIV2K  -- https://www.kaggle.com/datasets/joe1995/div2k-dataset.  

Urban100 --https://www.kaggle.com/datasets/jesucristo/super-resolution-benchmarks. 

We have Utilized DIV2k for training the models and Urban100 for testing.  

### Installation
You will need the following to run the above:

Python  


tensorflow   


keras   


numpy   


tqdm  


matplotlib, skimage, scipy

## File Structure
ISR_VDSR.py : Contains preprocessing of the data and models with basic ISR with CNN and VDSR.  

ISR_EDSR.py : Contains EDSR model.  

## Output
Below are few results
![image](https://github.com/DeepLearningDSCI601103/Image-Super-Resolution/assets/128659646/626a0a0a-a756-4976-8dd2-ff71e4f4deff)
![image](https://github.com/DeepLearningDSCI601103/Image-Super-Resolution/assets/128659646/dff17bb8-082e-473b-a697-8ad91f46576a)


