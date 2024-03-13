# Low-Rank Adapted Masked Autoencoder for Anomaly Detection

## Introduction

This repo is a personal project that I've been thinking about for a while. In anomaly detection,
a popular method is to use generative models to model the normal data distribution. In particular,
autoencoders can be used to learn compact latent representations of normal samples. An abnormal sample
would cause a large reconstruction error, which can serve as an anomaly flag.

This model takes the same idea but uses a Visual Transformer (ViT) trained with the Masked Autoencoder (MAE) method.
In MAE, random patches of the image are masked during training and the model learns to reconstruct them from context.
Presented with images of defective objects, some patches will incur a large reconstruciton error, which
indicates the presence of a potential anomaly.

One issue with this approach is that anomaly detection datasets are often small, while ViT models require large
amounts of data to train. To address this, I propose to use a low-rank approximation of the ViT model, whereby the
linear layers in the attention blocks are replaced by [LoRA](https://github.com/microsoft/LoRA/tree/main/loralib)
layers. This reduces the number of parameters in the model, making it easier to train with small datasets.

## Model

The code here is based on the [official PyTorch implementation of MAE](https://github.com/facebookresearch/mae) with
various adaptations made for anomaly detection and to implement LoRA.
In particular, patch masking behaves differently during training and inference.

During training, masking is performed in the same way as in the original MAE, with a high masking ratio (75% of the
patches are masked). Moreover, I have reduced regularisation significantly, removing any data augmentation and lowering
weight decay. This is because anomaly detection assumes consistency in the data samples and their presentation,
therefore
we only need to learn from a relatively constrained data distribution.

During inference, masking is deterministic. Each image receives 4 different masks, each removing 25% of the image.
The model reconstructs each masked image and the reconstruction error of the masked patches is averaged across the 4
masks.

## Usage

### Installation

Clone the repo, create a virtual environment and install PyTorch according
to [instructions](https://pytorch.org/get-started/locally/).
