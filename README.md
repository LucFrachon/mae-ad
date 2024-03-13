# Low-Rank Adapted Masked Autoencoder for Anomaly Detection [WIP]

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

This is still work in progress and I will update this repo as I make progress.

## Model development

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
masks. Another difference during inference is that the loss is calculated pixel-wise, then smoothed with a Gaussian
filter of radius 4. This reduces the impact of small random reconstruction errors.

The inference process requires a number of patches per side that is divisible by the number of masks, in this case 4 (
but other values could be experimented with). This mandates a patch size `p = img_size / patches_per_side`, which cannot
be equal to the original implementation where `p = 16` and `img_size = 224`. Therefore, while the rest of the encoder
starts with pretrained weights from ImageNet (provided by the MAE authors), the patch embedding layer is initialised
with random weights.

During development, I noticed that the reconstruction was worse in patches that contained curves or more complex shapes.
Therefore, I reduced the patch size further to `p = 7`, which means `patches_per_side = 32`. This improved
reconstruction accuracy around such areas of the image, which helps defects stand out more.

## Usage

### Installation

1. Clone the repo, create a virtual environment and install PyTorch:

    ```bash
    git clone https://github.com/LucFrachon/mae-ad.git
    cd mae-ad
    python3 -m venv .venv
    source .venv/bin/activate
    ```

then follow [instructions](https://pytorch.org/get-started/locally/).

2. Install the requirements:

    ```bash
    pip install -r requirements.txt
    ```

3. Test your installation:

    ```bash
    python3 -c "import torch; import timm; import loralib as lora; print(torch.__version__, timm.__version__, lora)"
    ```
   If you see the versions of the packages and the path to loralib, you're good to go.

### Data

The examples provided in this repo are based on
the [MVTec AD dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad) (non-commercial licence).
More specifically, I used the "capsule" subset. The training split contains only images of normal samples, while the
test split contains both normal and defective samples with 5 types of defects (crack, poke, print, scratch, squeeze).

The data loader expects a per-class directory structure. Since we're not interested in classifying the defects, we only
need 2 classes: good and bad. However, the file names within defect classes are the same, so you need to rename them
to avoid conflicts (e.g., `crack/000.png` --> `bad/crack_000.png`). **[TODO: Provide a script to do this]**

```  
<data_root>
├── train
│   ├── good
│   │   ├── 000.png
│   │   ├── 001.png
│   │   └── ...
└── test
    ├── good
    │   ├── 000.png
    │   ├── 001.png
    │   └── ...
    └── bad
        ├── crack_000.png
        ├── ...
        ├── poke_000.png
        ├── ...
        ├── print_000.png
        ├── ...
        ├── scratch_000.png
        ├── ...
        ├── squueze_000.png
        └── ...
```

### Training

To train the model, run `main_pretrain.py` with any relevant arguments. For example:

```bash
python main_train.py --epochs 4000 --model mae_vit_large_patch7 --freeze_non_lora --blr 1e-2 \
  --output_dir ../output_p7_ep4000 --log_dir ../output_p7_ep4000 --wandb_name p7_ep4000 \
  --pretrained ../checkpoints/mae_pretrain_vit_large.pth
```

Besides paths to the data and the outputs and logs, the default values should work in most cases. Here are the main ones
you might want to play with:

| Argument          | Description             | Type  | Default              | Remarks                                      |
|-------------------|-------------------------|-------|----------------------|----------------------------------------------|
| --model           | Model name              | str   | mae_vit_large_patch7 | Default model: 32x32 patches of size 7x7     |
| --epochs          | Number of epochs        | int   | 4000                 |                                              |
| --batch_size      | Batch size per GPU      | int   | 32                   | Adjust according to your GPU.                |
| --accum_iter      | Gradient accumulation   | int   | 1                    | Values > 1 increase the effective batch size |
| --freeze_non_lora | Freeze non-Lora weights | bool  | False                |                                              |
| --weight_decay    | Weight decay factor     | float | 0.001                |                                              |
| --blr             | Learning rate           | float | 1e-3                 | Effective LR is blr * eff. batch sz / 256    |
| --pretrained      | Pretrained weights      | str   | None                 | Path to a pretrained encoder                 |
| --resume          | Resume training         | str   | None                 | Path to a checkpoint (full autoencoder)      |
| --start_epoch     | Start epoch if resuming | int   | 0                    | Only required with `resume`                  |
| --output_dir      | Output directory        | str   | ../output_dir        |                                              |
| --log_dir         | Log directory           | str   | ../output_dir        |                                              |
| --wandb_name      | Wandb run name          | str   | None                 | If None, no Wandb logging                    |
| --num_workers     | Data loader workers     | int   | 4                    |                                              |

These models need to train for a relatively long time. The loss decreases slowly but steadily. Since the dataset is
small, many epochs are required. With 4000 epochs, training should take roughly half a day on a reasonably fast GPU.

### Inference

**[TODO: Write inference and evaluation code]** Until this is done, you can play with the `mae-ad_demo.ipynb` notebook.

## TODOs

- Provide a script to rename defect images and set up the expected folder structure.
- Write inference and evaluation code.
- Clean up the repo - remove unnecessary files inherited from the original MAE repo.

## References

```bibtex
@Article{MaskedAutoencoders2021,
  author  = {Kaiming He and Xinlei Chen and Saining Xie and Yanghao Li and Piotr Doll{\'a}r and Ross Girshick},
  journal = {arXiv:2111.06377},
  title   = {Masked Autoencoders Are Scalable Vision Learners},
  year    = {2021},
}
```

## Licence

<p>
<a property="dct:title" rel="cc:attributionURL" href="https://github.com/LucFrachon/mae-ad">
  Masked Autoencoder for Anomaly Detection
</a> 
  by 
<a rel="cc:attributionURL dct:creator" property="cc:attributionName" href="https://github.com/LucFrachon">
  Luc Frachon
</a> is licensed under 
<a href="http://creativecommons.org/licenses/by-nc/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">
  CC BY-NC 4.0
<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1">
<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1">
<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/nc.svg?ref=chooser-v1">
</a>
</p>
