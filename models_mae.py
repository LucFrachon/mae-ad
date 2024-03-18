# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
from timm.layers import PatchEmbed

from lora_layers import LoraBlock
from util import misc
from util.pos_embed import get_2d_sincos_pos_embed


class MaskedAutoencoderViT(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone modified to perform
    alternating deterministic masking for anomaly detection."""

    def __init__(
        self,
        img_size=224,
        patch_size=14,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
        inference_mask_ratio=0.25,
        train_mask_ratio=0.75,
        lora_rank=8,
    ):
        super().__init__()

        # --------------------------------------------------------------------------
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_per_side = img_size // patch_size
        self.inference_mask_ratio = inference_mask_ratio
        self.train_mask_ratio = train_mask_ratio
        # used for inference:
        self.masks_per_img = max(
            int(1 / self.inference_mask_ratio), int(1 / (1 - self.inference_mask_ratio))
        )  # M
        assert self.patches_per_side % self.masks_per_img == 0

        # Since the masking is deterministic during inference, we compute the mask
        # once and store it.
        self.mask, self.ids_keep, self.ids_restore = self.get_mask(
            self.patches_per_side
        )

        self.blocks = nn.ModuleList(
            [
                LoraBlock(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    qk_norm=None,
                    norm_layer=norm_layer,
                    lora_rank=lora_rank,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False
        )

        self.decoder_blocks = nn.ModuleList(
            [
                LoraBlock(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    qk_norm=None,
                    norm_layer=norm_layer,
                    lora_rank=lora_rank,
                )
                for i in range(decoder_depth)
            ]
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size ** 2 * in_chans, bias=True
        )  # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss
        # self.register_buffer(
        #     "loss_mean", torch.zeros(
        #         (num_patches,),
        #         requires_grad=False
        #     ).type_as(self.mask_token)
        # )
        # self.register_buffer(
        #     "loss_sqrd_mean", torch.zeros(
        #         (num_patches,),
        #         requires_grad=False
        #     ).type_as(self.mask_token)
        # )
        self.loss_map_smoother = misc.make_gaussian_kernel(11, 5).to(self.device)
        self.initialize_weights()

    def update_loss_statistics(self, loss):
        loss_ = loss.detach().mean(dim=0)
        sqrd_loss_ = torch.pow(loss_, 2).mean(dim=0)
        self.loss_mean = self.loss_mean * 0.99 + loss_ * 0.01
        self.loss_sqrd_mean = self.loss_sqrd_mean * 0.99 + sqrd_loss_ * 0.01

    def initialize_weights(self):
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches ** 0.5),
            cls_token=True,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.patch_embed.num_patches ** 0.5),
            cls_token=True,
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02)
        # as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    @torch.no_grad()
    def get_mask(self, patches_per_side):
        len_keep = int(patches_per_side * (1 - self.inference_mask_ratio))

        mask = torch.diagflat(torch.ones(self.masks_per_img, device=self.device))
        mask = mask.repeat(1, patches_per_side ** 2 // self.masks_per_img)

        ids_mask = torch.argsort(mask, dim=1, stable=True)
        ids_restore = torch.argsort(ids_mask, dim=1, stable=True)
        ids_keep = ids_mask[..., :len_keep]

        return mask, ids_keep, ids_restore

    def alternate_masking(self, x, i):
        N, L, D = x.shape
        ids_keep = self.ids_keep.clone().expand(N, -1, -1)
        mask = self.mask.clone().expand(N, -1, -1)
        x_masked = torch.gather(
            x, dim=1, index=ids_keep[:, i, :].unsqueeze(-1).repeat(1, 1, D)
        )
        return x_masked, mask[:, i, :]

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, inference=False):
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_token_ = cls_token.clone()
        cls_token = cls_token_.expand(x.shape[0], -1, -1)
        xs_list, masks_list = [], []
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]

        if inference:
            masks_per_img = self.masks_per_img
        else:
            masks_per_img = 1

        for i in range(masks_per_img):
            xi = x.clone()  # N, C, H, W
            if inference:
                xi, mask_i = self.alternate_masking(xi, i)
                ids_restore = self.ids_restore
            else:
                xi, mask_i, ids_restore = self.random_masking(xi, self.train_mask_ratio)

            xi = torch.cat((cls_token, xi), dim=1)
            # apply Transformer blocks
            for blk in self.blocks:
                xi = blk(xi)
            xi = self.norm(xi)

            xs_list.append(xi)
            masks_list.append(mask_i)

        x = torch.stack(xs_list, dim=1).to(self.device)  # (N, M, L, p * p * 3)
        mask = torch.stack(masks_list, dim=1).to(self.device)  # (N, M, L)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore, inference=False):
        x_out = []

        # add batch dimension to ids_restores:
        ids_restore = ids_restore.repeat(x.shape[0], 1, 1)
        if inference:
            masks_per_img = self.masks_per_img
        else:
            masks_per_img = 1

        for i in range(masks_per_img):
            xi = self.decoder_embed(x[:, i, ...])

            # append mask tokens to sequence
            ids_restore_i = ids_restore[:, i, :]
            mask_tokens = self.mask_token.repeat(
                xi.shape[0], ids_restore_i.shape[1] + 1 - xi.shape[1], 1
            )

            # raise Exception
            xi_ = torch.cat([xi[:, 1:, :], mask_tokens], dim=1)  # no cls token
            xi_ = torch.gather(
                xi_,
                dim=1,
                index=ids_restore_i.unsqueeze(-1).repeat(1, 1, xi.shape[2]),
            )  # unshuffle
            xi = torch.cat([xi[:, :1, :], xi_], dim=1)  # append cls token

            # add positional embedding
            xi = xi + self.decoder_pos_embed

            # apply Transformer blocks
            for blk in self.decoder_blocks:
                xi = blk(xi)
            xi = self.decoder_norm(xi)

            # predictor projection
            xi = self.decoder_pred(xi)

            # remove cls token
            xi = xi[:, 1:, :]
            x_out.append(xi)

        pred = torch.stack(x_out, dim=1).to(self.device)

        return pred

    def forward_loss(self, imgs, preds, masks, inference=False):
        """
        imgs: [N, 3, H, W]
        preds: [N, M, L, p*p*3]
        masks: [N, M, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)  # (N, L, p * p * 3)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        n_masks = self.masks_per_img if inference else 1
        target_ = target.unsqueeze(1).repeat(1, n_masks, 1, 1)

        # [N, M, L, 3p^2], mean loss per image, patch, pixel & mask
        loss = ((preds - target_) ** 2)

        if inference:
            # if inference, retain loss for individual pixels
            # [N, L, 3p^2], mean over masks

            loss = (loss * masks.unsqueeze(-1)).sum(1) / (masks.unsqueeze(-1).sum(1))
            return loss

        # [N, M, L], mean per patch & mask:
        loss = ((preds - target_) ** 2).mean(dim=-1)
        # [N, L], mean per image, removed patches only:
        loss = (loss * masks).sum(dim=(1, 2)) / masks.sum(dim=(1, 2))
        # if training, compute mean loss over masks and patches to get a scalar
        # mean on removed patches, per mask:
        # self.update_loss_statistics(loss)  # update loss stats for no-defect images

        # mean loss, overall:
        return loss.mean()

    def forward(self, imgs, mask_ratio=0.75):
        """Used during training"""
        latents, masks, ids_restore = self.forward_encoder(
            imgs, inference=False
        )
        preds = self.forward_decoder(latents, ids_restore)
        loss = self.forward_loss(imgs, preds, masks, inference=False)
        return loss, preds, masks

    @torch.no_grad()
    def inference(self, imgs, threshold=0.5, pixel_map=False):
        self.eval()

        latents, masks, ids_restore = self.forward_encoder(imgs, inference=True)
        preds = self.forward_decoder(latents, ids_restore)

        # # Compute pixel norm stats
        if self.norm_pix_loss:
            preds_unnormed = preds.clone()
            target = self.patchify(imgs)  # (N, L, p * p * 3)
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            preds = (preds + mean) * ((var + 1.0e-6) ** 0.5)
            del target

        loss = self.forward_loss(imgs, preds, masks, inference=True)

        preds = preds * masks.unsqueeze(-1)
        # the predictions do not overlap with inference mask ratio = 0.25:
        preds = preds.sum(dim=1)
        preds = self.unpatchify(preds)

        if pixel_map:
            loss_maps = self.unpatchify(loss).mean(dim=1)  # mean over 3 channels
            loss_maps = 2 * (
                    torch.sigmoid(
                        torch.nn.functional.conv2d(
                            loss_maps, self.loss_map_smoother, bias=None, stride=1,
                            padding=1
                        )
                    ) - 0.5
            )
        else:  # patch map
            # loss is [N, L, 3p^2], compute mean per patch and extend size:
            loss_maps = loss.mean(dim=-1, keepdim=True).repeat(
                1, 1, 3 * self.patch_size ** 2
            )
            # mean over 3 channels:
            loss_maps = 2 * (torch.sigmoid(self.unpatchify(loss_maps).mean(1)) - 0.5)

        ano_scores = loss_maps.max().detach().item()
        decisions = (ano_scores > threshold)
        return_dict = {
            "images": imgs.detach().cpu(),
            "preds": preds.detach().cpu(),
            "loss_maps": loss_maps.detach().cpu(),
            "anomaly_scores": ano_scores,
            "decisions": decisions
        }
        
        if self.norm_pix_loss:
            preds_unnormed = preds_unnormed * masks.unsqueeze(-1)
            # the predictions do not overlap with inference mask ratio = 0.25:
            preds_unnormed = preds_unnormed.sum(dim=1)
            preds_unnormed = self.unpatchify(preds_unnormed)
            return_dict["preds_unnormed"] = preds_unnormed.detach().cpu()

        return return_dict


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def mae_vit_large_patch14(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def mae_vit_huge_patch14(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def mae_vit_large_patch7(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=7,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
