"""Gemma 4 encoder-free 思路的图像机制实验：RGB patch → 线性投影 → LLM。

只处理固定大小图像，没有图像预训练、视频或音频能力。可用合成小图
练习视觉前缀、二维坐标和 label mask，不需要下载大权重。
"""

import torch
from torch import nn
from torch.nn import functional as F


class PatchProjector(nn.Module):
    def __init__(self, dim, image_size=8, patch_size=4):
        super().__init__()
        if image_size < 1 or patch_size < 1 or image_size % patch_size:
            raise ValueError("image_size 必须被 patch_size 整除")
        self.image_size, self.patch_size = image_size, patch_size
        self.grid = image_size // patch_size
        self.projection = nn.Linear(3 * patch_size**2, dim)
        self.rows = nn.Embedding(self.grid, dim)
        self.columns = nn.Embedding(self.grid, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, images):
        if images.ndim != 4 or images.shape[1:] != (3, self.image_size, self.image_size):
            raise ValueError("图像必须为 [B,3,image_size,image_size]")
        patches = F.unfold(images, self.patch_size, stride=self.patch_size).transpose(1, 2)
        position = torch.arange(self.grid**2, device=images.device)
        coordinates = self.rows(position // self.grid) + self.columns(position % self.grid)
        return self.norm(self.projection(patches) + coordinates)


class VisionPrefixModel(nn.Module):
    def __init__(self, language_model, image_size=8, patch_size=4):
        super().__init__()
        self.language_model = language_model
        self.projector = PatchProjector(language_model.config.dim, image_size, patch_size)

    def forward(self, images, input_ids, labels=None, attention_mask=None):
        visual = self.projector(images)
        batch, patches, _ = visual.shape
        text = self.language_model.token_embedding(input_ids)
        embeddings = torch.cat((visual, text), 1)
        dummy_ids = torch.cat((input_ids.new_zeros(batch, patches), input_ids), 1)
        targets = (
            None
            if labels is None
            else torch.cat((labels.new_full((batch, patches), -100), labels), 1)
        )
        mask = (
            torch.ones_like(input_ids, dtype=torch.bool)
            if attention_mask is None
            else attention_mask
        )
        mask = torch.cat(
            (torch.ones(batch, patches, dtype=torch.bool, device=mask.device), mask), 1
        )
        return self.language_model(
            dummy_ids, labels=targets, attention_mask=mask, inputs_embeds=embeddings
        )
