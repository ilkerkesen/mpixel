import math

import torch
import torch.nn as nn


PATCH_EMBEDDING_INIT_METHODS = (
    "xavier_uniform",
    "xavier_normal",
    "kaiming_uniform",
    "kaiming_normal",
    "trunc_normal",
)


def _init_patch_conv_weight(weight, method):
    if method == "xavier_uniform":
        torch.nn.init.xavier_uniform_(weight.view([weight.shape[0], -1]))
    elif method == "xavier_normal":
        torch.nn.init.xavier_normal_(weight.view([weight.shape[0], -1]))
    elif method == "kaiming_uniform":
        torch.nn.init.kaiming_uniform_(weight, mode="fan_in", nonlinearity="relu")
    elif method == "kaiming_normal":
        torch.nn.init.kaiming_normal_(weight, mode="fan_in", nonlinearity="relu")
    elif method == "trunc_normal":
        torch.nn.init.trunc_normal_(weight, std=0.02)
    else:
        raise ValueError(
            f"Unknown patch_embedding_init method: {method!r}. "
            f"Expected one of {PATCH_EMBEDDING_INIT_METHODS}."
        )


class PatchEmbeddings(nn.Module):
    """
    Image to Patch Embedding.
    Based on timm implementation, which can be found here:
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    """

    def __init__(self, max_num_patches=529, patch_size=16, num_channels=3, embed_dim=768):
        super().__init__()
        self.max_num_patches = max_num_patches
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_channels = num_channels

        self.projection = nn.Conv2d(num_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values):
        batch_size, num_channels, height, width = pixel_values.shape
        x = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return x

    def initialize_weights(self, method=None):
        method = method or "xavier_uniform"
        _init_patch_conv_weight(self.projection.weight.data, method)
