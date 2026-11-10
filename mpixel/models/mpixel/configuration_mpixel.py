# coding=utf-8
# Copyright 2022 Facebook AI and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
MPIXEL model configuration
adapted from ViT MAE model configuration:
https://github.com/huggingface/transformers/blob/v4.17.0/src/transformers/models/vit_mae/configuration_vit_mae.py
"""

from transformers import PretrainedConfig


class MPIXELConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MPIXELModel`]. It is used to instantiate a
    MPIXEL model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a base-size encoder (hidden size 768, 12 layers, 12 attention heads).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        max_num_patches (`int`, *optional*, defaults to 529):
            The maximum number of patches in the input sequence.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the queries, keys and values.
        decoder_num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the decoder.
        decoder_hidden_size (`int`, *optional*, defaults to 512):
            Dimensionality of the decoder.
        decoder_num_hidden_layers (`int`, *optional*, defaults to 8):
            Number of hidden layers in the decoder.
        decoder_intermediate_size (`int`, *optional*, defaults to 2048):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the decoder.
        mask_ratio (`float`, *optional*, defaults to 0.25):
            The ratio of the number of masked tokens in the input sequence.
        norm_pix_loss (`bool`, *optional*, defaults to `True`):
            Whether or not to train with normalized pixels (see Table 3 in the paper).
        temperature (`float`, *optional*, defaults to 1.0):
            Temperature scaling factor for the contrastive loss in headless pretraining.
        patch_embedding_init (`str`, *optional*, defaults to `None`):
            Initialization algorithm for the patch embedding conv weights. One of
            `"xavier_uniform"`, `"xavier_normal"`, `"kaiming_uniform"`, `"kaiming_normal"`,
            `"trunc_normal"`. If `None`, the patch embedding class default is used.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            Base wavelength for rotary position embeddings. Shared between encoder
            and decoder.
        pooling_mode (`str`, *optional*, defaults to `"mean"`):
            Pooling of the patch embeddings in [`MPIXELForSequenceClassification`]. One of `"mean"`, `"max"`,
            `"cls"` or `"pma<num_heads>"` (for example `"pma8"`).
        pooler_add_layer_norm (`bool`, *optional*, defaults to `True`):
            Whether the pooler of [`MPIXELForSequenceClassification`] applies a layer norm before pooling.
        head_init (`str`, *optional*, defaults to `"pytorch"`):
            Initialization of a new pooler and classifier in [`MPIXELForSequenceClassification`]. `"pytorch"` uses
            the PyTorch defaults (uniform weights and biases), the same as the Lightning fine-tuning runs. `"normal"`
            uses the HF default: weights from a normal distribution with std `initializer_range`, and zero biases.

    Example:

    ```python
    >>> from mpixel.models.mpixel.configuration_mpixel import MPIXELConfig
    >>> from mpixel.models.mpixel.modeling_mpixel import MPIXELModel

    >>> # Initializing a base-size configuration
    >>> configuration = MPIXELConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = MPIXELModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "mpixel"

    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        is_encoder_decoder=False,
        max_num_patches=529,
        patch_size=16,
        num_channels=3,
        qkv_bias=True,
        decoder_num_attention_heads=16,
        decoder_hidden_size=512,
        decoder_num_hidden_layers=8,
        decoder_intermediate_size=2048,
        mask_ratio=0.0,
        norm_pix_loss=True,
        temperature=1.0,
        patch_embedding_init=None,
        rope_theta=10000.0,
        reference_compile=None,
        pooling_mode="mean",
        pooler_add_layer_norm=True,
        head_init="pytorch",
        **kwargs
    ):
        super().__init__(**kwargs)
        if head_init not in ("pytorch", "normal"):
            raise ValueError(f'head_init must be "pytorch" or "normal", got {head_init!r}')

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.max_num_patches = max_num_patches
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.qkv_bias = qkv_bias
        self.decoder_num_attention_heads = decoder_num_attention_heads
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_num_hidden_layers = decoder_num_hidden_layers
        self.decoder_intermediate_size = decoder_intermediate_size
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss
        self.temperature = temperature
        self.patch_embedding_init = patch_embedding_init
        self.rope_theta = rope_theta
        self.reference_compile = reference_compile
        self.pooling_mode = pooling_mode
        self.pooler_add_layer_norm = pooler_add_layer_norm
        self.head_init = head_init
