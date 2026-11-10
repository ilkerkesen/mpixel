"""
PyTorch MPIXEL models
"""

import collections.abc
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
from torch import nn

from transformers.activations import ACT2FN
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb
from transformers.processing_utils import Unpack
from transformers.pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import ModelOutput, TransformersKwargs, logging, torch_int
from transformers.utils.generic import can_return_tuple, check_model_inputs

from ..pooling import PMA, PoolingForSequenceClassificationHead, PoolingMode
from .embeddings import PIXELEmbeddingsRoPE
from .configuration_mpixel import MPIXELConfig


logger = logging.get_logger(__name__)


@dataclass
class MPIXELModelOutput(ModelOutput):
    r"""
    mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
        Tensor indicating which patches are masked (1) and which are not (0).
    ids_restore (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
        Tensor containing the original index of the (shuffled) masked patches.
    """

    last_hidden_state: Optional[torch.FloatTensor] = None
    mask: Optional[torch.LongTensor] = None
    ids_restore: Optional[torch.LongTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None


@dataclass
class MPIXELDecoderOutput(ModelOutput):
    r"""
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, patch_size ** 2 * num_channels)`):
        Pixel reconstruction logits.
    """

    logits: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None


@dataclass
class MPIXELForPreTrainingOutput(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`):
        Pixel reconstruction loss.
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, patch_size ** 2 * num_channels)`):
        Pixel reconstruction logits.
    mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
        Tensor indicating which patches are masked (1) and which are not (0).
    ids_restore (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
        Tensor containing the original index of the (shuffled) masked patches.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    mask: Optional[torch.LongTensor] = None
    ids_restore: Optional[torch.LongTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None


# Copied from transformers.models.vit.modeling_vit.eager_attention_forward
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    # Take the dot product between "query" and "key" to get the raw attention scores.
    attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scaling

    # Apply the additive attention mask (0 for kept keys, dtype min for padding).
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    # Normalize the attention scores to probabilities.
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


# Copied from transformers.models.vit.modeling_vit.ViTSelfAttention ViT->MPIXEL
class MPIXELSelfAttention(nn.Module):
    def __init__(self, config: MPIXELConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.dropout_prob = config.attention_probs_dropout_prob
        self.scaling = self.attention_head_size**-0.5
        self.is_causal = False

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = hidden_states.shape[0]
        new_shape = batch_size, -1, self.num_attention_heads, self.attention_head_size

        key_layer = self.key(hidden_states).view(*new_shape).transpose(1, 2)
        value_layer = self.value(hidden_states).view(*new_shape).transpose(1, 2)
        query_layer = self.query(hidden_states).view(*new_shape).transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_layer, key_layer = apply_rotary_pos_emb(query_layer, key_layer, cos, sin)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        context_layer, attention_probs = attention_interface(
            self,
            query_layer,
            key_layer,
            value_layer,
            attention_mask,
            is_causal=self.is_causal,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.dropout_prob,
        )

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.reshape(new_context_layer_shape)

        return context_layer, attention_probs


# Copied from transformers.models.vit.modeling_vit.ViTSelfOutput with ViT->MPIXEL
class MPIXELSelfOutput(nn.Module):
    """
    The residual connection is defined in MPIXELLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: MPIXELConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


# Copied from transformers.models.vit.modeling_vit.ViTAttention with ViT->MPIXEL
class MPIXELAttention(nn.Module):
    def __init__(self, config: MPIXELConfig):
        super().__init__()
        self.attention = MPIXELSelfAttention(config)
        self.output = MPIXELSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads: set[int]):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        self_attn_output, _ = self.attention(hidden_states, attention_mask, head_mask, position_embeddings)
        output = self.output(self_attn_output, hidden_states)
        return output


# Copied from transformers.models.vit.modeling_vit.ViTIntermediate ViT->MPIXEL
class MPIXELIntermediate(nn.Module):
    def __init__(self, config: MPIXELConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.vit.modeling_vit.ViTOutput ViT->MPIXEL
class MPIXELOutput(nn.Module):
    def __init__(self, config: MPIXELConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        return hidden_states


# Copied from transformers.models.vit.modeling_vit.ViTLayer with ViT->MPIXEL,VIT->VITMAE
class MPIXELLayer(GradientCheckpointingLayer):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config: MPIXELConfig):
        super().__init__()
        self.config = config
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = MPIXELAttention(config)
        self.intermediate = MPIXELIntermediate(config)
        self.output = MPIXELOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    @torch.compile(dynamic=True)
    def compiled_mlp(self, hidden_states: torch.Tensor) -> torch.Tensor:
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)
        return self.output(layer_output, hidden_states)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        hidden_states_norm = self.layernorm_before(hidden_states)
        attention_output = self.attention(hidden_states_norm, attention_mask, head_mask, position_embeddings)

        # first residual connection
        hidden_states = attention_output + hidden_states

        # in MPIXEL, layernorm is also applied after self-attention
        if self.config.reference_compile:
            return self.compiled_mlp(hidden_states)

        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)
        return self.output(layer_output, hidden_states)


# Copied from transformers.models.vit.modeling_vit.ViTEncoder with ViT->MPIXEL
class MPIXELEncoder(nn.Module):
    def __init__(self, config: MPIXELConfig):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([MPIXELLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> BaseModelOutput:
        for i, layer_module in enumerate(self.layer):
            layer_head_mask = head_mask[i] if head_mask is not None else None
            hidden_states = layer_module(hidden_states, attention_mask, layer_head_mask, position_embeddings)

        return BaseModelOutput(last_hidden_state=hidden_states)


class MPIXELPreTrainedModel(PreTrainedModel):
    config: MPIXELConfig
    base_model_prefix = "vit"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_flex_attn = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": MPIXELLayer,
        "attentions": MPIXELSelfAttention,
    }

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, PIXELEmbeddingsRoPE):
            module.initialize_weights()
        elif isinstance(module, MPIXELDecoder):
            module.mask_token.data.zero_()

    def _maybe_set_compile(self):
        # Default to False; set to True in pretraining when flash_attn is active,
        # since full torch.compile is incompatible with flash_attn's einops-based unpad path.
        if self.config.reference_compile is None:
            self.config.reference_compile = False


class MPIXELModel(MPIXELPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = PIXELEmbeddingsRoPE(config)
        self.encoder = MPIXELEncoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        rotary_config = deepcopy(config)
        rotary_config.max_position_embeddings = config.max_num_patches + 1
        self.rotary_emb = LlamaRotaryEmbedding(rotary_config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @torch.compile(dynamic=True)
    def _compiled_embeddings(self, pixel_values, attention_mask, patch_mask, interpolate_pos_encoding):
        return self.embeddings(pixel_values, attention_mask=attention_mask, patch_mask=patch_mask, interpolate_pos_encoding=interpolate_pos_encoding)

    @check_model_inputs(tie_last_hidden_states=False)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        patch_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        interpolate_pos_encoding: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MPIXELModelOutput:
        r"""
        attention_mask (`torch.FloatTensor` of shape `(batch_size, num_patches)`, *optional*):
            Mask indicating which patch tokens are valid (1) vs. padding (0). Defaults to all ones.
        patch_mask (`torch.FloatTensor` of shape `(batch_size, num_patches)`, *optional*):
            Mask driving controlled (span) masking in the embeddings. If `None`, the embeddings apply no masking.
        interpolate_pos_encoding (`bool`, *optional*, default `False`):
            Whether to interpolate the pre-trained position encodings. This is mainly used to use the model on higher
            resolution images.

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, MPIXELModel
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
        >>> model = MPIXELModel.from_pretrained("facebook/vit-mae-base")

        >>> inputs = image_processor(images=image, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        pixel_values = pixel_values.to(dtype=self.dtype)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        self._maybe_set_compile()

        if attention_mask is None:
            attention_mask = torch.ones(
                (pixel_values.shape[0], self.embeddings.max_num_patches), device=self.device
            )

        # flash_attn's unpad/pad uses einops with SymInt — incompatible with full torch.compile.
        # When reference_compile=True, compile only the embedding step (attention-free).
        if self.config.reference_compile:
            embedding_output, attention_mask, mask, ids_restore, position_ids = self._compiled_embeddings(
                pixel_values, attention_mask, patch_mask, interpolate_pos_encoding
            )
        else:
            embedding_output, attention_mask, mask, ids_restore, position_ids = self.embeddings(
                pixel_values,
                attention_mask=attention_mask,
                patch_mask=patch_mask,
                interpolate_pos_encoding=interpolate_pos_encoding,
            )

        if "flash" in self.config._attn_implementation:
            encoder_attention_mask = attention_mask
        else:
            encoder_attention_mask = self.get_extended_attention_mask(
                attention_mask, embedding_output.shape, self.device
            ).to(self.dtype)

        position_embeddings = self.rotary_emb(embedding_output, position_ids)

        encoder_outputs: BaseModelOutput = self.encoder(
            embedding_output,
            attention_mask=encoder_attention_mask,
            head_mask=head_mask,
            position_embeddings=position_embeddings,
        )
        sequence_output = encoder_outputs.last_hidden_state
        sequence_output = self.layernorm(sequence_output)

        return MPIXELModelOutput(last_hidden_state=sequence_output, mask=mask, ids_restore=ids_restore)


class MPIXELDecoder(nn.Module):
    def __init__(self, config: MPIXELConfig, num_patches: int):
        super().__init__()
        self.decoder_embed = nn.Linear(config.hidden_size, config.decoder_hidden_size, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.decoder_hidden_size))

        decoder_config = deepcopy(config)
        decoder_config.hidden_size = config.decoder_hidden_size
        decoder_config.num_hidden_layers = config.decoder_num_hidden_layers
        decoder_config.num_attention_heads = config.decoder_num_attention_heads
        decoder_config.intermediate_size = config.decoder_intermediate_size
        decoder_config.max_position_embeddings = num_patches + 1
        self.decoder_layers = nn.ModuleList(
            [MPIXELLayer(decoder_config) for _ in range(config.decoder_num_hidden_layers)]
        )

        self.decoder_norm = nn.LayerNorm(config.decoder_hidden_size, eps=config.layer_norm_eps)
        self.decoder_pred = nn.Linear(
            config.decoder_hidden_size, config.patch_size**2 * config.num_channels, bias=True
        )  # encoder to decoder
        self.gradient_checkpointing = False
        self.config = decoder_config
        self.rotary_emb = LlamaRotaryEmbedding(decoder_config)

    @property
    def dtype(self) -> torch.dtype:
        return self.decoder_embed.weight.dtype

    def get_extended_attention_mask(
        self, attention_mask: torch.Tensor, input_shape: tuple[int], device: torch.device
    ) -> torch.Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (`Tuple[int]`):
                The shape of the input to the model.
            device: (`torch.device`):
                The device of the input to the model.

        Returns:
            `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                extended_attention_mask = self.create_extended_attention_mask_for_decoder(
                    input_shape, attention_mask, device
                )
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        ids_restore: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        interpolate_pos_encoding: bool = False,
    ):
        # Embed tokens
        x = self.decoder_embed(hidden_states)

        batch_size = hidden_states.shape[0]

        # Append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token

        # Unshuffle
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]).to(x_.device))
        hidden_states = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # Position info is supplied via RoPE inside attention; tokens are unshuffled
        # so sequence position equals original spatial position.
        seq_len = hidden_states.shape[1]
        position_ids = torch.arange(
            seq_len, dtype=torch.long, device=hidden_states.device
        ).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Build the attention mask for the decoder blocks.
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, hidden_states.shape[1] - 1), device=hidden_states.device)
        attention_mask = torch.cat(
            (torch.ones((batch_size, 1), device=attention_mask.device), attention_mask), dim=1
        )
        if "flash" in self.config._attn_implementation:
            decoder_attention_mask = attention_mask
        else:
            decoder_attention_mask = self.get_extended_attention_mask(
                attention_mask, hidden_states.shape, attention_mask.device
            ).to(self.dtype)

        # Apply Transformer layers (blocks)
        for layer_module in self.decoder_layers:
            hidden_states = layer_module(
                hidden_states,
                attention_mask=decoder_attention_mask,
                head_mask=None,
                position_embeddings=position_embeddings,
            )

        hidden_states = self.decoder_norm(hidden_states)

        # Predictor projection
        logits = self.decoder_pred(hidden_states)

        # Remove cls token
        logits = logits[:, 1:, :]

        return MPIXELDecoderOutput(logits=logits)


class MPIXELForPreTraining(MPIXELPreTrainedModel):
    def __init__(self, config: MPIXELConfig):
        super().__init__(config)
        self.config = config

        self.vit = MPIXELModel(config)
        self.decoder = MPIXELDecoder(config, num_patches=self.vit.embeddings.max_num_patches)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.vit.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def patchify(self, pixel_values):
        """
        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
                Pixel values.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, num_patches, patch_size**2 * num_channels)`:
                Patchified pixel values.
        """
        patch_size, num_channels = self.config.patch_size, self.config.num_channels
        # sanity checks
        if pixel_values.shape[2] % patch_size != 0 or pixel_values.shape[3] % patch_size != 0:
            raise ValueError("Make sure the pixel values are divisible by the patch size")
        if pixel_values.shape[1] != num_channels:
            raise ValueError(
                "Make sure the number of channels of the pixel values is equal to the one set in the configuration"
            )

        # patchify
        batch_size = pixel_values.shape[0]
        num_patches_h = pixel_values.shape[2] // patch_size
        num_patches_w = pixel_values.shape[3] // patch_size
        patchified_pixel_values = pixel_values.reshape(
            batch_size, num_channels, num_patches_h, patch_size, num_patches_w, patch_size
        )
        patchified_pixel_values = torch.einsum("nchpwq->nhwpqc", patchified_pixel_values)
        patchified_pixel_values = patchified_pixel_values.reshape(
            batch_size, num_patches_h * num_patches_w, patch_size**2 * num_channels
        )
        return patchified_pixel_values

    def unpatchify(self, patchified_pixel_values, original_image_size: Optional[tuple[int, int]] = None):
        """
        Args:
            patchified_pixel_values (`torch.FloatTensor` of shape `(batch_size, num_patches, patch_size**2 * num_channels)`:
                Patchified pixel values.
            original_image_size (`tuple[int, int]`, *optional*):
                Original image size.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`:
                Pixel values.
        """
        patch_size, num_channels = self.config.patch_size, self.config.num_channels
        if original_image_size is None:
            num_patches = patchified_pixel_values.shape[1]
            original_image_size = (patch_size, patch_size * num_patches)
        original_height, original_width = original_image_size
        num_patches_h = original_height // patch_size
        num_patches_w = original_width // patch_size
        # sanity check
        if num_patches_h * num_patches_w != patchified_pixel_values.shape[1]:
            raise ValueError(
                f"The number of patches in the patchified pixel values {patchified_pixel_values.shape[1]}, does not match the number of patches on original image {num_patches_h}*{num_patches_w}"
            )

        # unpatchify
        batch_size = patchified_pixel_values.shape[0]
        patchified_pixel_values = patchified_pixel_values.reshape(
            batch_size,
            num_patches_h,
            num_patches_w,
            patch_size,
            patch_size,
            num_channels,
        )
        patchified_pixel_values = torch.einsum("nhwpqc->nchpwq", patchified_pixel_values)
        pixel_values = patchified_pixel_values.reshape(
            batch_size,
            num_channels,
            num_patches_h * patch_size,
            num_patches_w * patch_size,
        )
        return pixel_values

    def forward_loss(self, pixel_values, pred, mask):
        """
        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
                Pixel values.
            pred (`torch.FloatTensor` of shape `(batch_size, num_patches, patch_size**2 * num_channels)`:
                Predicted pixel values.
            mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
                Tensor indicating which patches are masked (1) and which are not (0).

        Returns:
            `torch.FloatTensor`: Pixel reconstruction loss.
        """
        target = self.patchify(pixel_values)
        if self.config.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    @can_return_tuple
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        patch_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        interpolate_pos_encoding: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MPIXELForPreTrainingOutput:
        r"""
        attention_mask (`torch.FloatTensor` of shape `(batch_size, num_patches)`, *optional*):
            Mask indicating which patch tokens are valid (1) vs. padding (0). Defaults to all ones.
        patch_mask (`torch.FloatTensor` of shape `(batch_size, num_patches)`, *optional*):
            Mask driving controlled (span) masking. If `None`, falls back to random masking.
        interpolate_pos_encoding (`bool`, *optional*, default `False`):
            Whether to interpolate the pre-trained position encodings. This is mainly used to use the model on higher
            resolution images.

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, MPIXELForPreTraining
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
        >>> model = MPIXELForPreTraining.from_pretrained("facebook/vit-mae-base")

        >>> inputs = image_processor(images=image, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> loss = outputs.loss
        >>> mask = outputs.mask
        >>> ids_restore = outputs.ids_restore
        ```"""

        if attention_mask is None:
            attention_mask = torch.ones(
                (pixel_values.shape[0], self.vit.embeddings.max_num_patches),
                device=pixel_values.device,
                dtype=torch.long,
            )

        if patch_mask is None:
            # Preserve legacy pretraining behavior: fallback to random masking when no explicit patch mask is provided.
            patch_mask = torch.rand(attention_mask.shape, device=attention_mask.device)
            noise_mask = torch.argmax(patch_mask * attention_mask, dim=1)
            patch_mask[torch.arange(patch_mask.size(0), device=patch_mask.device), noise_mask] = 100.0

        outputs: MPIXELModelOutput = self.vit(
            pixel_values,
            attention_mask=attention_mask,
            patch_mask=patch_mask,
            head_mask=head_mask,
            interpolate_pos_encoding=interpolate_pos_encoding,
            **kwargs,
        )

        latent = outputs.last_hidden_state
        ids_restore = outputs.ids_restore
        mask = outputs.mask

        decoder_outputs: MPIXELDecoderOutput = self.decoder(
            latent, ids_restore, attention_mask=attention_mask, interpolate_pos_encoding=interpolate_pos_encoding
        )
        logits = decoder_outputs.logits  # shape (batch_size, num_patches, patch_size*patch_size*num_channels)

        merged_mask = torch.bitwise_and(mask == 1, attention_mask == 1).long()
        loss = self.forward_loss(pixel_values, logits, merged_mask)

        return MPIXELForPreTrainingOutput(
            loss=loss,
            logits=logits,
            mask=mask,
            ids_restore=ids_restore,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class MPIXELForSequenceClassification(MPIXELPreTrainedModel):
    def __init__(self, config: MPIXELConfig):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.pooling_mode = PoolingMode.from_string(config.pooling_mode)
        self.vit = MPIXELModel(config)

        # Classifier head
        self.pooler = PoolingForSequenceClassificationHead(
            hidden_size=config.hidden_size,
            hidden_dropout_prob=config.hidden_dropout_prob,
            add_layer_norm=config.pooler_add_layer_norm,
            pooling_mode=self.pooling_mode,
        )
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()

        # Initialize weights and apply final processing
        self.post_init()

    def _init_weights(self, module):
        """
        Initialize the classifier head as `config.head_init` gives. With "pytorch" (the default), the head gets the
        PyTorch defaults, the same as the Lightning fine-tuning code (`mpixel/models/tasks/classification.py`), to
        match the reported runs. With "normal", it gets the HF default of `MPIXELPreTrainedModel`. The encoder is a
        sub-model and always keeps the init of `MPIXELPreTrainedModel`.
        """
        if module is self.classifier or any(module is m for m in self.pooler.modules()):
            if isinstance(module, PMA):
                nn.init.xavier_uniform_(module.S)  # the same as PMA.__init__
            elif self.config.head_init == "normal":
                super()._init_weights(module)
            elif hasattr(module, "reset_parameters"):
                module.reset_parameters()
            return
        super()._init_weights(module)

    @can_return_tuple
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        interpolate_pos_encoding: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> SequenceClassifierOutput:
        r"""
        attention_mask (`torch.FloatTensor` of shape `(batch_size, num_patches)`, *optional*):
            Mask indicating which patch tokens are valid (1) vs. padding (0). Defaults to all ones.
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        interpolate_pos_encoding (`bool`, *optional*, default `False`):
            Whether to interpolate the pre-trained position encodings. This is mainly used to use the model on higher
            resolution images.

        Examples:

        ```python
        >>> from transformers import AutoModelForSequenceClassification
        >>> import mpixel.models.auto_register  # noqa: F401
        >>> from mpixel.processing import MPIXELProcessor

        >>> processor = MPIXELProcessor.from_pretrained("path/to/mpixel-base")
        >>> model = AutoModelForSequenceClassification.from_pretrained("path/to/mpixel-base", num_labels=7)

        >>> inputs = processor(["Hello world!"])
        >>> logits = model(**inputs).logits
        ```"""

        if attention_mask is None:
            attention_mask = torch.ones(
                (pixel_values.shape[0], self.vit.embeddings.max_num_patches), device=pixel_values.device
            )

        outputs: MPIXELModelOutput = self.vit(
            pixel_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            interpolate_pos_encoding=interpolate_pos_encoding,
            **kwargs,
        )

        sequence_output = outputs.last_hidden_state
        if self.pooling_mode == PoolingMode.CLS:
            sequence_output = sequence_output[:, 0, :]
        else:
            # Drop the CLS token, so the patches line up with attention_mask
            sequence_output = sequence_output[:, 1:, :]
        pooled_output = self.pooler(sequence_output, attention_mask)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss = self.loss_function(labels, logits, self.config, **kwargs)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class MPIXELForTokenClassification(MPIXELPreTrainedModel):
    def __init__(self, config: MPIXELConfig):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.vit = MPIXELModel(config)

        # Classifier head
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def _init_weights(self, module):
        """
        Initialize the classifier head with the PyTorch defaults, the same as the Lightning fine-tuning code
        (`mpixel/models/tasks/token_classification.py`), to match the reported runs. The encoder is a sub-model and
        keeps the init of `MPIXELPreTrainedModel`.
        """
        if module is self.classifier:
            module.reset_parameters()
            return
        super()._init_weights(module)

    @can_return_tuple
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        interpolate_pos_encoding: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> TokenClassifierOutput:
        r"""
        attention_mask (`torch.FloatTensor` of shape `(batch_size, num_patches)`, *optional*):
            Mask indicating which patch tokens are valid (1) vs. padding (0). Defaults to all ones.
        labels (`torch.LongTensor` of shape `(batch_size, num_patches)`, *optional*):
            Labels for computing the token classification loss, one per patch. Indices should be in `[0, ...,
            config.num_labels - 1]`. Patches with the label `-100` are ignored.
        interpolate_pos_encoding (`bool`, *optional*, default `False`):
            Whether to interpolate the pre-trained position encodings. This is mainly used to use the model on higher
            resolution images.

        Examples:

        ```python
        >>> from transformers import AutoModelForTokenClassification
        >>> import mpixel.models.auto_register  # noqa: F401
        >>> from mpixel.processing import MPIXELProcessor

        >>> processor = MPIXELProcessor.from_pretrained("path/to/mpixel-base")
        >>> model = AutoModelForTokenClassification.from_pretrained("path/to/mpixel-base", num_labels=7)

        >>> inputs = processor(["Hello world!"])
        >>> logits = model(**inputs).logits
        ```"""

        outputs: MPIXELModelOutput = self.vit(
            pixel_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            interpolate_pos_encoding=interpolate_pos_encoding,
            **kwargs,
        )

        # Drop the CLS token, so patch i of the input gets logits[:, i]
        sequence_output = outputs.last_hidden_state[:, 1:, :]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.config, **kwargs)

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "MPIXELForPreTraining",
    "MPIXELForSequenceClassification",
    "MPIXELForTokenClassification",
    "MPIXELLayer",
    "MPIXELModel",
    "MPIXELPreTrainedModel",
]

