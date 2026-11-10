"""
Sequence-classification wrapper that initialises from a MPIXELForPreTraining
Lightning checkpoint or HF model folder.  The decoder is discarded; a pooler +
linear head is appended on top of the shared ViT encoder.
"""

import warnings

import torch
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput

from mpixel.models.pooling import PoolingForSequenceClassificationHead, PoolingMode
from mpixel.utils.checkpoint import load_pretrained_state_dict


class PixelForSequenceClassification(nn.Module):
    """
    Wraps a *ForPreTraining encoder loaded from a Lightning checkpoint and adds
    a sequence classification head.

    Exposes `.vit`, `.config`, and a `forward(pixel_values, attention_mask,
    return_dict)` signature compatible with TextClassificationExperiment.
    """

    def __init__(
        self,
        pretraining_model: nn.Module,
        num_labels: int,
        pooling_mode: PoolingMode = PoolingMode.MEAN,
        add_layer_norm: bool = True,
    ):
        super().__init__()
        self.config = pretraining_model.config
        self.config.num_labels = num_labels
        self.num_labels = num_labels
        self.pooling_mode = pooling_mode

        self.vit = pretraining_model.vit

        hidden_dropout_prob = getattr(self.config, "hidden_dropout_prob", 0.0)
        self.pooler = PoolingForSequenceClassificationHead(
            hidden_size=self.config.hidden_size,
            hidden_dropout_prob=hidden_dropout_prob,
            add_layer_norm=add_layer_norm,
            pooling_mode=pooling_mode,
        )
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

    def forward(
        self,
        pixel_values=None,
        attention_mask=None,
        return_dict=True,
    ):
        outputs = self.vit(pixel_values, attention_mask=attention_mask)

        if self.pooling_mode == PoolingMode.CLS:
            sequence_output = outputs.last_hidden_state[:, 0, :]
        else:
            # Discard CLS token; use original attention_mask for patch positions
            sequence_output = outputs.last_hidden_state[:, 1:, :]

        logits = self.pooler(sequence_output, attention_mask)
        logits = self.classifier(logits)

        return SequenceClassifierOutput(logits=logits)


def load_for_sequence_classification(
    checkpoint_path: str,
    config,
    pretraining_class,
    num_labels: int,
    pooling_mode: PoolingMode = PoolingMode.MEAN,
    add_layer_norm: bool = True,
) -> PixelForSequenceClassification:
    """
    Build a PixelForSequenceClassification from a Lightning checkpoint or an HF model folder. Both give the same
    model, and with the same seed the same head.

    Args:
        checkpoint_path: Path to the .ckpt file, or an HF model folder or Hub ID
            (scripts/utils/convert_to_huggingface.py).
        config: Pre-loaded HuggingFace PretrainedConfig for the encoder.
        pretraining_class: MPIXELForPreTraining.
        num_labels: Number of output classes.
        pooling_mode: Pooling strategy for the classification head.
        add_layer_norm: Whether the pooler includes a layer norm.
    """
    pretraining_model = pretraining_class._from_config(config)
    state_dict = load_pretrained_state_dict(checkpoint_path)
    result = pretraining_model.load_state_dict(state_dict, strict=False)
    non_decoder_missing = [k for k in result.missing_keys if not k.startswith("decoder.")]
    if non_decoder_missing:
        warnings.warn(f"Missing non-decoder keys when loading checkpoint: {non_decoder_missing[:10]}")
    return PixelForSequenceClassification(
        pretraining_model, num_labels, pooling_mode, add_layer_norm
    )
