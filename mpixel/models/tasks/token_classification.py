"""
Token-classification wrapper that initialises from a MPIXELForPreTraining
Lightning checkpoint or HF model folder.  The decoder is discarded; a dropout +
linear head is appended on top of the shared ViT encoder, producing one logit
vector per patch.
"""

import warnings

import torch.nn as nn
from transformers.modeling_outputs import TokenClassifierOutput

from mpixel.utils.checkpoint import load_pretrained_state_dict


class PixelForTokenClassification(nn.Module):
    """
    Wraps a *ForPreTraining encoder loaded from a Lightning checkpoint and adds
    a per-patch token classification head.

    Exposes `.vit`, `.config`, and a `forward(pixel_values, attention_mask,
    return_dict)` signature compatible with NERExperiment. The CLS token is
    discarded; each remaining patch position gets its own logits.
    """

    def __init__(
        self,
        pretraining_model: nn.Module,
        num_labels: int,
    ):
        super().__init__()
        self.config = pretraining_model.config
        self.config.num_labels = num_labels
        self.num_labels = num_labels

        self.vit = pretraining_model.vit

        hidden_dropout_prob = getattr(self.config, "hidden_dropout_prob", 0.0)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

    def forward(
        self,
        pixel_values=None,
        attention_mask=None,
        return_dict=True,
    ):
        outputs = self.vit(pixel_values, attention_mask=attention_mask)

        # Discard CLS token; classify each patch position
        sequence_output = outputs.last_hidden_state[:, 1:, :]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        return TokenClassifierOutput(logits=logits)


def load_for_token_classification(
    checkpoint_path: str,
    config,
    pretraining_class,
    num_labels: int,
) -> PixelForTokenClassification:
    """
    Build a PixelForTokenClassification from a Lightning checkpoint or an HF model folder. Both give the same
    model, and with the same seed the same head.

    Args:
        checkpoint_path: Path to the .ckpt file, or an HF model folder or Hub ID
            (scripts/utils/convert_to_huggingface.py).
        config: Pre-loaded HuggingFace PretrainedConfig for the encoder.
        pretraining_class: MPIXELForPreTraining.
        num_labels: Number of output classes.
    """
    pretraining_model = pretraining_class._from_config(config)
    state_dict = load_pretrained_state_dict(checkpoint_path)
    result = pretraining_model.load_state_dict(state_dict, strict=False)
    non_decoder_missing = [k for k in result.missing_keys if not k.startswith("decoder.")]
    if non_decoder_missing:
        warnings.warn(f"Missing non-decoder keys when loading checkpoint: {non_decoder_missing[:10]}")
    return PixelForTokenClassification(pretraining_model, num_labels)
