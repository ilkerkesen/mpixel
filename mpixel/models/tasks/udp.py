"""
Biaffine dependency-parsing wrapper that initialises from a
MPIXELForPreTraining Lightning checkpoint or HF model folder.  The decoder is
discarded; two Biaffine heads are appended on top of the shared ViT encoder.
"""

import warnings

import torch
import torch.nn as nn
import mpixel.models.auto_register  # noqa: F401 — registers custom model types
from mpixel.datamodules.udp import DependencyParsingModelOutput
from mpixel.models.tasks.biaffine import Biaffine
from mpixel.utils.checkpoint import (
    config_from_checkpoint,
    load_hf_model_config,
    load_pretrained_state_dict,
    load_state_dict,
)


class PixelForBiaffineParsing(nn.Module):
    """
    Wraps a *ForPreTraining encoder loaded from a Lightning checkpoint and adds
    biaffine arc and relation heads for dependency parsing.

    Exposes `.vit`, `.config`, and a `forward(pixel_values, attention_mask,
    word_starts, arc_labels, rel_labels, return_dict)` signature.

    word_starts: LongTensor (B, max_num_patches+1) — patch index of the first
    patch of each word, followed by a sentinel (= num_text_patches) that marks
    the end of the last word, then -100 padding.
    """

    def __init__(
        self,
        pretraining_model: nn.Module,
        num_labels: int,
        pad_token_id: int = -100,
    ):
        super().__init__()
        self.config = pretraining_model.config
        self.config.num_labels = num_labels
        self.num_labels = num_labels
        self.pad_token_id = pad_token_id

        self.vit = pretraining_model.vit

        hidden_size = self.config.hidden_size
        hidden_dropout_prob = getattr(self.config, "hidden_dropout_prob", 0.0)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.pooler = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        # Match pixel-m4's post_init: HF's ViTPreTrainedModel._init_weights
        # draws Normal(0, initializer_range) for Linear and zeros the bias.
        pretraining_model._init_weights(self.pooler[0])
        self.biaffine_arcs = Biaffine(n_in=hidden_size, bias_x=True, bias_y=False)
        self.biaffine_rels = Biaffine(n_in=hidden_size, n_out=num_labels, bias_x=True, bias_y=True)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        pixel_values=None,
        attention_mask=None,
        word_starts=None,
        arc_labels=None,
        rel_labels=None,
        return_dict=True,
    ):
        outputs = self.vit(pixel_values, attention_mask=attention_mask)

        # CLS token → root representation (pooled via Linear+Tanh, mirroring HF
        # ViT pooler); patch tokens → word representations
        cls_output = self.pooler(outputs.last_hidden_state[:, 0, :])          # (B, H)
        patch_output = self.dropout(outputs.last_hidden_state[:, 1:, :])  # (B, S-1, H)

        word_outputs_deps = self._merge_word_patches(patch_output, word_starts)
        # Prepend CLS as the representation of the artificial root node
        word_outputs_heads = torch.cat([cls_output.unsqueeze(1), word_outputs_deps], dim=1)

        arc_logits = self.biaffine_arcs(word_outputs_deps, word_outputs_heads).squeeze(1)
        rel_logits = self.biaffine_rels(word_outputs_deps, word_outputs_heads).permute(0, 2, 3, 1)

        loss = None
        if arc_labels is not None and rel_labels is not None:
            loss = self._compute_loss(arc_logits, rel_logits, arc_labels, rel_labels)

        if arc_logits.dim() == 2:
            arc_logits = arc_logits.unsqueeze(0)

        return DependencyParsingModelOutput(
            loss=loss,
            arc_logits=arc_logits,
            rel_logits=rel_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _merge_word_patches(self, patch_output: torch.Tensor, word_starts: torch.Tensor) -> torch.Tensor:
        """
        Average patch embeddings within each word's patch span to produce one
        vector per word.

        patch_output: (B, S, H) — patch hidden states (CLS already removed)
        word_starts:  (B, max_num_patches+1) — start indices + end sentinel + -100 pad
        Returns:      (B, S, H) — word vectors, zero-padded to length S
        """
        B, S, H = patch_output.shape
        zero = torch.zeros(1, H, device=patch_output.device)
        instances = []

        for i in range(B):
            vecs = patch_output[i]                          # (S, H)
            starts = word_starts[i]
            starts = starts[starts.ne(self.pad_token_id)]   # strip -100 padding

            word_vecs = []
            for j in range(len(starts) - 1):               # sentinel drives the last word's end
                start, end = int(starts[j]), int(starts[j + 1])
                span = vecs[start:end]
                word_vecs.append(span.mean(0, keepdim=True) if len(span) > 1 else vecs[start].unsqueeze(0))

            # Pad to sequence length S
            word_vecs += [zero] * (S - len(word_vecs))
            instances.append(torch.cat(word_vecs, dim=0).unsqueeze(0))

        return torch.cat(instances, dim=0)

    def _compute_loss(
        self,
        arc_logits: torch.Tensor,
        rel_logits: torch.Tensor,
        arc_labels: torch.Tensor,
        rel_labels: torch.Tensor,
    ) -> torch.Tensor:
        if arc_logits.dim() == 2:
            arc_logits = arc_logits.unsqueeze(0)

        mask = arc_labels.ne(self.pad_token_id)
        arc_scores, arcs = arc_logits[mask], arc_labels[mask]
        loss = self.loss_fn(arc_scores, arcs)

        rel_scores = rel_logits[mask]                           # (n_valid, n_heads, n_labels)
        rel_scores = rel_scores[torch.arange(len(arcs)), arcs]  # select gold head's rel scores
        loss += self.loss_fn(rel_scores, rel_labels[mask])

        return loss


def load_for_biaffine_parsing(
    checkpoint_path: str,
    pretraining_class,
    num_labels: int,
    dropout: float = 0.1,
    pad_token_id: int = -100,
) -> PixelForBiaffineParsing:
    """
    Build a PixelForBiaffineParsing from a Lightning pretraining checkpoint or an HF model folder. The config is
    read from the checkpoint's own model_config entry, or from the config.json of the folder. Both give the same
    model, and with the same seed the same heads.

    Args:
        checkpoint_path: Path to the .ckpt file, or an HF model folder or Hub ID
            (scripts/utils/convert_to_huggingface.py).
        pretraining_class: MPIXELForPreTraining.
        num_labels: Number of dependency relation labels.
        dropout: Dropout override (attention + hidden); overrides the
            checkpoint config.
        pad_token_id: Value used to pad word_starts / label tensors.
    """
    if checkpoint_path.endswith(".ckpt"):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = config_from_checkpoint(checkpoint["model_config"])
        state_dict = load_state_dict(checkpoint)
    else:
        config = load_hf_model_config(checkpoint_path)
        state_dict = load_pretrained_state_dict(checkpoint_path)
    config.num_labels = num_labels
    config.attention_probs_dropout_prob = dropout
    config.hidden_dropout_prob = dropout

    pretraining_model = pretraining_class._from_config(config)
    result = pretraining_model.load_state_dict(state_dict, strict=False)
    non_decoder_missing = [k for k in result.missing_keys if not k.startswith("decoder.")]
    if non_decoder_missing:
        warnings.warn(f"Missing non-decoder keys when loading checkpoint: {non_decoder_missing[:10]}")
    return PixelForBiaffineParsing(pretraining_model, num_labels, pad_token_id)
