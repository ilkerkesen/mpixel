import torch
import torch.nn as nn

from .patch_embeddings import PatchEmbeddings


class PIXELEmbeddingsRoPE(nn.Module):
    """
    Mirrors PIXELEmbeddings but drops the additive absolute position embeddings.
    Position information is supplied to the model via rotary position embeddings
    applied inside attention. Returns position_ids built from the original
    (pre-shuffle) patch indices, so each kept patch carries its true spatial
    position regardless of shuffle order.
    """

    def __init__(self, config):
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.patch_embeddings = PatchEmbeddings(
            max_num_patches=config.max_num_patches,
            patch_size=config.patch_size,
            num_channels=config.num_channels,
            embed_dim=config.hidden_size,
        )
        self.max_num_patches = config.max_num_patches
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config
        self.initialize_weights()

    def initialize_weights(self):
        self.patch_embeddings.initialize_weights(
            method=getattr(self.config, "patch_embedding_init", None),
        )
        torch.nn.init.normal_(self.cls_token, std=self.config.initializer_range)

    def controlled_masking(self, sequence, attention_mask, patch_mask):
        batch_size, seq_length, dim = sequence.shape

        len_keep = int(seq_length * (1 - self.config.mask_ratio))

        noise = patch_mask + (torch.rand(batch_size, seq_length, device=sequence.device) / 100)

        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        sequence_masked = torch.gather(sequence, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, dim))
        attention_mask_masked = torch.gather(attention_mask, dim=1, index=ids_keep)

        mask = torch.ones([batch_size, seq_length], device=sequence.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return sequence_masked, attention_mask_masked, mask, ids_restore, ids_keep

    def forward(self, pixel_values, attention_mask=None, patch_mask=None, interpolate_pos_encoding=False):
        batch_size = pixel_values.shape[0]
        embeddings = self.patch_embeddings(pixel_values)

        if patch_mask is not None:
            embeddings, attention_mask, mask, ids_restore, ids_keep = self.controlled_masking(
                embeddings, attention_mask, patch_mask
            )

            cls_tokens = self.cls_token.expand(embeddings.shape[0], -1, -1)
            embeddings = torch.cat((cls_tokens, embeddings), dim=1)
            attention_mask = torch.cat(
                (torch.ones((batch_size, 1), device=attention_mask.device), attention_mask), dim=1
            )

            # CLS at position 0; kept patches retain their pre-shuffle index (+1 to leave room for CLS)
            cls_position = torch.zeros((batch_size, 1), dtype=ids_keep.dtype, device=ids_keep.device)
            position_ids = torch.cat([cls_position, ids_keep + 1], dim=1)

            return embeddings, attention_mask, mask, ids_restore, position_ids
        else:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            embeddings = torch.cat((cls_tokens, embeddings), dim=1)

            if attention_mask is not None:
                attention_mask = torch.cat(
                    (torch.ones((batch_size, 1), device=attention_mask.device), attention_mask), dim=1
                )

            embeddings = self.dropout(embeddings)
            position_ids = torch.arange(
                embeddings.shape[1], dtype=torch.long, device=embeddings.device
            ).unsqueeze(0).expand(batch_size, -1)

            return embeddings, attention_mask, None, None, position_ids
