"""S1 — Per-epoch neighbor embedding cache.

Avoids re-encoding the up-to-K context neighbors of every batch through the
text encoder by computing embeddings for ALL context documents once at the
start of each epoch and serving subsequent batches via O(1) lookup.

The anchor document still flows through the live encoder so gradients reach
SciBERT/LoRA. Only context neighbors are served from this frozen-per-epoch
snapshot.
"""

from __future__ import annotations

from typing import Any

import polars as pl
import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase


class NeighborEmbeddingCache:
    """Holds a per-doc text embedding tensor refreshed each epoch."""

    def __init__(
        self,
        documents: pl.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        text_dim: int,
        max_seq_length: int,
        device: torch.device,
        store_dtype: torch.dtype = torch.float16,
        store_on_cpu: bool = False,
    ) -> None:
        self.tokenizer       = tokenizer
        self.text_dim        = int(text_dim)
        self.max_seq_length  = int(max_seq_length)
        self.device          = device
        self.store_dtype     = store_dtype
        self.store_device    = torch.device("cpu") if store_on_cpu else device

        rows = list(documents.iter_rows(named=True))
        self.doc_id_to_idx: dict[int, int] = {int(r["doc_id"]): i for i, r in enumerate(rows)}
        n = len(rows)

        # Pre-tokenize once at this lower seq-length.
        all_input_ids = torch.zeros(n, self.max_seq_length, dtype=torch.long)
        all_attn_mask = torch.zeros(n, self.max_seq_length, dtype=torch.long)
        for i, row in enumerate(rows):
            enc = tokenizer(
                str(row.get("title", "") or ""),
                str(row.get("abstract", "") or ""),
                max_length=self.max_seq_length, padding="max_length",
                truncation=True, return_tensors="pt")
            all_input_ids[i] = enc["input_ids"][0]
            all_attn_mask[i] = enc["attention_mask"][0]
        self.input_ids      = all_input_ids
        self.attention_mask = all_attn_mask

        self.embeddings = torch.zeros(n, self.text_dim, dtype=store_dtype, device=self.store_device)
        self._built = False

    @torch.no_grad()
    def rebuild(self, text_encoder: Any, batch_size: int = 128) -> None:
        """Re-encode all docs with current text_encoder weights."""
        was_training = text_encoder.training
        text_encoder.eval()
        use_cuda = self.device.type == "cuda" if isinstance(self.device, torch.device) else False
        amp_dtype = torch.bfloat16 if use_cuda and torch.cuda.is_bf16_supported() else torch.float16
        try:
            n = self.input_ids.size(0)
            for i in range(0, n, batch_size):
                ids  = self.input_ids[i:i + batch_size].to(self.device, non_blocking=True)
                mask = self.attention_mask[i:i + batch_size].to(self.device, non_blocking=True)
                if use_cuda:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        emb = text_encoder(ids, mask)
                else:
                    emb = text_encoder(ids, mask)
                self.embeddings[i:i + batch_size] = emb.to(self.embeddings.dtype).to(self.store_device)
        finally:
            if was_training:
                text_encoder.train()
        self._built = True

    def lookup(self, context_doc_ids: Tensor) -> Tensor:
        """Map a (B, M) tensor of doc ids to a (B, M, text_dim) embedding tensor.

        Unknown / padding ids (not in the index) collapse to row 0; downstream
        masking via context_mask zeroes out those slots.
        """
        shape = context_doc_ids.shape
        flat  = context_doc_ids.reshape(-1).detach().cpu().tolist()
        idx   = [self.doc_id_to_idx.get(int(d), 0) for d in flat]
        idx_t = torch.tensor(idx, dtype=torch.long, device=self.embeddings.device)
        emb   = self.embeddings.index_select(0, idx_t)
        return emb.to(context_doc_ids.device).float().view(*shape, self.text_dim)
