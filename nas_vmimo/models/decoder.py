"""
Semantic Decoder for NAS-VMIMO.

After ZF equalization the receiver gets ŝ ∈ C^{r} (or real approximation).
The decoder maps that back into the embedding space so the loss can compare
reconstructed embeddings with the original ones.

Architecture mirrors the encoder in depth and width (symmetric), but the
direction is reversed:
    ŝ  ─→  [Linear→LN→GELU]  ─→  shared decoder backbone
                                  ─→  [per-modality projection]
                                  ─→  {ê^(m)} + task_emb_pred

The shared backbone is a small Transformer (same block type as encoder).
Since the received stream ŝ has dimension r (number of spatial DoFs),
we first project it up to d_model, then add a CLS slot, then run L layers.

Key difference from the encoder:
  - Input is ŝ (B, r), a compressed representation of all modalities
  - We cannot recover M separate token slots from a single r-dim stream
    unless we know the VMIMO case used.  So we use a *single* context token
    plus M learnable query tokens to "decode" each modality back.
  - This is a cross-attention decoder: queries are modal slots, keys/values
    come from the projected ŝ.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder import TransformerBlock, MLP


# ─────────────────────────────────────────────────────────────────────
#  Received-stream projector:  ŝ ∈ R^r  →  context token ∈ R^{d_model}
# ─────────────────────────────────────────────────────────────────────
class StreamProjector(nn.Module):
    """Projects the equalized stream ŝ into the d_model embedding space."""

    def __init__(self, r: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(r, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, s_hat: torch.Tensor) -> torch.Tensor:
        """s_hat: (B, r)  →  (B, 1, d_model)"""
        return self.proj(s_hat).unsqueeze(1)


# ─────────────────────────────────────────────────────────────────────
#  Cross-attention decoder layer
# ─────────────────────────────────────────────────────────────────────
class CrossAttentionBlock(nn.Module):
    """
    Single cross-attention decoder block.
    Query  = modal query tokens  (B, M+1, d_model)
    Key/V  = projected stream    (B, 1,   d_model)
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm_q  = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.norm_ff = nn.LayerNorm(d_model)
        self.cross   = nn.MultiheadAttention(d_model, n_heads,
                                              dropout=dropout, batch_first=True)
        self.self_   = nn.MultiheadAttention(d_model, n_heads,
                                              dropout=dropout, batch_first=True)
        self.norm_sa = nn.LayerNorm(d_model)
        self.mlp     = MLP(d_model, d_ff, dropout)
        self.drop    = nn.Dropout(dropout)

    def forward(
        self,
        queries: torch.Tensor,    # (B, M+1, d_model)
        context: torch.Tensor,    # (B, 1,   d_model)
    ) -> torch.Tensor:
        # Self-attention among query slots
        q_norm = self.norm_sa(queries)
        sa_out, _ = self.self_(q_norm, q_norm, q_norm)
        queries = queries + self.drop(sa_out)
        # Cross-attention: queries attend to stream context
        q_norm = self.norm_q(queries)
        c_norm = self.norm_kv(context)
        ca_out, _ = self.cross(q_norm, c_norm, c_norm)
        queries = queries + self.drop(ca_out)
        # FFN
        queries = queries + self.drop(self.mlp(self.norm_ff(queries)))
        return queries


# ─────────────────────────────────────────────────────────────────────
#  Full Semantic Decoder
# ─────────────────────────────────────────────────────────────────────
class SemanticDecoder(nn.Module):
    """
    Maps equalized stream ŝ → reconstructed embeddings {ê^(m)} + task_emb_pred.

    Structure:
      1. StreamProjector:   ŝ (B,r) → context (B,1,d)
      2. Learnable queries: [q_task | q_1 | ... | q_M]  (M+1 slots)
      3. CrossAttentionBlock × n_layers
      4. Read out: task_emb_pred = slot_0,  ê^(m) = slot_m

    Args:
        r         : stream dimension (= spatial DoFs, rank of H)
        d_model   : embedding dimension (must match encoder)
        n_modal   : number of modalities M
        n_heads   : attention heads
        n_layers  : decoder depth
        d_ff      : FFN width
        dropout   : dropout
    """

    def __init__(
        self,
        r:        int,
        d_model:  int,
        n_modal:  int,
        n_heads:  int   = 4,
        n_layers: int   = 2,
        d_ff:     Optional[int] = None,
        dropout:  float = 0.1,
    ):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model

        self.n_modal = n_modal
        self.d_model = d_model

        # Step 1: project stream to context token
        self.stream_proj = StreamProjector(r, d_model, dropout)

        # Step 2: learnable query tokens (task slot + M modal slots)
        self.queries = nn.Parameter(torch.zeros(1, 1 + n_modal, d_model))
        nn.init.trunc_normal_(self.queries, std=0.02)

        # Step 3: cross-attention decoder blocks
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        s_hat: torch.Tensor,   # (B, r)  – equalized received stream
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Returns:
            task_emb_pred : (B, d_model)         – reconstructed task embedding
            modal_embs    : list of M (B, d_model)– reconstructed modal embeddings
        """
        B = s_hat.shape[0]

        # Project stream → context
        context = self.stream_proj(s_hat)           # (B, 1, d_model)

        # Expand learnable queries to batch
        queries = self.queries.expand(B, -1, -1)    # (B, 1+M, d_model)

        # Cross-attention decode
        for block in self.blocks:
            queries = block(queries, context)
        queries = self.norm(queries)                # (B, 1+M, d_model)

        # Readout
        task_emb_pred = queries[:, 0, :]                              # (B, d)
        modal_embs    = [queries[:, m+1, :] for m in range(self.n_modal)]

        return task_emb_pred, modal_embs
