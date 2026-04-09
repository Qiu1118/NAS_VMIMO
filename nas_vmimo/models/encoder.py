"""
Multimodal Encoder for NAS-VMIMO.

Follows the Meta-Transformer architecture (Zhang et al., 2023):
  Step 1 — Data-to-Sequence tokenization  (per-modality, trainable)
  Step 2 — Shared Transformer encoder    (modality-agnostic backbone)
  Step 3 — Per-modality token readout    (for VMIMO embedding interface)

Key design decisions vs. the original Meta-Transformer:
  ┌─────────────────┬──────────────────────┬───────────────────────────┐
  │ Aspect          │ Original MT          │ Our adaptation            │
  ├─────────────────┼──────────────────────┼───────────────────────────┤
  │ Input           │ images/audio/etc.    │ feature vectors (synthetic)│
  │ Tokenizer       │ patch conv / FPS+KNN │ Linear projection (1 token)│
  │ Backbone        │ frozen LAION-2B ViT  │ trainable (NAS searches it)│
  │ Output used     │ CLS token only       │ CLS (task) + per-modal tkns│
  │ Position emb.   │ learnable 1D         │ learnable 1D + modal-type  │
  └─────────────────┴──────────────────────┴───────────────────────────┘

Token layout in the shared sequence (following Equ. 7 of the paper):
    z_0 = [x_CLS | E_1·x^(1) | E_2·x^(2) | ... | E_M·x^(M)] + E_pos

For our synthetic modalities each x^(m) ∈ R^{d_m} is projected to a
single d_model-dimensional token, so N_tokens = 1 + M (CLS + M modalities).
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────
#  1.  Per-modality tokenizer  (Data-to-Sequence, Step 1)
# ─────────────────────────────────────────────────────────────────────
class ModalityTokenizer(nn.Module):
    """
    Lightweight per-modality linear projection that maps a raw feature
    vector x^(m) ∈ R^{d_m} to a single token t^(m) ∈ R^{d_model}.

    This is the simplest instance of the Meta-Transformer "meta-scheme"
    (Fig 3a in the paper): Grouping(=identity) → Convolution(=Linear)
    → Transformation(=LayerNorm).

    Each modality has its own projection matrix (trainable), but they all
    project to the same d_model space, enabling the shared encoder.
    """

    def __init__(self, d_in: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.proj    = nn.Linear(d_in, d_model)
        self.norm    = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, d_in)   – raw per-modality feature vector
        Returns (B, 1, d_model) – single token in shared manifold space
        """
        t = self.dropout(self.norm(self.proj(x)))   # (B, d_model)
        return t.unsqueeze(1)                        # (B, 1, d_model)


# ─────────────────────────────────────────────────────────────────────
#  2.  Transformer block  (from timm, reproduced without dependency)
# ─────────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1  = nn.Linear(d_model, d_ff)
        self.fc2  = nn.Linear(d_ff, d_model)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.drop(self.act(self.fc1(x))))


class TransformerBlock(nn.Module):
    """
    Standard ViT block as used in Meta-Transformer backbone.
    Matches timm's Block(dim, num_heads, mlp_ratio, qkv_bias, norm_layer, act_layer).

    Equ. 8–9 from the paper:
        z'_l  = MSA(LN(z_{l-1})) + z_{l-1}
        z_l   = MLP(LN(z'_l))   + z'_l
    """

    def __init__(
        self,
        d_model:   int,
        n_heads:   int,
        d_ff:      int,
        dropout:   float = 0.1,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(
            d_model, n_heads,
            dropout=attn_drop,
            batch_first=True,
        )
        self.mlp   = MLP(d_model, d_ff, dropout)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # MSA sub-layer
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.drop(attn_out)
        # MLP sub-layer
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x


# ─────────────────────────────────────────────────────────────────────
#  3.  Shared Transformer Encoder  (modality-agnostic backbone, Step 2)
# ─────────────────────────────────────────────────────────────────────
class SharedTransformerEncoder(nn.Module):
    """
    The modality-shared ViT backbone from Meta-Transformer (Sec. 3.3).

    Input : concatenated token sequence
            z_0 = [CLS | tok_1 | tok_2 | ... | tok_M] + pos_emb
            shape (B, 1+M, d_model)
    Output: encoded sequence  z_L  (B, 1+M, d_model)

    The CLS token output z_L[:,0,:] is used as the global task embedding.
    The per-modality token outputs z_L[:,1:,:] are the semantic embeddings
    e^(m) fed into VMIMO.

    Args:
        d_model  : embedding dimension
        n_heads  : attention heads
        n_layers : number of stacked TransformerBlocks  (NAS will vary this)
        d_ff     : feed-forward width  (defaults to 4×d_model as in ViT)
        n_tokens : total sequence length = 1 (CLS) + M (modalities)
        dropout  : dropout rate
    """

    def __init__(
        self,
        d_model:  int,
        n_heads:  int,
        n_layers: int,
        n_tokens: int,
        d_ff:     Optional[int] = None,
        dropout:  float = 0.1,
    ):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model

        # Learnable CLS token (Equ. 7)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Learnable 1D position embeddings (Equ. 7)
        self.pos_emb = nn.Parameter(torch.zeros(1, n_tokens, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        # Modal-type embeddings: one learnable vector per modality slot
        # (0 = CLS slot; 1..M = modality slots)
        self.modal_type_emb = nn.Parameter(torch.zeros(1, n_tokens, d_model))
        nn.init.trunc_normal_(self.modal_type_emb, std=0.02)

        # Stacked Transformer blocks (Equ. 8–9)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)    # final LN (Equ. 10)

    def forward(
        self,
        modal_tokens: List[torch.Tensor],    # list of M (B, 1, d_model) tensors
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Returns:
          task_emb   : (B, d_model)         – CLS token output (z_L^0)
          modal_embs : list of M (B, d_model)– per-modality token outputs
        """
        B = modal_tokens[0].shape[0]

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)          # (B, 1, d_model)
        seq = torch.cat([cls] + modal_tokens, dim=1)     # (B, 1+M, d_model)

        # Add position and modal-type embeddings
        seq = seq + self.pos_emb + self.modal_type_emb  # (B, 1+M, d_model)

        # Pass through stacked blocks
        for block in self.blocks:
            seq = block(seq)

        seq = self.norm(seq)                             # (B, 1+M, d_model)

        # Readout
        task_emb   = seq[:, 0, :]                        # (B, d_model)  – CLS
        modal_embs = [seq[:, i+1, :] for i in range(len(modal_tokens))]

        return task_emb, modal_embs


# ─────────────────────────────────────────────────────────────────────
#  4.  Full Multimodal Encoder  (tokenizers + shared backbone)
# ─────────────────────────────────────────────────────────────────────
class MultimodalMetaEncoder(nn.Module):
    """
    End-to-end multimodal encoder following Meta-Transformer style.

    Pipeline:
        {x^(m)} → [ModalityTokenizer_m] → {tok^(m)} ∈ R^{B×1×d_model}
                → SharedTransformerEncoder → task_emb, {e^(m)}

    The output {e^(m)} is fed directly into the VMIMO Controller.

    Args:
        modal_dims  : list of input dims per modality  [128, 32, 32, 64, 64]
        d_model     : shared embedding dimension       (NAS searches: 64–256)
        n_heads     : attention heads                  (NAS searches: 2–8)
        n_layers    : Transformer depth                (NAS searches: 1–6)
        d_ff        : FFN width                        (defaults to 4×d_model)
        dropout     : dropout
        freeze_backbone : if True, freeze SharedTransformerEncoder (Meta-MT style)
                          if False (default), full E2E training
    """

    def __init__(
        self,
        modal_dims:       List[int],
        d_model:          int   = 128,
        n_heads:          int   = 4,
        n_layers:         int   = 2,
        d_ff:             Optional[int] = None,
        dropout:          float = 0.1,
        freeze_backbone:  bool  = False,
    ):
        super().__init__()
        self.modal_dims = modal_dims
        self.d_model    = d_model
        n_modalities    = len(modal_dims)

        # Step 1: per-modality tokenizers
        self.tokenizers = nn.ModuleList([
            ModalityTokenizer(d_in, d_model, dropout)
            for d_in in modal_dims
        ])

        # Step 2: shared encoder (1 CLS + M modal tokens)
        self.encoder = SharedTransformerEncoder(
            d_model  = d_model,
            n_heads  = n_heads,
            n_layers = n_layers,
            n_tokens = 1 + n_modalities,
            d_ff     = d_ff,
            dropout  = dropout,
        )

        if freeze_backbone:
            for p in self.encoder.parameters():
                p.requires_grad_(False)

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
        modalities: Dict[str, torch.Tensor],   # {name: (B, d_m)}
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            modalities : dict mapping modality name → raw feature tensor (B, d_m)
                         Keys must match the order given in modal_dims at init.

        Returns:
            task_emb   : (B, d_model)         – CLS token; task-level embedding
            modal_embs : list of M (B, d_model)– per-modality semantic embeddings
                         fed into VMIMO Controller as e^(m)
        """
        # Tokenise each modality in order
        modal_tokens = [
            tok(x) for tok, x in zip(self.tokenizers, modalities.values())
        ]
        # Shared encoder → CLS + per-modal embeddings
        task_emb, modal_embs = self.encoder(modal_tokens)
        return task_emb, modal_embs

    # ── NAS interface ──────────────────────────────────────────────
    def count_params(self) -> int:
        """Total trainable parameter count  (for NAS P_total constraint)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def estimated_flops(self, seq_len: Optional[int] = None) -> float:
        """
        Rough FLOPs estimate for NAS resource filtering (Equ. 15).
        seq_len defaults to 1 + M.
        Approximation: 4 * L * d^2 * N  (attention + FFN dominant terms).
        """
        M = len(self.modal_dims)
        N = seq_len or (1 + M)
        L = len(self.encoder.blocks)
        d = self.d_model
        # Attention: 2 * N^2 * d  (QK and AV)
        attn_flops = 2 * (N ** 2) * d * L
        # FFN: 2 * N * d * d_ff  ≈ 2 * N * d * 4d = 8 * N * d^2
        ffn_flops  = 8 * N * (d ** 2) * L
        return float(attn_flops + ffn_flops)


# ─────────────────────────────────────────────────────────────────────
#  5.  NAS-searchable variant  (architecture parameters α_{i,j})
# ─────────────────────────────────────────────────────────────────────
class MixedOp(nn.Module):
    """
    DARTS-style mixed operation (Equ. 31 from our paper):
        output = Σ_i  α_i * op_i(x)

    Used inside the NAS search space to soft-select between candidate
    Transformer operations (common unit vs. reduction unit).
    """

    def __init__(self, ops: nn.ModuleList):
        super().__init__()
        self.ops    = ops
        self._alpha = nn.Parameter(torch.zeros(len(ops)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(self._alpha, dim=0)
        return sum(w * op(x) for w, op in zip(weights, self.ops))

    @property
    def alpha(self) -> torch.Tensor:
        return self._alpha


class NASTransformerBlock(nn.Module):
    """
    NAS-searchable Transformer block.  Implements a MixedOp between
    'common unit' (standard MSA+FFN) and 'reduction unit' (cross-modal
    compression / dimension reduction) as described in Sec. IV-C.

    For now the reduction unit is a mean-pooling projection (lightweight
    stand-in; can be swapped for cross-attention or gating in Phase 4).
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        common    = TransformerBlock(d_model, n_heads, d_ff, dropout)
        reduction = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.mixed = MixedOp(nn.ModuleList([common, reduction]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mixed(x)

    @property
    def arch_params(self) -> List[torch.Tensor]:
        return [self.mixed.alpha]


class NASMultimodalEncoder(MultimodalMetaEncoder):
    """
    NAS-searchable version of MultimodalMetaEncoder.

    Replaces fixed TransformerBlocks in the backbone with NASTransformerBlocks
    containing learnable architecture parameters α_{i,j} (Equ. 31).

    Usage in bilevel NAS training:
        encoder.set_arch_training(True)   # train α on validation set
        encoder.set_arch_training(False)  # train weights on training set
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        d_model = self.d_model
        n_heads = kwargs.get("n_heads", 4)
        d_ff    = kwargs.get("d_ff", 4 * d_model)
        dropout = kwargs.get("dropout", 0.1)
        n_layers = kwargs.get("n_layers", 2)

        # Replace standard blocks with NAS blocks
        self.encoder.blocks = nn.ModuleList([
            NASTransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

    def arch_parameters(self) -> List[torch.Tensor]:
        """Returns all architecture parameters α for bilevel optimisation."""
        params = []
        for block in self.encoder.blocks:
            if hasattr(block, "arch_params"):
                params.extend(block.arch_params)
        return params

    def weight_parameters(self) -> List[torch.Tensor]:
        """Returns non-architecture parameters for weight training."""
        arch_ids = {id(p) for p in self.arch_parameters()}
        return [p for p in self.parameters() if id(p) not in arch_ids]

    def set_arch_training(self, mode: bool):
        """Toggle which parameter group is active for gradient updates."""
        for p in self.arch_parameters():
            p.requires_grad_(mode)
        for p in self.weight_parameters():
            p.requires_grad_(not mode)

    def discretize(self) -> MultimodalMetaEncoder:
        """
        Convert soft architecture to discrete by keeping the highest-α op
        per block.  Returns a standard MultimodalMetaEncoder.
        Used at the end of NAS search (Stage 3 of Algorithm 2).
        """
        # Count common vs reduction wins across blocks
        common_wins = sum(
            1 for blk in self.encoder.blocks
            if blk.mixed.alpha[0] >= blk.mixed.alpha[1]
        )
        n_layers = len(self.encoder.blocks)
        print(f"[NAS] Discretized: {common_wins}/{n_layers} blocks → common unit")

        discrete = MultimodalMetaEncoder(
            modal_dims = self.modal_dims,
            d_model    = self.d_model,
            n_heads    = len(self.encoder.blocks[0].mixed.ops[0].attn.heads
                            if hasattr(self.encoder.blocks[0].mixed.ops[0], 'attn')
                            else [None]*4),  # fallback
            n_layers   = n_layers,
        )
        return discrete


# ─────────────────────────────────────────────────────────────────────
#  Sanity check
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(0)

    MODAL_DIMS = [128, 32, 32, 64, 64]   # paper Sec. V
    MODAL_NAMES = ["visual", "radar", "infrared", "audio", "environmental"]
    B = 4

    # ── Standard encoder ──────────────────────────────────────────
    print("=== MultimodalMetaEncoder ===")
    enc = MultimodalMetaEncoder(
        modal_dims = MODAL_DIMS,
        d_model    = 128,
        n_heads    = 4,
        n_layers   = 2,
        dropout    = 0.1,
    )
    modalities = {name: torch.randn(B, d) for name, d in zip(MODAL_NAMES, MODAL_DIMS)}
    task_emb, modal_embs = enc(modalities)
    print(f"  task_emb shape   : {tuple(task_emb.shape)}")
    print(f"  modal_embs[0]    : {tuple(modal_embs[0].shape)}  (× {len(modal_embs)})")
    print(f"  trainable params : {enc.count_params():,}")
    print(f"  estimated FLOPs  : {enc.estimated_flops():.0f}")

    # ── NAS encoder ───────────────────────────────────────────────
    print("\n=== NASMultimodalEncoder ===")
    nas_enc = NASMultimodalEncoder(
        modal_dims = MODAL_DIMS,
        d_model    = 128,
        n_heads    = 4,
        n_layers   = 2,
    )
    task_emb2, modal_embs2 = nas_enc(modalities)
    print(f"  task_emb shape   : {tuple(task_emb2.shape)}")
    print(f"  arch params      : {len(nas_enc.arch_parameters())} tensors")
    print(f"  weight params    : {sum(p.numel() for p in nas_enc.weight_parameters()):,}")

    # ── Gradient flow check ───────────────────────────────────────
    print("\n=== Gradient flow ===")
    loss = sum(e.sum() for e in modal_embs) + task_emb.sum()
    loss.backward()
    grads_ok = all(
        p.grad is not None
        for p in enc.parameters() if p.requires_grad
    )
    print(f"  All gradients present: {grads_ok}")

    # ── Interface to VMIMO ────────────────────────────────────────
    print("\n=== VMIMO interface check ===")
    print(f"  modal_embs type  : list of {len(modal_embs)} tensors")
    print(f"  each shape       : {tuple(modal_embs[0].shape)}")
    print("  → Ready for VMIMOController(modal_embs, V_sig, p_joint, p_succ, task_emb)")
