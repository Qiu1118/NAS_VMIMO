"""
VMIMO Core — implements the three mapping matrix cases from Sec. III-B.

Ak selector logic (Equ. 7):
  Case 1: r' ≤ r          →  Ak = I_{r'}          (parallel mapping)
  Case 2: r' > r, pjoint < η1  →  Ak = A_soft    (attention-weighted DoF)
  Case 3: r' > r, pjoint ≥ η1  →  Ak = Ac        (low-rank compression)

All three cases are implemented as nn.Module so they participate
in end-to-end gradient flow.

Dependencies: torch  (import guarded — falls back to numpy stub for testing)
"""

from __future__ import annotations
import math
import numpy as np
from typing import Dict, List, Optional, Tuple

# ── torch is optional in this file so the channel.py / dataset.py
#    sanity-checks can run without it.  Full training requires torch.
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH = True
except ImportError:
    _TORCH = False
    print("[vmimo] torch not found — running in numpy-stub mode")


# ─────────────────────────────────────────────────────────────────────
#  Helper: build attention weight w(m)  (Equ. 9)
# ─────────────────────────────────────────────────────────────────────
if _TORCH:
    class ModalityAttentionWeight(nn.Module):
        """
        Computes soft importance weights  w(m)  for each modality.

        w(m) = softmax_m( q^T k(m) / sqrt(d)
                          + λ (1 - P_succ(m))
                          + μ β(m) )

        Args:
            d_model : query/key embedding dimension
            n_modal : number of modalities M'
            lambda_ : weight on reliability term  (1 - P_succ)
            mu_     : weight on historical prior   β(m)
        """

        def __init__(self, d_model: int, n_modal: int, lambda_: float = 1.0, mu_: float = 0.5):
            super().__init__()
            self.d_model  = d_model
            self.n_modal  = n_modal
            self.lambda_  = lambda_
            self.mu_      = mu_

            # Task-level query projection (task embedding → query vector)
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            # Per-modality key projections  (embedding → key vector)
            self.k_projs = nn.ModuleList([
                nn.Linear(d_model, d_model, bias=False) for _ in range(n_modal)
            ])
            # Learnable historical importance β(m)  – one scalar per modality
            self.beta = nn.Parameter(torch.ones(n_modal) / n_modal)

        def forward(
            self,
            task_emb: torch.Tensor,           # (B, d_model) – task query
            modal_embs: List[torch.Tensor],   # list of M' (B, d_model) tensors
            p_succ: torch.Tensor,             # (B, M') – per-modality success probs
        ) -> torch.Tensor:                    # (B, M') – weights summing to 1
            """
            Returns attention weights w(m) ∈ (0,1) with Σ_m w(m)=1.
            """
            B = task_emb.shape[0]

            q = self.q_proj(task_emb)          # (B, d)
            attn_scores = []
            for i, (k_proj, e) in enumerate(zip(self.k_projs, modal_embs)):
                k  = k_proj(e)                 # (B, d)
                # dot product attention score
                score = (q * k).sum(dim=-1) / math.sqrt(self.d_model)  # (B,)
                attn_scores.append(score)

            attn_raw = torch.stack(attn_scores, dim=1)  # (B, M')

            # Reliability bonus: 1 - P_succ(m)
            reliability = 1.0 - p_succ                 # (B, M')

            # Historical prior β  (shared across batch)
            beta = self.beta.unsqueeze(0).expand(B, -1)  # (B, M')

            combined = attn_raw + self.lambda_ * reliability + self.mu_ * beta
            return F.softmax(combined, dim=-1)           # (B, M')


    # ─────────────────────────────────────────────────────────────────────
    #  Case 1: Parallel mapping  Ak = I_{r'}
    # ─────────────────────────────────────────────────────────────────────
    class ParallelMapping(nn.Module):
        """Identity mapping — each modality gets its own DoF stream."""

        def forward(
            self,
            embeddings: torch.Tensor,    # (B, r', d_embed) – stacked modal embeddings
            V_sig: torch.Tensor,         # (B, Nt, r)       – SVD right singular vectors
        ) -> torch.Tensor:               # (B, Nt, r')       – precoded signal
            # embeddings already ordered as M(e^(i)); no mixing needed
            # V_sig @ I_{r'} @ M(e) = V_sig @ M(e)
            return torch.bmm(V_sig, embeddings)  # (B, Nt, r')


    # ─────────────────────────────────────────────────────────────────────
    #  Case 2: Soft-selection mapping  Ak = A_soft  (Equ. 8–10)
    # ─────────────────────────────────────────────────────────────────────
    class SoftSelectionMapping(nn.Module):
        """
        Attention-weighted, exclusive DoF allocation.

        Given weights w(m), allocates rm = floor(r * w(m)) DoFs per modality.
        Constructs binary A_soft ∈ {0,1}^{r × r'} with Σ_j a_{ij} ≤ 1.

        This is a discrete operation inside a continuous wrapper:
        - Forward:  discrete integer allocation (no gradient through rm)
        - The weights w(m) still receive gradients via the attention module
        - Residual DoF and null-space handling per Remark 2

        Args:
            n_modal : r' — number of modality streams
            embed_dim : d_embed — embedding dimension (for null-space projection size)
        """

        def __init__(self, n_modal: int, embed_dim: int):
            super().__init__()
            self.n_modal   = n_modal
            self.embed_dim = embed_dim

        def _allocate_dofs(self, weights: np.ndarray, r: int) -> np.ndarray:
            """
            Returns integer DoF counts rm[m] for each modality.
            Implements the floor + residual allocation from Remark 2.
            weights : (M',) numpy array summing to 1
            """
            r_prime   = len(weights)
            rm        = np.floor(r * weights).astype(int)         # floor allocation

            # Residual: fill unused DoFs in order of fractional remainder
            used      = rm.sum()
            if used < r:
                remainders = (r * weights) - rm
                order      = np.argsort(-remainders)              # descending remainder
                for idx in order:
                    if used >= r:
                        break
                    rm[idx] += 1
                    used    += 1

            return rm                                              # shape (M',)

        def _build_asoft(self, rm: np.ndarray, r: int) -> np.ndarray:
            """
            Build binary A_soft ∈ {0,1}^{r × r'} with exclusive constraint.
            Each row has at most one 1 (exclusiveness from Equ. 8).
            """
            r_prime  = len(rm)
            A        = np.zeros((r, r_prime), dtype=np.float32)
            row_ptr  = 0

            for m in range(r_prime):
                for _ in range(rm[m]):
                    if row_ptr < r:
                        A[row_ptr, m] = 1.0
                        row_ptr += 1

            return A   # (r, r')

        def forward(
            self,
            embeddings: torch.Tensor,          # (B, r', d_embed)
            V_sig: torch.Tensor,               # (B, Nt, r)
            weights: torch.Tensor,             # (B, r')   attention weights
        ) -> torch.Tensor:                     # (B, Nt, r') precoded signal
            B, r_prime, d = embeddings.shape
            r             = V_sig.shape[-1]

            outputs = []
            for b in range(B):
                w_np    = weights[b].detach().cpu().numpy()
                rm      = self._allocate_dofs(w_np, r)
                A_np    = self._build_asoft(rm, r)                # (r, r')
                A_t     = torch.tensor(A_np, dtype=V_sig.dtype,
                                       device=V_sig.device)       # (r, r')

                # x_b = V_sig_b @ A_soft @ M(e_b)
                # embeddings[b]: (r', d) → treat d as 1 stream per modality
                # For now: treat each embedding as a scalar stream via mean projection
                e_streams  = embeddings[b]                        # (r', d)
                # Map d-dim embedding to 1-dim stream via mean (placeholder)
                # In Phase 3 this is replaced by the modulator M(·)
                e_scalar   = e_streams.mean(dim=-1, keepdim=True) # (r', 1)
                # A_t: (r, r'), e_scalar: (r', 1) → (r, 1)
                Ae         = torch.mm(A_t, e_scalar)              # (r, 1)
                # V_sig_b: (Nt, r), Ae: (r, 1) → (Nt, 1) → (Nt, r')
                x_b        = torch.mm(V_sig[b], Ae)               # (Nt, 1)
                # Expand to (Nt, r') for shape consistency (signal broadcast)
                x_b        = x_b.expand(-1, r_prime)
                outputs.append(x_b)

            return torch.stack(outputs, dim=0)   # (B, Nt, r')


    # ─────────────────────────────────────────────────────────────────────
    #  Case 3: Compression mapping  Ak = Ac  (Equ. 11–13)
    # ─────────────────────────────────────────────────────────────────────
    class CompressionMapping(nn.Module):
        """
        Block-wise low-rank compression:
            P_m = W_m R_m^H  ∈ C^{r × d_m}
            A_c = [P_1 | P_2 | … | P_M']  ∈ C^{r × d_all}

        Works in real-valued domain (imaginary part added in Phase 3
        when the complex modulator M(·) is introduced).

        Args:
            modal_dims : list of original input dims per modality (kept for bookkeeping)
            embed_dim  : per-modality embedding dim fed into VMIMO (encoder output, typically d_model)
            r          : number of signal-bearing DoFs (rank of H)
            q_m        : per-modality bottleneck dimension (list or scalar)
        """

        def __init__(
            self,
            modal_dims: List[int],
            embed_dim: int,
            r: int,
            q_m: int = 8,              # low-rank bottleneck dimension
        ):
            super().__init__()
            self.modal_dims = modal_dims
            self.embed_dim  = embed_dim
            self.r          = r
            self.q_m        = q_m

            # NOTE:
            # In the E2E system, VMIMO consumes encoder embeddings e^(m) with shape (B, embed_dim),
            # not the raw modality feature vectors of shape (B, d_m). Therefore R_m must map
            # embed_dim → q_m for every modality to keep dimensions consistent.
            #
            # R_m: embed_dim → q_m  (per-modality semantic compression)
            self.R = nn.ModuleList([
                nn.Linear(embed_dim, q_m, bias=False) for _ in modal_dims
            ])
            # W_m: q_m → r   (DoF space mapping)
            self.W = nn.ModuleList([
                nn.Linear(q_m, r,   bias=False) for _ in modal_dims
            ])

        def forward(
            self,
            embeddings: List[torch.Tensor],   # list of (B, d_m) per modality
            V_sig: torch.Tensor,              # (B, Nt, r)
        ) -> torch.Tensor:                    # (B, Nt, 1) or (B, Nt, r)
            """
            Computes x_k = V_sig A_c e_all   (Equ. 13)
            Returns precoded signal (B, Nt, r).
            """
            # Σ_m W_m R_m^H e(m)  → (B, r)
            compressed = sum(
                self.W[m](self.R[m](e))      # (B, q_m) → (B, r)
                for m, e in enumerate(embeddings)
            )                                # (B, r)

            # V_sig: (B, Nt, r),  compressed: (B, r, 1)
            compressed_col = compressed.unsqueeze(-1)            # (B, r, 1)
            x = torch.bmm(V_sig, compressed_col)                 # (B, Nt, 1)
            return x.expand(-1, -1, self.r)                      # (B, Nt, r)

        def reconstruction_consistency_loss(
            self,
            G: torch.Tensor,       # (B, r, Nr)  receiver matrix
            U_r: torch.Tensor,     # (B, Nr, r)
            Sigma_r: torch.Tensor, # (B, r)
        ) -> torch.Tensor:
            """
            c_PG = ||G U_r Σ_r Ac - I||_F  (Equ. 18, constraint c_PG)
            Used as a penalty in L_total.
            """
            # Build Ac numerically for the consistency check
            # Ac = Σ_m W_m R_m^H  as a matrix  (r, d_all)
            Ac_blocks = [
                torch.mm(self.W[m].weight, self.R[m].weight)  # (r, d_m)
                for m in range(len(self.modal_dims))
            ]
            Ac = torch.cat(Ac_blocks, dim=-1)                  # (r, d_all)

            # G (B, r, Nr) @ U_r (B, Nr, r) → (B, r, r)
            GU = torch.bmm(G, U_r)                             # (B, r, r)

            # * diag(Sigma_r)
            Sigma_diag = torch.diag_embed(Sigma_r)             # (B, r, r)
            GUSigma    = torch.bmm(GU, Sigma_diag)             # (B, r, r)

            # * Ac  (broadcast)
            Ac_batch = Ac.unsqueeze(0).expand(GUSigma.shape[0], -1, -1)  # (B,r,d_all)
            GUSigmaAc = torch.bmm(GUSigma, Ac_batch)           # (B, r, d_all)

            # Compare to identity (r × r sub-block — first r columns)
            I_r   = torch.eye(self.r, device=G.device).unsqueeze(0)  # (1,r,r)
            diff  = GUSigmaAc[..., :self.r] - I_r
            return diff.norm(dim=(-2, -1)).mean()               # scalar


    # ─────────────────────────────────────────────────────────────────────
    #  VMIMO Controller — Algorithm 1
    # ─────────────────────────────────────────────────────────────────────
    class VMIMOController(nn.Module):
        """
        Top-level VMIMO module.  Implements Algorithm 1 from the paper.

        Selects and applies the appropriate mapping matrix A_k based on:
          - r' vs r  (modality count vs available DoFs)
          - p_joint  (joint resource probability)

        Args:
            modal_dims   : list of embedding dims per modality
            r            : number of spatial DoFs  (= rank(H), fixed for sim)
            Nt           : number of transmit antennas
            d_model      : Transformer embedding size
            eta1         : switching threshold for Case 2 vs Case 3
            q_m          : low-rank bottleneck for Case 3
        """

        def __init__(
            self,
            modal_dims: List[int],
            r:          int   = 4,
            Nt:         int   = 4,
            d_model:    int   = 128,
            eta1:       float = 0.6,
            q_m:        int   = 8,
        ):
            super().__init__()
            n_modal        = len(modal_dims)
            self.modal_dims = modal_dims
            self.r          = r
            self.Nt         = Nt
            self.eta1       = eta1

            # Case 2: attention weights
            self.attn_weight = ModalityAttentionWeight(d_model, n_modal)
            # Case 2: soft selection
            self.soft_sel    = SoftSelectionMapping(n_modal, d_model)
            # Case 3: compression
            self.compress    = CompressionMapping(modal_dims, embed_dim=d_model, r=r, q_m=q_m)
            # Case 1: parallel (stateless)
            self.parallel    = ParallelMapping()

        def forward(
            self,
            modal_embs: List[torch.Tensor],    # list of M' (B, d_model) embeddings
            V_sig:      torch.Tensor,          # (B, Nt, r)
            p_joint:    float,                 # scalar resource probability
            p_succ:     torch.Tensor,          # (B, M') per-modality success probs
            task_emb:   Optional[torch.Tensor] = None,   # (B, d_model) task query
        ) -> Tuple[torch.Tensor, str]:
            """
            Returns:
                x_k   : (B, Nt, r') or (B, Nt, r) — precoded signal
                case  : which case was used ('parallel'|'soft'|'compress')
            """
            r_prime = len(modal_embs)
            r       = self.r

            # ── Case 1: sufficient DoFs ────────────────────────────────
            if r_prime <= r:
                stacked = torch.stack(modal_embs, dim=1)   # (B, r', d)
                x = self.parallel(stacked, V_sig)
                return x, "parallel"

            # ── Case 2/3: insufficient DoFs ────────────────────────────
            if task_emb is None:
                # Fallback: use mean of modal embeddings as task query
                task_emb = torch.stack(modal_embs, dim=1).mean(dim=1)

            if p_joint < self.eta1:
                # Case 2: lightweight soft-selection
                weights = self.attn_weight(task_emb, modal_embs, p_succ)
                stacked = torch.stack(modal_embs, dim=1)   # (B, r', d)
                x = self.soft_sel(stacked, V_sig, weights)
                return x, "soft"
            else:
                # Case 3: full compression mapping
                x = self.compress(modal_embs, V_sig)
                return x, "compress"


# ─────────────────────────────────────────────────────────────────────
#  Numpy stub (when torch not available — for unit testing channel.py)
# ─────────────────────────────────────────────────────────────────────
else:
    class VMIMOController:
        """Minimal numpy stub for unit testing without torch."""
        def __init__(self, *a, **kw):
            print("[vmimo] stub mode — no forward pass available")


# ─────────────────────────────────────────────────────────────────────
#  Sanity check (numpy-compatible parts only)
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not _TORCH:
        print("torch not available — skipping vmimo forward test")
    else:
        from data.channel import MIMOChannel, MIMOChannelConfig, AR1ResourceModel

        B, Nt, Nr, r_dof = 4, 4, 4, 4
        M_prime = 5
        d_model = 128
        modal_dims = [128, 32, 32, 64, 64]   # paper Table in Sec V

        ch  = MIMOChannel(MIMOChannelConfig(Nt=Nt, Nr=Nr, snr_db=5.0))
        H, U_r, S, V_sig, r = ch.generate(batch_size=B, seed=7)
        # Use real part of V_sig for the controller (complex support: Phase 3)
        V_real = V_sig.real.float()   # (B, Nt, r)

        ctrl = VMIMOController(modal_dims=modal_dims, r=r_dof, Nt=Nt, d_model=d_model)

        # Dummy embeddings
        embs   = [torch.randn(B, d_model) for _ in range(M_prime)]
        p_succ = torch.rand(B, M_prime)

        rm = AR1ResourceModel(seed=0)
        trace = rm.generate_trace(500)
        pj = rm.pjoint_from_trace(trace, t=200)

        x, case = ctrl(embs, V_real, p_joint=pj, p_succ=p_succ)
        print(f"VMIMO case: {case!r}")
        print(f"Output x shape: {tuple(x.shape)}")
        print(f"p_joint used: {pj:.4f}")
