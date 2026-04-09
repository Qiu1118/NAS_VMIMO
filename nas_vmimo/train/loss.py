"""
End-to-end loss for NAS-VMIMO (Phase 3).

Implements Equ. 17 and 23 from the paper:

    L_total = L_recv + Σ_i  λ_i · ReLU(c_i)²          (Equ. 23)

    L_recv  = 1 - cosine(E(T̂), E(T))                   (Equ. 17)

Constraints (Equ. 18):
    c_PG       = ||G U_r Σ_r A_c − I||_F − ε_PG
    c_demod(m) = ||ê(m) − e(m)||² − ε_m

Gradient derivations are in Sec. IV-B (Equ. 24–30).
We rely on PyTorch autograd for backprop through the full chain.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────
#  Semantic reconstruction loss  L_recv  (Equ. 17)
# ─────────────────────────────────────────────────────────────────────
def semantic_loss(
    task_emb_pred: torch.Tensor,    # (B, d) – encoder output on received signal
    task_emb_true: torch.Tensor,    # (B, d) – encoder output on original signal
) -> torch.Tensor:
    """
    L_recv = 1 - cosine_similarity(E(T̂), E(T))   averaged over batch.
    """
    cos = F.cosine_similarity(task_emb_pred, task_emb_true, dim=-1)  # (B,)
    return (1.0 - cos).mean()


# ─────────────────────────────────────────────────────────────────────
#  Physical-layer constraints  (Equ. 18)
# ─────────────────────────────────────────────────────────────────────
def constraint_pg(
    G:       torch.Tensor,    # (B, r, Nr)   receiver matrix
    U_r:     torch.Tensor,    # (B, Nr, r)   left singular vectors
    Sigma_r: torch.Tensor,    # (B, r)       singular values
    Ac:      torch.Tensor,    # (r, d_all)   compression matrix (real)
    eps_pg:  float = 0.1,
) -> torch.Tensor:
    """
    c_PG = ||G U_r Σ_r A_c − I_r||_F − ε_PG

    Measures how far the cascade G U_r Σ_r A_c is from identity.
    When c_PG ≤ 0, the constraint is satisfied.
    """
    # G (B,r,Nr) @ U_r (B,Nr,r) → (B,r,r)
    GU = torch.bmm(G, U_r)
    # × diag(Sigma_r)
    Sigma_diag = torch.diag_embed(Sigma_r)           # (B,r,r)
    GUSigma    = torch.bmm(GU, Sigma_diag)           # (B,r,r)
    # × Ac  (broadcast over batch)
    r, d_all   = Ac.shape
    Ac_b       = Ac.unsqueeze(0).expand(GUSigma.shape[0], -1, -1)  # (B,r,d_all)
    GUSigmaAc  = torch.bmm(GUSigma, Ac_b)           # (B,r,d_all)
    # Compare sub-block to I_r
    I_r  = torch.eye(r, device=G.device).unsqueeze(0)
    diff = GUSigmaAc[..., :r] - I_r                 # (B,r,r)
    frob = diff.norm(dim=(-2, -1))                   # (B,)
    return frob.mean() - eps_pg                      # scalar


def constraint_demod(
    e_hat: torch.Tensor,   # (B, d_model)  reconstructed embedding
    e_ref: torch.Tensor,   # (B, d_model)  original embedding
    eps_m: float = 0.1,
) -> torch.Tensor:
    """
    c_demod^(m) = ||ê(m) − e(m)||² − ε_m   (Equ. 18)
    Returns mean over batch.
    """
    mse = ((e_hat - e_ref) ** 2).sum(dim=-1).mean()   # scalar
    return mse - eps_m


# ─────────────────────────────────────────────────────────────────────
#  Classification head loss  (task accuracy during E2E training)
# ─────────────────────────────────────────────────────────────────────
def classification_loss(
    logits: torch.Tensor,   # (B, n_classes)
    labels: torch.Tensor,   # (B,)
) -> torch.Tensor:
    return F.cross_entropy(logits, labels)


# ─────────────────────────────────────────────────────────────────────
#  Total loss  L_total  (Equ. 23)
# ─────────────────────────────────────────────────────────────────────
class VMIMOLoss(nn.Module):
    """
    Aggregated differentiable loss for end-to-end VMIMO training.

    L_total = L_recv
            + λ_cls   · L_cls             (task accuracy)
            + λ_PG    · ReLU(c_PG)²       (precoder-decoder consistency)
            + Σ_m λ_m · ReLU(c_demod_m)²  (per-modality demodulation)

    Args:
        lambda_cls   : weight for classification cross-entropy
        lambda_pg    : penalty weight for c_PG constraint
        lambda_demod : penalty weight for c_demod constraints (per modality)
        eps_pg       : ε_PG tolerance
        eps_demod    : ε_m tolerance per modality
    """

    def __init__(
        self,
        lambda_cls:   float = 1.0,
        lambda_pg:    float = 0.5,
        lambda_demod: float = 0.5,
        eps_pg:       float = 0.1,
        eps_demod:    float = 0.1,
    ):
        super().__init__()
        self.lambda_cls   = lambda_cls
        self.lambda_pg    = lambda_pg
        self.lambda_demod = lambda_demod
        self.eps_pg       = eps_pg
        self.eps_demod    = eps_demod

    def forward(
        self,
        # Semantic reconstruction
        task_emb_pred: torch.Tensor,         # (B, d)
        task_emb_true: torch.Tensor,         # (B, d)
        # Classification
        logits:        torch.Tensor,         # (B, n_classes)
        labels:        torch.Tensor,         # (B,)
        # Physical-layer consistency (optional — only for Case 3 Ac)
        modal_embs_pred: Optional[List[torch.Tensor]] = None,  # list (B, d)
        modal_embs_true: Optional[List[torch.Tensor]] = None,  # list (B, d)
        G:               Optional[torch.Tensor]       = None,
        U_r:             Optional[torch.Tensor]       = None,
        Sigma_r:         Optional[torch.Tensor]       = None,
        Ac:              Optional[torch.Tensor]       = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Returns:
            total_loss  : differentiable scalar
            breakdown   : dict of named loss components for logging
        """
        # ── Semantic loss ──────────────────────────────────────────
        l_recv = semantic_loss(task_emb_pred, task_emb_true)

        # ── Classification loss ────────────────────────────────────
        l_cls  = classification_loss(logits, labels)

        total  = l_recv + self.lambda_cls * l_cls
        breakdown = {
            "L_recv": l_recv.item(),
            "L_cls":  l_cls.item(),
        }

        # ── PG constraint penalty (Case 3 only) ───────────────────
        if all(x is not None for x in [G, U_r, Sigma_r, Ac]):
            c_pg   = constraint_pg(G, U_r, Sigma_r, Ac, self.eps_pg)
            pen_pg = self.lambda_pg * F.relu(c_pg) ** 2
            total  = total + pen_pg
            breakdown["c_PG"]  = c_pg.item()
            breakdown["pen_PG"] = pen_pg.item()

        # ── Demodulation constraint penalties ─────────────────────
        if modal_embs_pred is not None and modal_embs_true is not None:
            pen_demod_total = 0.0
            for m, (e_hat, e_ref) in enumerate(zip(modal_embs_pred, modal_embs_true)):
                c_m = constraint_demod(e_hat, e_ref, self.eps_demod)
                pen = self.lambda_demod * F.relu(c_m) ** 2
                pen_demod_total = pen_demod_total + pen
                breakdown[f"c_demod_{m}"] = c_m.item()
            total = total + pen_demod_total
            breakdown["pen_demod_total"] = float(
                sum(v for k, v in breakdown.items() if k.startswith("c_demod"))
            )

        breakdown["L_total"] = total.item()
        return total, breakdown


# ─────────────────────────────────────────────────────────────────────
#  Task classification head  (plugged on top of task_emb)
# ─────────────────────────────────────────────────────────────────────
class TaskHead(nn.Module):
    """
    Lightweight MLP head for the downstream multimodal classification task.
    Input: task_emb (CLS token output, B × d_model)
    Output: logits  (B × n_classes)
    """

    def __init__(self, d_model: int, n_classes: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )

    def forward(self, task_emb: torch.Tensor) -> torch.Tensor:
        return self.net(task_emb)


# ─────────────────────────────────────────────────────────────────────
#  Sanity check
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import torch
    torch.manual_seed(0)

    B, d, M, r, Nr = 4, 128, 5, 4, 4

    # Dummy encoder outputs
    task_pred = torch.randn(B, d, requires_grad=True)
    task_true = torch.randn(B, d)
    modal_pred = [torch.randn(B, d, requires_grad=True) for _ in range(M)]
    modal_true = [torch.randn(B, d) for _ in range(M)]

    # Dummy classification
    head   = TaskHead(d, n_classes=4)
    logits = head(task_pred)
    labels = torch.randint(0, 4, (B,))

    # Dummy channel matrices for PG constraint
    G       = torch.randn(B, r, Nr)
    U_r     = torch.randn(B, Nr, r)
    Sigma_r = torch.rand(B, r) + 0.1
    d_all   = sum([128, 32, 32, 64, 64])
    Ac      = torch.randn(r, d_all)

    loss_fn = VMIMOLoss(lambda_cls=1.0, lambda_pg=0.5, lambda_demod=0.5)

    total, breakdown = loss_fn(
        task_emb_pred    = task_pred,
        task_emb_true    = task_true,
        logits           = logits,
        labels           = labels,
        modal_embs_pred  = modal_pred,
        modal_embs_true  = modal_true,
        G=G, U_r=U_r, Sigma_r=Sigma_r, Ac=Ac,
    )

    print("=== VMIMOLoss sanity check ===")
    for k, v in breakdown.items():
        print(f"  {k:20s}: {v:.4f}")
    print(f"  total (from breakdown): {breakdown['L_total']:.4f}")

    # Backprop check
    total.backward()
    print(f"  task_pred grad norm  : {task_pred.grad.norm():.4f}")
    print(f"  modal_pred[0] grad   : {modal_pred[0].grad.norm():.4f}")
    print("  Gradient flow: OK")
