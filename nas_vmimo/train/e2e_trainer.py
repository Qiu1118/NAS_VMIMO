"""
End-to-end trainer for NAS-VMIMO.

Wires together the full forward pass:

  dataset ──► encoder ──► VMIMO ──► channel ──► decoder ──► loss
               (true emb)    (precoder)  (Hx+n)   (recon emb)

Training loop implements the paper's setup (Sec. V):
  • Adam, lr=5e-5, batch=64, 30 epochs
  • Rayleigh fading MIMO, Nt=Nr=4
  • 5 modalities, 4 classes
  • pjoint from AR(1) trace
  • VMIMO switches between Case 1/2/3 per batch

Also handles:
  • Per-SNR evaluation (Fig 4/5 in the paper)
  • Metric logging (loss breakdown, accuracy, VMIMO case distribution)
  • Checkpoint save/load
"""

from __future__ import annotations
import os
import time
import math
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Project imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset  import (MultimodalDataset, collate_fn,
                            MODALITY_CONFIGS, N_CLASSES)
from data.channel  import MIMOChannel, MIMOChannelConfig, AR1ResourceModel
from models.encoder import MultimodalMetaEncoder
from models.decoder import SemanticDecoder
from models.vmimo   import VMIMOController
from train.loss     import VMIMOLoss, TaskHead


# ─────────────────────────────────────────────────────────────────────
#  Helper: real-valued projection of complex stream
# ─────────────────────────────────────────────────────────────────────
def complex_to_real_stream(s: torch.Tensor) -> torch.Tensor:
    """
    Convert complex equalized stream (B, r) → real (B, 2r) by stacking
    real and imaginary parts.  The decoder works in real domain.
    """
    if torch.is_complex(s):
        return torch.cat([s.real, s.imag], dim=-1)
    return s


def embed_to_complex_signal(
    emb: torch.Tensor,        # (B, d_model) per-modality embedding
    r:   int,
) -> torch.Tensor:
    """
    Project embedding to a complex symbol stream of length r.
    This is a simplified modulator M(e):
        M(e) = Linear(e)  →  (B, r) complex
    We just use a linear projection on the real part for now.
    """
    # Use the first r dims as real, next r dims as imag (or zero-pad)
    d = emb.shape[-1]
    if d >= 2 * r:
        real_part = emb[..., :r]
        imag_part = emb[..., r:2*r]
    else:
        # zero-pad imag
        real_part = emb[..., :min(r, d)]
        imag_part = torch.zeros_like(real_part)
        if d < r:
            pad = torch.zeros(emb.shape[0], r - d, device=emb.device)
            real_part = torch.cat([real_part, pad], dim=-1)

    return torch.complex(real_part.float(), imag_part.float())


# ─────────────────────────────────────────────────────────────────────
#  VMIMOSystem:  encapsulates the full forward pass
# ─────────────────────────────────────────────────────────────────────
class VMIMOSystem(nn.Module):
    """
    Full NAS-VMIMO system as a single nn.Module.

    Components
    ----------
    encoder    : MultimodalMetaEncoder   (Data2Seq + shared ViT backbone)
    vmimo      : VMIMOController         (Ak selector)
    decoder    : SemanticDecoder         (stream → reconstructed embeddings)
    task_head  : TaskHead                (task_emb_true → logits)

    The channel is NOT an nn.Module (no parameters), it lives outside and
    its matrices H, U, S, V are passed in at each forward call.
    """

    def __init__(
        self,
        modal_dims:  List[int],
        d_model:     int   = 128,
        n_heads:     int   = 4,
        n_layers:    int   = 2,
        r:           int   = 4,     # spatial DoFs = rank(H)
        Nt:          int   = 4,
        n_classes:   int   = 4,
        eta1:        float = 0.6,
        q_m:         int   = 8,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.r = r

        self.encoder = MultimodalMetaEncoder(
            modal_dims=modal_dims, d_model=d_model,
            n_heads=n_heads, n_layers=n_layers, dropout=dropout,
        )
        self.vmimo = VMIMOController(
            modal_dims=modal_dims, r=r, Nt=Nt,
            d_model=d_model, eta1=eta1, q_m=q_m,
        )
        # Decoder input dimension: 2r (real + imag from ZF receiver)
        self.decoder = SemanticDecoder(
            r=2*r, d_model=d_model, n_modal=len(modal_dims),
            n_heads=n_heads, n_layers=n_layers, dropout=dropout,
        )
        self.task_head = TaskHead(d_model, n_classes, dropout)

    def forward(
        self,
        modalities:  Dict[str, torch.Tensor],   # {name: (B, d_m)}
        V_sig:       torch.Tensor,              # (B, Nt, r)  real-valued SVD
        U_r:         torch.Tensor,              # (B, Nr, r)
        Sigma_r:     torch.Tensor,              # (B, r)
        H:           torch.Tensor,              # (B, Nr, Nt) complex
        p_joint:     float,
        snr_db:      float = 5.0,
    ) -> Dict:
        """
        Full forward pass.  Returns a dict with everything needed by the loss.
        """
        B = next(iter(modalities.values())).shape[0]
        device = next(iter(modalities.values())).device

        # ── 1. Encode original modalities ─────────────────────────
        task_emb_true, modal_embs_true = self.encoder(modalities)

        # ── 2. VMIMO: build precoded signal ───────────────────────
        p_succ_dummy = torch.ones(B, len(modal_embs_true),
                                  device=device) * 0.7  # see §IV-A
        x_k, vmimo_case = self.vmimo(
            modal_embs=modal_embs_true,
            V_sig=V_sig,
            p_joint=p_joint,
            p_succ=p_succ_dummy,
            task_emb=task_emb_true,
        )
        # x_k: (B, Nt, r') — take mean over stream dimension for transmit
        x_tx = x_k.mean(dim=-1)   # (B, Nt) real

        # ── 3. Channel: transmit through H ────────────────────────
        # Build complex transmit vector from real embedding
        x_complex = torch.complex(x_tx.float(),
                                   torch.zeros_like(x_tx))  # (B, Nt)

        # y = H x + n
        snr_lin   = 10 ** (snr_db / 10.0)
        sig_power = x_complex.abs().pow(2).mean() + 1e-9
        noise_std = (sig_power / snr_lin).sqrt()
        x_col     = x_complex.unsqueeze(-1)             # (B, Nt, 1)
        y         = torch.bmm(H.to(x_col.dtype), x_col).squeeze(-1)  # (B, Nr)
        n_real    = torch.randn_like(y.real) * noise_std / math.sqrt(2)
        n_imag    = torch.randn_like(y.imag) * noise_std / math.sqrt(2)
        y         = y + torch.complex(n_real, n_imag)

        # ── 4. ZF Equalization: ŝ = diag(1/σ) U_r^H y ────────────
        U_c       = U_r.to(y.dtype)
        y_col     = y.unsqueeze(-1)                      # (B, Nr, 1)
        proj      = torch.bmm(U_c.conj().transpose(-2, -1), y_col)  # (B, r, 1)
        proj      = proj.squeeze(-1)                     # (B, r)
        Sigma_c   = Sigma_r.to(proj.dtype)
        s_hat     = proj / (Sigma_c + 1e-6)              # (B, r)

        # Convert to real (2r) for decoder
        s_hat_real = complex_to_real_stream(s_hat)       # (B, 2r)

        # ── 5. Decode: ŝ → reconstructed embeddings ───────────────
        task_emb_pred, modal_embs_pred = self.decoder(s_hat_real)

        # ── 6. Task head: classification from TRUE embedding ──────
        #    (we classify from original encoding; reconstruction is the
        #     semantic communication objective, not the classifier input)
        logits = self.task_head(task_emb_true)

        # Build Ac for PG constraint loss (Case 3 only)
        Ac = None
        if vmimo_case == "compress":
            Ac_blocks = [
                torch.mm(w_layer.weight, r_layer.weight)
                for w_layer, r_layer in zip(
                    self.vmimo.compress.W,
                    self.vmimo.compress.R,
                )
            ]
            Ac = torch.cat(Ac_blocks, dim=-1)  # (r, d_all)

        return {
            "task_emb_true":  task_emb_true,
            "task_emb_pred":  task_emb_pred,
            "modal_embs_true": modal_embs_true,
            "modal_embs_pred": modal_embs_pred,
            "logits":          logits,
            "vmimo_case":      vmimo_case,
            "s_hat":           s_hat,
            "Ac":              Ac,
            # Pass through for PG constraint
            "U_r":             U_r,
            "Sigma_r":         Sigma_r,
        }


# ─────────────────────────────────────────────────────────────────────
#  Training config
# ─────────────────────────────────────────────────────────────────────
class TrainConfig:
    # Data
    n_train:    int   = 50_000
    n_test:     int   = 10_000
    batch_size: int   = 64
    num_workers:int   = 0

    # Model
    d_model:    int   = 128
    n_heads:    int   = 4
    n_layers:   int   = 2
    r:          int   = 4
    Nt:         int   = 4
    Nr:         int   = 4
    eta1:       float = 0.6
    q_m:        int   = 8
    dropout:    float = 0.1

    # Training
    epochs:     int   = 30
    lr:         float = 5e-5
    lambda_cls: float = 1.0
    lambda_pg:  float = 0.5
    lambda_demod:float= 0.5
    snr_db:     float = 5.0        # training SNR

    # Eval
    eval_snrs:  List[float] = (-5.0, 2.5, 10.0)  # Fig 4 SNRs
    eval_every: int   = 1

    # Paths
    log_dir:    str   = "runs"
    seed:       int   = 42


# ─────────────────────────────────────────────────────────────────────
#  Metric tracker
# ─────────────────────────────────────────────────────────────────────
class MetricTracker:
    def __init__(self):
        self._sums:  Dict[str, float] = {}
        self._counts:Dict[str, int]   = {}

    def update(self, metrics: Dict[str, float], n: int = 1):
        for k, v in metrics.items():
            self._sums[k]   = self._sums.get(k, 0.0)   + v * n
            self._counts[k] = self._counts.get(k, 0)    + n

    def mean(self) -> Dict[str, float]:
        return {k: self._sums[k] / max(self._counts[k], 1)
                for k in self._sums}

    def reset(self):
        self._sums.clear()
        self._counts.clear()


# ─────────────────────────────────────────────────────────────────────
#  Main Trainer
# ─────────────────────────────────────────────────────────────────────
class E2ETrainer:
    """
    Full end-to-end trainer for the NAS-VMIMO system.

    Usage
    -----
    trainer = E2ETrainer(cfg)
    trainer.train()                          # runs all epochs
    trainer.evaluate(snr_db=-5.0)           # single SNR eval
    trainer.save_checkpoint("ckpt.pt")
    trainer.load_checkpoint("ckpt.pt")
    """

    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)

        # ── Device ────────────────────────────────────────────────
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")   # M4 Max
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print(f"[trainer] device: {self.device}")

        # ── Data ──────────────────────────────────────────────────
        modal_dims  = [mc.dim for mc in MODALITY_CONFIGS]
        modal_names = [mc.name for mc in MODALITY_CONFIGS]

        train_ds = MultimodalDataset(n_samples=cfg.n_train, seed=cfg.seed)
        test_ds  = MultimodalDataset(n_samples=cfg.n_test,  seed=cfg.seed+999,
                                     ssnr_degradation=False)
        self.train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True,
            collate_fn=collate_fn, num_workers=cfg.num_workers,
        )
        self.test_loader  = DataLoader(
            test_ds,  batch_size=cfg.batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=cfg.num_workers,
        )

        # ── Model ─────────────────────────────────────────────────
        self.model = VMIMOSystem(
            modal_dims=modal_dims,
            d_model=cfg.d_model, n_heads=cfg.n_heads, n_layers=cfg.n_layers,
            r=cfg.r, Nt=cfg.Nt, n_classes=N_CLASSES,
            eta1=cfg.eta1, q_m=cfg.q_m, dropout=cfg.dropout,
        ).to(self.device)

        # ── Channel & resource model ──────────────────────────────
        self.channel = MIMOChannel(
            MIMOChannelConfig(Nt=cfg.Nt, Nr=cfg.Nr, snr_db=cfg.snr_db)
        )
        self.resource_model = AR1ResourceModel(seed=cfg.seed)
        self._resource_trace = self.resource_model.generate_trace(
            n_steps=cfg.epochs * len(self.train_loader) + 1000
        )
        self._step = 0    # global step for trace indexing

        # ── Loss & optimizer ──────────────────────────────────────
        self.loss_fn = VMIMOLoss(
            lambda_cls=cfg.lambda_cls,
            lambda_pg=cfg.lambda_pg,
            lambda_demod=cfg.lambda_demod,
        )
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=cfg.lr, betas=(0.9, 0.999),
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 0.1,
        )

        # ── Logging ───────────────────────────────────────────────
        Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
        self.log_path  = Path(cfg.log_dir) / "train_log.jsonl"
        self.best_acc  = 0.0
        self.history: List[Dict] = []

    # ──────────────────────────────────────────────────────────────
    #  Channel batch helper
    # ──────────────────────────────────────────────────────────────
    def _get_channel_batch(self, batch_size: int, snr_db: Optional[float] = None):
        """Sample a fresh channel realization and move to device."""
        snr = snr_db if snr_db is not None else self.cfg.snr_db
        self.channel.cfg.snr_db = snr
        H, U_r, Sigma_r, V_sig, r = self.channel.generate(
            batch_size=batch_size, seed=None
        )
        # Use real part only for the VMIMOController (complex support TODO Phase 5)
        V_real    = V_sig.real.float().to(self.device)
        U_real    = U_r.real.float().to(self.device)
        Sigma_f   = Sigma_r.real.float().to(self.device)
        H_c       = H.to(torch.complex64).to(self.device)
        return H_c, U_real, Sigma_f, V_real

    # ──────────────────────────────────────────────────────────────
    #  Single training step
    # ──────────────────────────────────────────────────────────────
    def _train_step(
        self,
        modalities: Dict[str, torch.Tensor],
        labels:     torch.Tensor,
    ) -> Dict[str, float]:

        B = labels.shape[0]
        self.model.train()
        self.optimizer.zero_grad()

        # Channel
        H, U_r, Sigma_r, V_sig = self._get_channel_batch(B)

        # pjoint from AR(1) trace
        p_joint = self.resource_model.pjoint_from_trace(
            self._resource_trace, t=min(self._step, len(self._resource_trace["X_comp"])-1)
        )
        self._step += 1

        # Forward
        out = self.model(
            modalities=modalities,
            V_sig=V_sig, U_r=U_r, Sigma_r=Sigma_r, H=H,
            p_joint=p_joint, snr_db=self.cfg.snr_db,
        )

        # Build G for PG constraint: use (U_r Σ_r)^{-1} as approximate G
        G = None
        if out["Ac"] is not None:
            Sigma_diag_inv = 1.0 / (Sigma_r.unsqueeze(-1) + 1e-6)  # (B, r, 1)
            G = (U_r * Sigma_diag_inv.transpose(-2, -1)).transpose(-2, -1)

        # Loss
        total, breakdown = self.loss_fn(
            task_emb_pred    = out["task_emb_pred"],
            task_emb_true    = out["task_emb_true"],
            logits           = out["logits"],
            labels           = labels,
            modal_embs_pred  = out["modal_embs_pred"],
            modal_embs_true  = out["modal_embs_true"],
            G       = G,
            U_r     = U_r if G is not None else None,
            Sigma_r = Sigma_r if G is not None else None,
            Ac      = out["Ac"],
        )

        total.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Accuracy
        preds = out["logits"].argmax(dim=-1)
        acc   = (preds == labels).float().mean().item()
        breakdown["accuracy"]   = acc
        breakdown["vmimo_case"] = {"parallel": 0, "soft": 1, "compress": 2}[
            out["vmimo_case"]
        ]
        breakdown["p_joint"]    = p_joint
        return breakdown

    # ──────────────────────────────────────────────────────────────
    #  Evaluation
    # ──────────────────────────────────────────────────────────────
    @torch.no_grad()
    def evaluate(self, snr_db: float = 5.0) -> Dict[str, float]:
        self.model.eval()
        tracker = MetricTracker()

        for modalities, labels, _ in self.test_loader:
            modalities = {k: v.to(self.device) for k, v in modalities.items()}
            labels     = labels.to(self.device)
            B          = labels.shape[0]

            H, U_r, Sigma_r, V_sig = self._get_channel_batch(B, snr_db)
            p_joint = 0.7   # fixed at eval time for reproducibility

            out = self.model(
                modalities=modalities,
                V_sig=V_sig, U_r=U_r, Sigma_r=Sigma_r, H=H,
                p_joint=p_joint, snr_db=snr_db,
            )
            _, breakdown = self.loss_fn(
                task_emb_pred   = out["task_emb_pred"],
                task_emb_true   = out["task_emb_true"],
                logits          = out["logits"],
                labels          = labels,
                modal_embs_pred = out["modal_embs_pred"],
                modal_embs_true = out["modal_embs_true"],
            )
            preds = out["logits"].argmax(dim=-1)
            breakdown["accuracy"] = (preds == labels).float().mean().item()
            tracker.update(breakdown, n=B)

        return tracker.mean()

    # ──────────────────────────────────────────────────────────────
    #  Training loop
    # ──────────────────────────────────────────────────────────────
    def train(self):
        cfg = self.cfg
        print(f"[trainer] Starting E2E training  —  {cfg.epochs} epochs, "
              f"batch={cfg.batch_size}, lr={cfg.lr}")
        print(f"[trainer] Model params: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

        for epoch in range(1, cfg.epochs + 1):
            t0 = time.time()
            train_tracker = MetricTracker()
            case_counts = {"parallel": 0, "soft": 0, "compress": 0}
            case_names  = {0: "parallel", 1: "soft", 2: "compress"}

            for modalities, labels, _ in self.train_loader:
                modalities = {k: v.to(self.device) for k, v in modalities.items()}
                labels     = labels.to(self.device)
                step_metrics = self._train_step(modalities, labels)
                train_tracker.update(step_metrics, n=labels.shape[0])
                case_counts[case_names[int(step_metrics["vmimo_case"])]] += 1

            self.scheduler.step()
            train_means = train_tracker.mean()
            elapsed     = time.time() - t0

            # ── Evaluation ────────────────────────────────────────
            eval_results = {}
            if epoch % cfg.eval_every == 0:
                for snr in cfg.eval_snrs:
                    res = self.evaluate(snr_db=snr)
                    eval_results[f"test_acc_snr{snr:+.0f}dB"] = res["accuracy"]
                    eval_results[f"test_loss_snr{snr:+.0f}dB"] = res["L_total"]

            # ── Log ───────────────────────────────────────────────
            log_entry = {
                "epoch":       epoch,
                "train_loss":  train_means.get("L_total", 0),
                "train_acc":   train_means.get("accuracy", 0),
                "L_recv":      train_means.get("L_recv", 0),
                "L_cls":       train_means.get("L_cls", 0),
                "vmimo_cases": case_counts,
                "p_joint_avg": train_means.get("p_joint", 0),
                "elapsed_s":   elapsed,
                **eval_results,
            }
            self.history.append(log_entry)
            with open(self.log_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

            self._print_epoch(epoch, log_entry, elapsed)

            # ── Checkpoint best model ────────────────────────────
            primary_snr_key = f"test_acc_snr+5dB" if "test_acc_snr+5dB" in eval_results \
                              else (list(eval_results.keys())[0] if eval_results else None)
            if primary_snr_key and eval_results.get(primary_snr_key, 0) > self.best_acc:
                self.best_acc = eval_results[primary_snr_key]
                self.save_checkpoint(Path(cfg.log_dir) / "best_model.pt")

        print(f"\n[trainer] Training complete.  Best test acc: {self.best_acc:.4f}")
        return self.history

    # ──────────────────────────────────────────────────────────────
    #  Logging helper
    # ──────────────────────────────────────────────────────────────
    def _print_epoch(self, epoch: int, log: Dict, elapsed: float):
        cases  = log["vmimo_cases"]
        total_batches = sum(cases.values()) or 1
        case_str = (f"par={cases['parallel']}/{total_batches} "
                    f"soft={cases['soft']}/{total_batches} "
                    f"cmp={cases['compress']}/{total_batches}")

        # Collect eval accuracies
        eval_acc_strs = []
        for k, v in log.items():
            if k.startswith("test_acc"):
                snr_tag = k.replace("test_acc_snr", "").replace("dB", "")
                eval_acc_strs.append(f"{snr_tag}dB:{v*100:.1f}%")

        eval_str = "  eval: " + "  ".join(eval_acc_strs) if eval_acc_strs else ""

        print(
            f"[{epoch:3d}/{self.cfg.epochs}] "
            f"loss={log['train_loss']:.4f}  "
            f"L_recv={log['L_recv']:.4f}  "
            f"acc={log['train_acc']*100:.1f}%  "
            f"pjoint={log['p_joint_avg']:.3f}  "
            f"VMIMO:[{case_str}]"
            f"{eval_str}  "
            f"({elapsed:.1f}s)"
        )

    # ──────────────────────────────────────────────────────────────
    #  Checkpoint I/O
    # ──────────────────────────────────────────────────────────────
    def save_checkpoint(self, path):
        torch.save({
            "model_state":  self.model.state_dict(),
            "optim_state":  self.optimizer.state_dict(),
            "sched_state":  self.scheduler.state_dict(),
            "epoch":        len(self.history),
            "best_acc":     self.best_acc,
            "history":      self.history,
            "config":       self.cfg.__dict__,
        }, path)

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optim_state"])
        self.scheduler.load_state_dict(ckpt["sched_state"])
        self.best_acc = ckpt.get("best_acc", 0.0)
        self.history  = ckpt.get("history", [])
        print(f"[trainer] Loaded checkpoint from {path}  "
              f"(epoch {ckpt['epoch']}, best_acc={self.best_acc:.4f})")


# ─────────────────────────────────────────────────────────────────────
#  CLI entry point
# ─────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="NAS-VMIMO E2E Training")
    p.add_argument("--epochs",     type=int,   default=30)
    p.add_argument("--batch-size", type=int,   default=64)
    p.add_argument("--lr",         type=float, default=5e-5)
    p.add_argument("--d-model",    type=int,   default=128)
    p.add_argument("--n-layers",   type=int,   default=2)
    p.add_argument("--snr-db",     type=float, default=5.0)
    p.add_argument("--log-dir",    type=str,   default="runs")
    p.add_argument("--resume",     type=str,   default=None)
    p.add_argument("--eval-only",  action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cfg = TrainConfig()
    cfg.epochs     = args.epochs
    cfg.batch_size = args.batch_size
    cfg.lr         = args.lr
    cfg.d_model    = args.d_model
    cfg.n_layers   = args.n_layers
    cfg.snr_db     = args.snr_db
    cfg.log_dir    = args.log_dir

    trainer = E2ETrainer(cfg)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    if args.eval_only:
        for snr in (-5.0, 0.0, 5.0, 10.0):
            res = trainer.evaluate(snr_db=snr)
            print(f"  SNR={snr:+.0f}dB  acc={res['accuracy']*100:.2f}%  "
                  f"L_recv={res['L_recv']:.4f}")
    else:
        trainer.train()
