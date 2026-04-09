#!/usr/bin/env python3
"""
Quick-start smoke test for NAS-VMIMO on M4 Max (or any torch machine).

Run:  python run_e2e.py [--quick] [--epochs N] [--snr-db X]

--quick  : 3 epochs, batch=16, tiny dataset (500 samples) — ~30s on M4 Max
default  : 30 epochs, batch=64, full dataset (50k/10k)    — ~20min on M4 Max
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick",     action="store_true", help="Smoke test: 3 epochs, small data")
    p.add_argument("--epochs",    type=int,   default=None)
    p.add_argument("--batch-size",type=int,   default=None)
    p.add_argument("--snr-db",    type=float, default=5.0)
    p.add_argument("--d-model",   type=int,   default=128)
    p.add_argument("--n-layers",  type=int,   default=2)
    p.add_argument("--log-dir",   type=str,   default="runs")
    p.add_argument("--resume",    type=str,   default=None)
    p.add_argument("--eval-only", action="store_true")
    args = p.parse_args()

    from train.e2e_trainer import E2ETrainer, TrainConfig
    from data.dataset import MultimodalDataset, collate_fn, MODALITY_CONFIGS
    import torch
    from torch.utils.data import DataLoader

    cfg = TrainConfig()

    if args.quick:
        print("[run_e2e] Quick mode: 3 epochs, 500 train / 100 test samples")
        cfg.epochs      = 3
        cfg.batch_size  = 16
        cfg.n_train     = 500
        cfg.n_test      = 100
        cfg.eval_snrs   = (-5.0, 5.0)
        cfg.eval_every  = 1
    else:
        cfg.epochs      = args.epochs    or 30
        cfg.batch_size  = args.batch_size or 64
        cfg.n_train     = 50_000
        cfg.n_test      = 10_000
        cfg.eval_snrs   = (-5.0, 2.5, 10.0)

    cfg.snr_db   = args.snr_db
    cfg.d_model  = args.d_model
    cfg.n_layers = args.n_layers
    cfg.log_dir  = args.log_dir

    # Print device info
    if torch.backends.mps.is_available():
        print("[run_e2e] Backend: Apple MPS (M-series)")
    elif torch.cuda.is_available():
        print(f"[run_e2e] Backend: CUDA ({torch.cuda.get_device_name(0)})")
    else:
        print("[run_e2e] Backend: CPU")

    trainer = E2ETrainer(cfg)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    if args.eval_only:
        print("[run_e2e] Eval-only mode")
        for snr in (-10.0, -5.0, 0.0, 5.0, 10.0):
            res = trainer.evaluate(snr_db=snr)
            print(f"  SNR={snr:+5.1f}dB  acc={res['accuracy']*100:5.1f}%  "
                  f"L_recv={res['L_recv']:.4f}  L_total={res['L_total']:.4f}")
    else:
        history = trainer.train()
        print(f"\n[run_e2e] Done. Log saved to {cfg.log_dir}/train_log.jsonl")

        # Brief results summary
        last = history[-1]
        print(f"\n── Final epoch results ──────────────────────────────────")
        print(f"  Train loss  : {last['train_loss']:.4f}")
        print(f"  Train acc   : {last['train_acc']*100:.1f}%")
        for k, v in last.items():
            if k.startswith("test_acc"):
                print(f"  {k:30s}: {v*100:.1f}%")

if __name__ == "__main__":
    main()
