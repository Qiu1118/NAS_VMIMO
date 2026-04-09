"""
Synthetic multimodal dataset for NAS-VMIMO.

5 modalities (Table in Sec. V of the paper):
  visual:      128-dim
  radar:        32-dim
  infrared:     32-dim
  audio:        64-dim
  environmental:64-dim
  → total semantic dim = 320

4 output classes, 50k train / 10k test.
Each class has a distinct Gaussian mixture centre per modality,
plus class-dependent noise to simulate varying SSNR.
"""

from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ─────────────────────────────────────────────
#  Modality configuration
# ─────────────────────────────────────────────
@dataclass
class ModalityConfig:
    name: str
    dim: int
    noise_std: float = 1.0          # base additive noise per modality


MODALITY_CONFIGS: List[ModalityConfig] = [
    ModalityConfig("visual",        128, noise_std=0.8),
    ModalityConfig("radar",          32, noise_std=1.2),
    ModalityConfig("infrared",       32, noise_std=1.0),
    ModalityConfig("audio",          64, noise_std=0.9),
    ModalityConfig("environmental",  64, noise_std=1.1),
]

N_CLASSES    = 4
TOTAL_DIM    = sum(m.dim for m in MODALITY_CONFIGS)   # 320
N_TRAIN      = 50_000
N_TEST       = 10_000
RANDOM_SEED  = 42


# ─────────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────────
class MultimodalDataset(Dataset):
    """
    Generates class-conditional Gaussian mixture data per modality.

    Each sample:
        modalities: dict[str, Tensor(dim,)]  – per-modality feature vectors
        label:      int                       – class index ∈ {0,…,N_CLASSES-1}
        ssnr:       dict[str, float]          – sensing SNR per modality (linear)

    The SSNR for modality m of class c is:
        SSNR(m,c) = ||signal||² / noise_std²
    where signal is the class mean (fixed) and noise is i.i.d. Gaussian.
    """

    def __init__(
        self,
        n_samples: int = N_TRAIN,
        n_classes: int = N_CLASSES,
        modality_configs: List[ModalityConfig] = MODALITY_CONFIGS,
        seed: int = RANDOM_SEED,
        ssnr_degradation: bool = True,  # randomly degrade some modalities
    ):
        super().__init__()
        rng = np.random.default_rng(seed)

        self.modality_configs = modality_configs
        self.n_classes = n_classes

        # Class-specific means per modality (fixed, reproducible)
        # Shape: (n_classes, dim) for each modality
        self._class_means: Dict[str, np.ndarray] = {
            mc.name: rng.standard_normal((n_classes, mc.dim)) * 3.0
            for mc in modality_configs
        }

        # Generate labels
        labels = rng.integers(0, n_classes, size=n_samples)

        # Generate features
        modality_data: Dict[str, np.ndarray] = {}
        ssnr_data:     Dict[str, np.ndarray] = {}

        for mc in modality_configs:
            means = self._class_means[mc.name][labels]      # (N, dim)
            noise = rng.standard_normal((n_samples, mc.dim)) * mc.noise_std

            if ssnr_degradation:
                # Randomly degrade ~20% of samples per modality
                degrade_mask = rng.random(n_samples) < 0.2
                extra_noise  = rng.standard_normal((n_samples, mc.dim)) * 4.0
                noise[degrade_mask] += extra_noise[degrade_mask]

            features = means + noise

            signal_power = np.sum(means ** 2, axis=1) / mc.dim          # (N,)
            noise_power  = np.sum(noise  ** 2, axis=1) / mc.dim         # (N,)
            ssnr         = signal_power / (noise_power + 1e-8)          # (N,)

            modality_data[mc.name] = features.astype(np.float32)
            ssnr_data[mc.name]     = ssnr.astype(np.float32)

        self.labels        = torch.from_numpy(labels.astype(np.int64))
        self.modality_data = {k: torch.from_numpy(v) for k, v in modality_data.items()}
        self.ssnr_data     = {k: torch.from_numpy(v) for k, v in ssnr_data.items()}

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, float]]:
        modalities = {k: v[idx] for k, v in self.modality_data.items()}
        ssnr       = {k: v[idx].item() for k, v in self.ssnr_data.items()}
        return modalities, self.labels[idx], ssnr


# ─────────────────────────────────────────────
#  Collate function
# ─────────────────────────────────────────────
def collate_fn(batch):
    """Stack a list of (modalities, label, ssnr) into batched tensors."""
    modalities_list, labels_list, ssnr_list = zip(*batch)

    # Stack each modality
    batched_modalities = {
        k: torch.stack([m[k] for m in modalities_list], dim=0)
        for k in modalities_list[0]
    }
    batched_labels = torch.stack(labels_list, dim=0)
    # ssnr stays as list-of-dicts (used for VMIMO switching, not loss)
    return batched_modalities, batched_labels, ssnr_list


# ─────────────────────────────────────────────
#  DataLoader factory
# ─────────────────────────────────────────────
def build_dataloaders(
    n_train: int = N_TRAIN,
    n_test:  int = N_TEST,
    batch_size: int = 64,
    num_workers: int = 0,
    seed: int = RANDOM_SEED,
) -> Tuple[DataLoader, DataLoader]:
    train_dataset = MultimodalDataset(n_samples=n_train, seed=seed,       ssnr_degradation=True)
    test_dataset  = MultimodalDataset(n_samples=n_test,  seed=seed + 999, ssnr_degradation=False)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True,  collate_fn=collate_fn, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset,  batch_size=batch_size,
        shuffle=False, collate_fn=collate_fn, num_workers=num_workers
    )
    return train_loader, test_loader


# ─────────────────────────────────────────────
#  Quick sanity check
# ─────────────────────────────────────────────
if __name__ == "__main__":
    train_loader, test_loader = build_dataloaders(batch_size=8)
    modalities, labels, ssnr = next(iter(train_loader))

    print("=== Dataset sanity check ===")
    print(f"Modalities: {list(modalities.keys())}")
    for name, tensor in modalities.items():
        print(f"  {name:16s}: shape={tuple(tensor.shape)}, mean={tensor.mean():.3f}")
    print(f"Labels shape : {labels.shape},  unique={labels.unique().tolist()}")
    print(f"SSNR sample  : { {k: f'{v:.3f}' for k,v in ssnr[0].items()} }")
    print(f"Train batches: {len(train_loader)},  Test batches: {len(test_loader)}")
    print(f"Total semantic dim: {TOTAL_DIM}")
