# NAS-VMIMO: Engineering Implementation

Task-Oriented Multimodal Wireless Semantic Communication
via Lightweight Language Models and Virtual MIMO.

## Project Structure

```
nas_vmimo/
├── data/
│   ├── dataset.py      ✅ Phase 0  — synthetic 5-modality dataset (50k/10k)
│   └── channel.py      ✅ Phase 0  — Rayleigh MIMO + AR(1) pjoint model
├── models/
│   ├── vmimo.py        ✅ Phase 1  — VMIMO: Case 1/2/3 + Controller
│   ├── encoder.py      ✅ Phase 2  — Meta-Transformer style multimodal encoder
│   └── resource.py     🔲 Phase 2  — pjoint runtime monitor wrapper
├── nas/
│   ├── search_space.py 🔲 Phase 4  — DARTS-style search space
│   ├── trainer.py      🔲 Phase 4  — bilevel NAS training loop
│   └── strategy.py     🔲 Phase 4  — 3-stage NAS (Algorithm 2)
├── train/
│   ├── loss.py         ✅ Phase 3  — L_recv + penalty terms (Equ. 23), grad verified
│   └── e2e_trainer.py  🔲 Phase 3  — end-to-end training loop
└── configs/
    └── default.yaml    ✅            — all hyperparameters

✅ = done   🔲 = next
```

## Setup (M4 Max / any Mac with Python 3.10+)

```bash
# 1. Install dependencies
pip install torch torchvision numpy scipy pyyaml tensorboard

# 2. Verify Phase 0
python data/dataset.py
python data/channel.py

# 3. Verify Phase 1 VMIMO
python models/vmimo.py
```

## Phase Roadmap

### Phase 0 — Data & Channel  ✅
- `dataset.py`  : 5-modality Gaussian mixture, SSNR labels
- `channel.py`  : Rayleigh SU-MIMO, SVD precoding, ZF receiver, AR(1) pjoint

Key design decision: pjoint uses a synthetic AR(1) trace (phi_comp=0.85,
phi_mem=0.90) rather than OS APIs.  To switch to real device monitoring,
replace `AR1ResourceModel.pjoint_from_trace` with:
```python
import psutil
X_comp = math.log(psutil.cpu_percent()/100 + 1e-3)
X_mem  = math.log(1 - psutil.virtual_memory().percent/100 + 1e-3)
```

### Phase 1 — VMIMO Core  ✅
- `vmimo.py`: Implements Equ. 7–13 in full
  - `ParallelMapping`        — Case 1: Ak = I_{r'}
  - `SoftSelectionMapping`   — Case 2: Ak = A_soft (Equ. 8–10)
  - `CompressionMapping`     — Case 3: Ak = Ac (Equ. 11–13)
  - `ModalityAttentionWeight`— Equ. 9: w(m) with λ, μ, β(m)
  - `VMIMOController`        — Algorithm 1 control flow

### Phase 2 — Encoder  🔲 (needs Meta-Transformer details)
The encoder interface is:
```python
class MultimodalEncoder(nn.Module):
    def forward(self, modalities: Dict[str, Tensor]) -> List[Tensor]:
        # returns list of M' embeddings, each (B, d_model)
        ...
```
When Meta-Transformer paper/code is provided, implement:
- Token-type embedding per modality
- Shared Transformer backbone (Meta-Transformer style)
- Per-modality output projection to d_model

Until then, `backend: "standard"` in config uses independent
per-modality linear projections + shared Transformer.

### Phase 3 — E2E Training  🔲
Loss (Equ. 23):
```
L_total = L_recv + λ_PG * ReLU(c_PG)² + Σ_m λ_m * ReLU(c_demod_m)²
L_recv  = 1 - cosine(E(T̂), E(T))
```
Backprop chain: Equ. 24–30.
Gradient through discrete A_soft (Case 2) uses straight-through estimator.

### Phase 4 — NAS  🔲
DARTS-style bilevel optimization:
- Architecture params α_{i,j} (Equ. 31) on validation set
- Weight params on training set
- Resource constraints: P_total ≤ θ_mem (Equ. 15)
- 3-stage search (Algorithm 2)

## Key Paper→Code Mapping

| Paper equation | File | Class/Function |
|---|---|---|
| Equ. 1  | channel.py | `AR1ResourceModel` |
| Equ. 2  | channel.py | `p_succ` |
| Equ. 3  | channel.py | `p_succ` (after channel) |
| Equ. 5–6| vmimo.py  | `VMIMOController.forward` |
| Equ. 7  | vmimo.py  | `VMIMOController.forward` (if-else) |
| Equ. 8–10| vmimo.py | `SoftSelectionMapping` |
| Equ. 9  | vmimo.py  | `ModalityAttentionWeight` |
| Equ. 11–13| vmimo.py| `CompressionMapping` |
| Equ. 15 | nas/search_space.py | `count_params` |
| Equ. 17 | train/loss.py | `semantic_loss` |
| Equ. 18 | vmimo.py | `CompressionMapping.reconstruction_consistency_loss` |
| Equ. 19 | channel.py | `p_succ` |
| Equ. 20–22| channel.py | `AR1ResourceModel.pjoint` |
| Equ. 23 | train/loss.py | `total_loss` |
| Equ. 31 | nas/search_space.py | `MixedOp` |
| Algorithm 1 | vmimo.py | `VMIMOController.forward` |
| Algorithm 2 | nas/strategy.py | `ThreeStageNAS` |
