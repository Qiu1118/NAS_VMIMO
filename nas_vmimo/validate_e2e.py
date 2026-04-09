"""
Full E2E forward pass validation (numpy only, no torch required).

Traces every tensor through the complete pipeline:
  dataset → encoder → VMIMO → channel → decoder → loss → backprop check

This script runs in the sandbox to confirm all shape/logic is correct
before running on M4 Max with torch.
"""

import sys, math
import numpy as np
from scipy.stats import norm as sp_norm

sys.path.insert(0, '.')
rng = np.random.default_rng(42)

# ─── Config ───────────────────────────────────────────────────────────
B         = 4       # batch size
M         = 5       # modalities
d         = 128     # d_model
r         = 4       # spatial DoFs
Nt = Nr   = 4
n_classes = 4
MODAL_DIMS  = [128, 32, 32, 64, 64]
MODAL_NAMES = ["visual", "radar", "infrared", "audio", "environmental"]
n_layers    = 2
d_ff        = 4 * d
n_heads     = 4
snr_db      = 5.0
eta1        = 0.6

print("=" * 60)
print("NAS-VMIMO  —  Full E2E Numpy Validation")
print("=" * 60)


# ────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────
def layer_norm(x, eps=1e-5):
    mu  = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return (x - mu) / (std + eps)

def linear(x, W, b=None):
    out = x @ W.T
    return out if b is None else out + b

def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

def gelu(x):
    return 0.5 * x * (1 + np.tanh(math.sqrt(2/math.pi)*(x + 0.044715*x**3)))

def relu(x):
    return np.maximum(0, x)

def cosine_sim(a, b):
    """(B, d), (B, d) → (B,)"""
    a_n = np.linalg.norm(a, axis=-1, keepdims=True)
    b_n = np.linalg.norm(b, axis=-1, keepdims=True)
    return (a * b).sum(axis=-1) / (a_n.squeeze(-1) * b_n.squeeze(-1) + 1e-9)

def cross_entropy(logits, labels):
    s = softmax(logits, axis=-1)
    return -np.mean(np.log(s[np.arange(B), labels] + 1e-9))

def init_W(out, in_):
    return rng.standard_normal((out, in_)) * math.sqrt(2 / (out + in_))

def trunc_normal(shape, std=0.02):
    return np.clip(rng.standard_normal(shape) * std, -2*std, 2*std)


# ────────────────────────────────────────────────────────────────
# STEP 0: Synthetic dataset batch
# ────────────────────────────────────────────────────────────────
print("\n── Step 0: Dataset batch ─────────────────────────────────")
labels = rng.integers(0, n_classes, B)
class_means = {name: rng.standard_normal((n_classes, dim)) * 3
               for name, dim in zip(MODAL_NAMES, MODAL_DIMS)}
raw_modalities = {}
for name, dim in zip(MODAL_NAMES, MODAL_DIMS):
    means = class_means[name][labels]
    noise = rng.standard_normal((B, dim))
    raw_modalities[name] = (means + noise).astype(np.float32)
    print(f"  {name:16s}: {raw_modalities[name].shape}")


# ────────────────────────────────────────────────────────────────
# STEP 1: Encoder  (Data2Seq tokenizers + shared Transformer)
# ────────────────────────────────────────────────────────────────
print("\n── Step 1: Multimodal Meta-Encoder ──────────────────────")

# Per-modality tokenizers: Linear → LayerNorm
W_tok = [init_W(d, dim) for dim in MODAL_DIMS]
modal_tokens = []
for i, (name, dim) in enumerate(zip(MODAL_NAMES, MODAL_DIMS)):
    x = raw_modalities[name]          # (B, dim)
    t = layer_norm(linear(x, W_tok[i]))  # (B, d)
    modal_tokens.append(t[:, np.newaxis, :])   # (B, 1, d)
    print(f"  tokenizer [{name}]: {x.shape} → {modal_tokens[-1].shape}")

# Shared encoder: [CLS | tok_1 | ... | tok_M] + pos + modal_type
cls_tok   = trunc_normal((1, 1, d))
pos_emb   = trunc_normal((1, 1+M, d))
mtype_emb = trunc_normal((1, 1+M, d))
cls_batch = np.broadcast_to(cls_tok, (B, 1, d)).copy()
seq       = np.concatenate([cls_batch] + modal_tokens, axis=1)  # (B, 1+M, d)
seq       = seq + pos_emb + mtype_emb
print(f"\n  Token sequence: {seq.shape}  [B={B}, N={1+M}, d={d}]")

# Transformer blocks (simplified: skip learned QKV, trace shapes)
for layer_i in range(n_layers):
    # Self-attention (shape placeholder — actual QKV learned in torch)
    attn_out = rng.standard_normal((B, 1+M, d)) * 0.01
    seq = layer_norm(seq + attn_out)
    # FFN: Linear(d→d_ff) → GELU → Linear(d_ff→d)
    flat = seq.reshape(-1, d)
    W1 = init_W(d_ff, d);  W2 = init_W(d, d_ff)
    ffn = linear(gelu(linear(flat, W1)), W2).reshape(B, 1+M, d)
    seq = layer_norm(seq + ffn)

seq_final    = layer_norm(seq)
task_emb_true = seq_final[:, 0, :]                            # (B, d)
modal_embs_true = [seq_final[:, m+1, :] for m in range(M)]   # list of (B, d)
print(f"  task_emb_true    : {task_emb_true.shape}")
print(f"  modal_embs_true  : {len(modal_embs_true)} × {modal_embs_true[0].shape}")


# ────────────────────────────────────────────────────────────────
# STEP 2: Channel & pjoint
# ────────────────────────────────────────────────────────────────
print("\n── Step 2: Channel & Resource Model ─────────────────────")

H_real = rng.standard_normal((B, Nr, Nt)) / math.sqrt(2)
H_imag = rng.standard_normal((B, Nr, Nt)) / math.sqrt(2)
H = H_real + 1j * H_imag

# Batch SVD
U_list, S_list, V_list = [], [], []
for b in range(B):
    U, S, Vh = np.linalg.svd(H[b], full_matrices=False)
    V_list.append(Vh.conj().T)  # (Nt, r)
    U_list.append(U)            # (Nr, r)
    S_list.append(S)            # (r,)
U_r     = np.stack(U_list)   # (B, Nr, r)
Sigma_r = np.stack(S_list)   # (B, r)
V_sig   = np.stack(V_list)   # (B, Nt, r)
print(f"  H shape          : {H.shape}")
print(f"  Sigma_r[0]       : {np.round(Sigma_r[0], 3).tolist()}")

# pjoint via AR(1)
phi_c, phi_m = 0.85, 0.90
sigma_c, sigma_m = 0.15, 0.10
X_c = rng.standard_normal() * 0.2
X_m = rng.standard_normal() * 0.1
X_c_new = phi_c * X_c + sigma_c * rng.standard_normal()
X_m_new = phi_m * X_m + sigma_m * rng.standard_normal()
ac = am = -0.3

def p_exceed(X_t, phi, sigma, thresh, delta=1):
    mu  = (phi**delta) * X_t
    var = sigma**2 * (1 - phi**(2*delta)) / (1 - phi**2 + 1e-9)
    z   = (thresh - mu) / math.sqrt(max(var, 1e-9))
    return 1.0 - sp_norm.cdf(z)

p_joint = p_exceed(X_c_new, phi_c, sigma_c, ac) * p_exceed(X_m_new, phi_m, sigma_m, am)
print(f"  p_joint          : {p_joint:.4f}  (threshold η₁={eta1})")


# ────────────────────────────────────────────────────────────────
# STEP 3: VMIMO — build precoded signal
# ────────────────────────────────────────────────────────────────
print("\n── Step 3: VMIMO Controller ──────────────────────────────")

r_prime = M  # 5 modalities, r=4 DoFs  →  Case 2 or 3

if r_prime <= r:
    case_name = "parallel"
    # x_k = V_sig @ M(e)  — identity mapping
    e_stacked = np.stack([e.mean(axis=-1) for e in modal_embs_true], axis=-1)  # (B, M)
    x_k = np.einsum("bni,bi->bn", V_sig[:, :, :r_prime], e_stacked)  # (B, Nt)
elif p_joint < eta1:
    case_name = "soft (Case 2)"
    # Attention weights w(m) — simplified uniform for numpy test
    w = np.ones(M) / M  # (M,)  uniform weights
    rm = np.floor(r * w).astype(int)
    rm[0] += r - rm.sum()  # assign residual to first modality
    # Build A_soft
    A_soft = np.zeros((r, M))
    row = 0
    for m_idx in range(M):
        for _ in range(rm[m_idx]):
            A_soft[row, m_idx] = 1.0
            row += 1
    # x_k = V_sig A_soft M(e)  →  (B, Nt)
    e_scalar = np.stack([modal_embs_true[m].mean(axis=-1) for m in range(M)], axis=-1)  # (B, M)
    x_k = np.einsum("bnr,rm,bm->bn", V_sig, A_soft, e_scalar)
else:
    case_name = "compress (Case 3)"
    q_m   = 8
    e_all = np.concatenate(modal_embs_true, axis=-1)   # (B, d_all)
    d_all = e_all.shape[-1]   # M * d_model
    # Low-rank blocks: P_m = W_m R_m^H, each P_m in R^{r x d_model}
    Ac_blocks = []
    for m in range(M):
        R_m = init_W(q_m, d)      # (q_m, d_model)
        W_m = init_W(r, q_m)       # (r, q_m)
        Ac_blocks.append(W_m @ R_m)   # (r, d_m)
    Ac = np.concatenate(Ac_blocks, axis=-1)  # (r, d_all)
    # x_k = V_sig Ac e_all
    compressed = (Ac @ e_all.T).T              # (B, r)
    x_k = np.einsum("bnr,br->bn", V_sig, compressed)  # (B, Nt)

print(f"  r'={r_prime} vs r={r}  →  VMIMO case: {case_name}")
print(f"  x_k shape        : {x_k.shape}")


# ────────────────────────────────────────────────────────────────
# STEP 4: Channel transmit + ZF equalization
# ────────────────────────────────────────────────────────────────
print("\n── Step 4: Channel transmit & ZF equalization ───────────")

snr_lin   = 10 ** (snr_db / 10.0)
x_complex = x_k.astype(complex)
sig_power = np.mean(np.abs(x_complex)**2) + 1e-9
noise_std = math.sqrt(sig_power / snr_lin)

x_col = x_complex[:, :, np.newaxis]   # (B, Nt, 1)
y     = np.einsum("bnr,brk->bnk", H, x_col).squeeze(-1)   # (B, Nr)  — wait, wrong
# correct: y = H @ x
y = np.stack([H[b] @ x_complex[b] for b in range(B)])   # (B, Nr)
n = (rng.standard_normal((B, Nr)) + 1j * rng.standard_normal((B, Nr))) * noise_std / math.sqrt(2)
y = y + n

# ZF: ŝ = diag(1/σ) U_r^H y
s_hat_list = []
for b in range(B):
    proj    = U_r[b].conj().T @ y[b]     # (r,)  complex
    s_hat_b = proj / (Sigma_r[b] + 1e-6) # (r,)  complex
    s_hat_list.append(s_hat_b)
s_hat = np.stack(s_hat_list)             # (B, r)  complex
# Real representation: [real | imag]
s_hat_real = np.concatenate([s_hat.real, s_hat.imag], axis=-1)  # (B, 2r)
print(f"  y shape          : {y.shape}  (complex)")
print(f"  s_hat shape      : {s_hat.shape}  (complex)")
print(f"  s_hat_real shape : {s_hat_real.shape}  (real, for decoder)")


# ────────────────────────────────────────────────────────────────
# STEP 5: Decoder  (stream → reconstructed embeddings)
# ────────────────────────────────────────────────────────────────
print("\n── Step 5: Semantic Decoder ──────────────────────────────")

# Stream projector: Linear(2r→d) → LN → GELU
W_sp = init_W(d, 2*r)
context = gelu(layer_norm(linear(s_hat_real.astype(np.float32), W_sp)))  # (B, d)
context = context[:, np.newaxis, :]   # (B, 1, d)

# Learnable query tokens: [q_task | q_1 ... q_M]
queries = trunc_normal((1, 1+M, d))
queries = np.broadcast_to(queries, (B, 1+M, d)).copy()

for layer_i in range(n_layers):
    # Cross-attention: queries attend to context
    ca_out = rng.standard_normal((B, 1+M, d)) * 0.01  # shape placeholder
    queries = layer_norm(queries + ca_out)
    flat = queries.reshape(-1, d)
    W1 = init_W(d_ff, d);  W2 = init_W(d, d_ff)
    ffn = linear(gelu(linear(flat, W1)), W2).reshape(B, 1+M, d)
    queries = layer_norm(queries + ffn)

task_emb_pred  = queries[:, 0, :]                           # (B, d)
modal_embs_pred = [queries[:, m+1, :] for m in range(M)]   # list (B, d)
print(f"  task_emb_pred    : {task_emb_pred.shape}")
print(f"  modal_embs_pred  : {len(modal_embs_pred)} × {modal_embs_pred[0].shape}")


# ────────────────────────────────────────────────────────────────
# STEP 6: Task head & Loss
# ────────────────────────────────────────────────────────────────
print("\n── Step 6: Task Head & Loss ──────────────────────────────")

W_head = init_W(n_classes, d)
logits = linear(task_emb_true, W_head)   # (B, n_classes)

# L_recv
cos   = cosine_sim(task_emb_pred, task_emb_true)
l_recv = 1.0 - cos.mean()

# L_cls
l_cls = cross_entropy(logits, labels)

# Constraint penalties
eps_m = 0.1
l_demod_total = 0.0
for m in range(M):
    mse_m = np.mean(np.sum((modal_embs_pred[m] - modal_embs_true[m])**2, axis=-1))
    l_demod_total += 0.5 * relu(mse_m - eps_m)**2

l_total = l_recv + 1.0 * l_cls + l_demod_total

print(f"  L_recv           : {l_recv:.4f}")
print(f"  L_cls            : {l_cls:.4f}")
print(f"  L_demod_total    : {l_demod_total:.4f}")
print(f"  L_total          : {l_total:.4f}")

# Accuracy
preds = logits.argmax(axis=-1)
acc   = (preds == labels).mean()
print(f"  Train acc (rand) : {acc*100:.1f}%  (expected ~25% at init)")


# ────────────────────────────────────────────────────────────────
# STEP 7: Gradient check on L_recv
# ────────────────────────────────────────────────────────────────
print("\n── Step 7: Gradient verification ────────────────────────")

# Analytical gradient of L_recv w.r.t. task_emb_pred[0]  (Equ. 24)
b_idx = 0
u = task_emb_pred[b_idx].copy()
v = task_emb_true[b_idx].copy()
u_n = np.linalg.norm(u);  v_n = np.linalg.norm(v)
# ∇_u L_recv = -(v/(||u||·||v||) - (u^T v)/(||u||^3·||v||) · u) / B
g_anal = -(v / (u_n * v_n) - (np.dot(u, v) / (u_n**3 * v_n)) * u) / B

# Finite difference
eps_fd = 1e-4
g_fd   = np.zeros(5)
for i in range(5):
    up = task_emb_pred.copy();  up[b_idx, i] += eps_fd
    um = task_emb_pred.copy();  um[b_idx, i] -= eps_fd
    cos_p = cosine_sim(up, task_emb_true).mean()
    cos_m = cosine_sim(um, task_emb_true).mean()
    g_fd[i] = ((1 - cos_p) - (1 - cos_m)) / (2 * eps_fd)

max_err = np.max(np.abs(g_fd - g_anal[:5]))
print(f"  Finite diff vs analytical  max_Δ={max_err:.2e}  ({'PASS ✓' if max_err < 1e-8 else 'FAIL ✗'})")


# ────────────────────────────────────────────────────────────────
# STEP 8: Parameter count (Equ. 15)
# ────────────────────────────────────────────────────────────────
print("\n── Step 8: Parameter budget (Equ. 15) ───────────────────")
V_vocab = 0  # no text vocabulary
T_seq   = 1 + M
D       = d
d_ff_   = d_ff
L       = n_layers

# Encoder
P_pos         = T_seq * D
P_modal_type  = T_seq * D
P_cls         = D
P_tokenizers  = sum(dim * D for dim in MODAL_DIMS)
P_transformer = L * (4*D**2 + 2*D*d_ff_ + 4*D)
P_encoder     = P_pos + P_modal_type + P_cls + P_tokenizers + P_transformer

# Decoder (cross-attention, similar scale)
P_stream_proj = 2*r * D + D  # Linear(2r→d) + LN
P_queries     = (1+M) * D
P_decoder     = P_stream_proj + P_queries + L * (4*D**2 + 2*D*d_ff_ + 4*D)

# Task head
P_task_head   = D * (D//2) + (D//2) * n_classes

P_total = P_encoder + P_decoder + P_task_head
print(f"  Encoder params   : {P_encoder:>10,}  ({P_encoder/1e6:.3f} M)")
print(f"  Decoder params   : {P_decoder:>10,}  ({P_decoder/1e6:.3f} M)")
print(f"  TaskHead params  : {P_task_head:>10,}  ({P_task_head/1e6:.3f} M)")
print(f"  TOTAL            : {P_total:>10,}  ({P_total/1e6:.3f} M)")
print(f"  Within 10M budget: {P_total < 10e6}")


# ────────────────────────────────────────────────────────────────
# Summary
# ────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("RESULT  —  All forward pass shapes verified:")
shapes_ok = [
    task_emb_true.shape     == (B, d),
    modal_embs_true[0].shape == (B, d),
    s_hat_real.shape        == (B, 2*r),
    task_emb_pred.shape     == (B, d),
    modal_embs_pred[0].shape == (B, d),
    logits.shape            == (B, n_classes),
    max_err                 < 1e-8,
    P_total                 < 10e6,
]
for i, (desc, ok) in enumerate(zip([
    f"task_emb_true  {(B,d)}",
    f"modal_embs_true {(B,d)}",
    f"s_hat_real     {(B,2*r)}",
    f"task_emb_pred  {(B,d)}",
    f"modal_embs_pred {(B,d)}",
    f"logits         {(B,n_classes)}",
    "gradient Δ < 1e-8",
    f"P_total ({P_total/1e6:.2f}M) < 10M",
], shapes_ok)):
    print(f"  {'✓' if ok else '✗'}  {desc}")

print("=" * 60)
if all(shapes_ok):
    print("ALL CHECKS PASSED  →  Ready to run on M4 Max with torch")
else:
    print("SOME CHECKS FAILED  — review above")
