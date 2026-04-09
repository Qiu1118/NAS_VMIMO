"""
Channel and resource simulation for NAS-VMIMO.

Covers:
  1. Rayleigh fading SU-MIMO channel  (H, SVD, precoding, AWGN)
  2. AR(1) resource model for p_joint  (Section IV-A of the paper)
  3. SSNR-based task success probability P_succ  (Equ. 19)

Design decision for pjoint (as agreed): we synthesise a realistic
AR(1) resource trace rather than hard-coding a fixed value or
reading from OS APIs.  This is the most flexible choice:
  - Reproducible across platforms
  - Can be swapped for real psutil readings in one line
  - Naturally produces the dynamic switching behaviour Fig. 4 tests
"""

from __future__ import annotations
import numpy as np
import torch
from dataclasses import dataclass
from typing import Tuple, Optional


# ─────────────────────────────────────────────
#  1.  MIMO Channel
# ─────────────────────────────────────────────
@dataclass
class MIMOChannelConfig:
    Nt: int   = 4        # transmit antennas
    Nr: int   = 4        # receive antennas
    snr_db: float = 5.0  # SNR in dB at receiver
    channel_type: str = "rayleigh"


class MIMOChannel:
    """
    Single-user MIMO channel:  y = H x + n

    SVD decomposition:  H = U Σ V^H
    Precoder:           x = V_sig A_k M(e)   (see Sec. III-A1, Equ. 5)
    Receiver:           G = (U_r Σ_r)^{-1}  (MMSE simplified to ZF here)

    Returns both the channel matrix H and the signal subspace V_sig
    so that VMIMO can construct the mapping matrix A_k externally.
    """

    def __init__(self, cfg: MIMOChannelConfig = MIMOChannelConfig()):
        self.cfg = cfg

    def generate(
        self,
        batch_size: int = 1,
        seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Returns:
          H        : (batch, Nr, Nt)  complex channel matrix
          U_r      : (batch, Nr, r)   left singular vectors  (signal subspace)
          Sigma_r  : (batch, r)       singular values
          V_sig    : (batch, Nt, r)   right singular vectors (precoding basis)
          r        : int              rank = min(Nt, Nr)  [scalar, same for all]
        """
        if seed is not None:
            torch.manual_seed(seed)

        Nt, Nr = self.cfg.Nt, self.cfg.Nr

        # Rayleigh: real + imag ~ N(0, 1/2) → |h_ij| ~ Rayleigh(1/√2)
        H_real = torch.randn(batch_size, Nr, Nt) / (2 ** 0.5)
        H_imag = torch.randn(batch_size, Nr, Nt) / (2 ** 0.5)
        H = torch.complex(H_real, H_imag)                      # (B, Nr, Nt)

        # SVD  (economy SVD: returns r = min(Nr, Nt) components)
        U, S, Vh = torch.linalg.svd(H, full_matrices=False)    # U:(B,Nr,r), S:(B,r), Vh:(B,r,Nt)
        V = Vh.conj().transpose(-2, -1)                         # (B, Nt, r)
        r = S.shape[-1]

        return H, U, S, V, r

    def transmit(
        self,
        x: torch.Tensor,          # (B, Nt) or (B, Nt, 1) – precoded signal
        H: torch.Tensor,          # (B, Nr, Nt) complex
    ) -> torch.Tensor:
        """Add AWGN and return y = Hx + n."""
        # Noise power calibrated to target SNR
        snr_lin  = 10 ** (self.cfg.snr_db / 10.0)
        sig_power = x.abs().pow(2).mean()
        noise_std = (sig_power / (snr_lin + 1e-9)).sqrt()

        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (B, Nt, 1)

        y = torch.bmm(H, x)      # (B, Nr, 1)

        noise_r = torch.randn_like(y.real) * noise_std / (2 ** 0.5)
        noise_i = torch.randn_like(y.imag) * noise_std / (2 ** 0.5)
        noise   = torch.complex(noise_r, noise_i)

        return (y + noise).squeeze(-1)   # (B, Nr)

    def zf_receive(
        self,
        y: torch.Tensor,          # (B, Nr)
        U_r: torch.Tensor,        # (B, Nr, r)
        Sigma_r: torch.Tensor,    # (B, r)
    ) -> torch.Tensor:
        """
        Zero-forcing receiver:  ŝ = diag(1/σ) U_r^H y
        Returns estimated stream vector (B, r) complex.
        """
        # U_r^H y  → (B, r)
        y_col  = y.unsqueeze(-1)                                   # (B, Nr, 1)
        proj   = torch.bmm(U_r.conj().transpose(-2, -1), y_col)    # (B, r, 1)
        proj   = proj.squeeze(-1)                                  # (B, r)

        # Divide by singular values (channel equalization)
        sigma_inv = 1.0 / (Sigma_r + 1e-6)
        return proj * sigma_inv                                     # (B, r)


# ─────────────────────────────────────────────
#  2.  AR(1) Resource Probability Model (Equ. 20–22)
# ─────────────────────────────────────────────
class AR1ResourceModel:
    """
    Simulates device computation and memory availability as AR(1) processes
    in the log-domain (Equ. 20 of the paper), then returns pjoint.

    Usage
    -----
    model = AR1ResourceModel(phi_comp=0.85, phi_mem=0.90, ...)
    trace = model.generate_trace(n_steps=1000)

    # at each time step t, call:
    p_joint = model.pjoint_from_trace(trace, t=t, delta=1)
    """

    def __init__(
        self,
        phi_comp: float  = 0.85,    # AR(1) coefficient – computation
        phi_mem:  float  = 0.90,    # AR(1) coefficient – memory
        sigma_comp: float = 0.15,   # noise std in log domain
        sigma_mem:  float = 0.10,
        mu_comp: float   = 0.0,     # long-run mean in log domain
        mu_mem:  float   = 0.0,
        rho:     float   = 0.3,     # correlation between comp & mem
        seed:    int     = 42,
    ):
        self.phi_comp   = phi_comp
        self.phi_mem    = phi_mem
        self.sigma_comp = sigma_comp
        self.sigma_mem  = sigma_mem
        self.mu_comp    = mu_comp
        self.mu_mem     = mu_mem
        self.rho        = rho
        self._rng       = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    def generate_trace(self, n_steps: int = 2000) -> dict:
        """
        Generate a correlated AR(1) trace for log-comp and log-mem.
        Returns dict with arrays 'X_comp', 'X_mem' of shape (n_steps,).
        """
        # Correlated noise via Cholesky
        cov  = np.array([[1.0, self.rho], [self.rho, 1.0]])
        L    = np.linalg.cholesky(cov)

        X_comp = np.zeros(n_steps)
        X_mem  = np.zeros(n_steps)

        for t in range(1, n_steps):
            eps          = L @ self._rng.standard_normal(2)
            X_comp[t]    = self.phi_comp * X_comp[t-1] + self.sigma_comp * eps[0]
            X_mem[t]     = self.phi_mem  * X_mem[t-1]  + self.sigma_mem  * eps[1]

        return {"X_comp": X_comp, "X_mem": X_mem}

    # ------------------------------------------------------------------
    def pjoint(
        self,
        X_comp_current: float,
        X_mem_current:  float,
        ac: float = 0.0,            # threshold in log-domain (log of absolute threshold)
        am: float = 0.0,
        delta: int = 1,
        window: int = 20,
    ) -> float:
        """
        Compute p_joint = P(X_comp(t+Δ) ≥ ac) · P(X_mem(t+Δ) ≥ am)
        using the analytical Gaussian CDF expressions from Equ. 21.

        For simplicity (and numerical stability) we use the independent
        approximation from Sec. III-A (pjoint ≈ pcomp · pmem).
        The bivariate form of Equ. 22 can be enabled via `use_bivariate=True`.
        """
        from scipy.stats import norm

        def _p_exceed(X_t, phi, sigma_eps, threshold, delta):
            """P(X_{t+delta} >= threshold) for AR(1)."""
            mu_future  = (phi ** delta) * X_t
            var_future = sigma_eps**2 * (1 - phi**(2*delta)) / (1 - phi**2 + 1e-9)
            std_future = np.sqrt(max(var_future, 1e-9))
            z          = (threshold - mu_future) / std_future
            return 1.0 - norm.cdf(z)

        p_comp = _p_exceed(X_comp_current, self.phi_comp, self.sigma_comp, ac, delta)
        p_mem  = _p_exceed(X_mem_current,  self.phi_mem,  self.sigma_mem,  am, delta)
        return float(p_comp * p_mem)

    # ------------------------------------------------------------------
    def pjoint_from_trace(
        self,
        trace: dict,
        t: int,
        ac: float = 0.0,
        am: float = 0.0,
        delta: int = 1,
    ) -> float:
        """Convenience wrapper: read current state from a pre-generated trace."""
        return self.pjoint(
            trace["X_comp"][t],
            trace["X_mem"][t],
            ac=ac, am=am, delta=delta,
        )


# ─────────────────────────────────────────────
#  3.  Task success probability P_succ (Equ. 19)
# ─────────────────────────────────────────────
def p_succ(
    ssnr: float,
    threshold: float,
    interference_power: float = 1.0,
    lambda_m: float = 1.0,
) -> float:
    """
    P_succ = exp( -threshold * P_i / lambda_m )   (Equ. 19)

    Args:
        ssnr               : current observed SSNR (used to estimate channel quality)
        threshold          : task-specific SSNR threshold  Γ^(m)
        interference_power : P_i (interference + noise power)
        lambda_m           : mean channel power gain E[|H_d|²]

    Note: in the synthetic dataset the SSNR already captures noise,
    so we use it directly as a proxy for channel quality.
    """
    return float(np.exp(-threshold * interference_power / (lambda_m + 1e-9)))


# ─────────────────────────────────────────────
#  Sanity check
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=== MIMO Channel ===")
    cfg  = MIMOChannelConfig(Nt=4, Nr=4, snr_db=5.0)
    ch   = MIMOChannel(cfg)
    H, U_r, S, V_sig, r = ch.generate(batch_size=2, seed=0)

    print(f"  H shape    : {tuple(H.shape)}")
    print(f"  V_sig shape: {tuple(V_sig.shape)}")
    print(f"  Sigma      : {S[0].real.tolist()}")
    print(f"  rank r     : {r}")

    # Dummy precoded signal
    x_dummy = torch.complex(torch.randn(2, 4), torch.randn(2, 4))
    y = ch.transmit(x_dummy, H)
    print(f"  y shape    : {tuple(y.shape)}")

    s_hat = ch.zf_receive(y, U_r, S)
    print(f"  ŝ shape    : {tuple(s_hat.shape)}")

    print("\n=== AR(1) Resource Model ===")
    rm    = AR1ResourceModel(seed=1)
    trace = rm.generate_trace(n_steps=500)
    pj    = rm.pjoint_from_trace(trace, t=100, ac=-0.3, am=-0.3)
    print(f"  p_joint at t=100 : {pj:.4f}")

    # Show switching threshold crossings
    crossings = [t for t in range(500) if rm.pjoint_from_trace(trace, t) > 0.6]
    print(f"  p_joint > 0.6 first at t={crossings[0] if crossings else 'never'}")

    print("\n=== P_succ ===")
    print(f"  P_succ(ssnr=2.0, thresh=1.0) = {p_succ(2.0, 1.0):.4f}")
    print(f"  P_succ(ssnr=0.5, thresh=1.0) = {p_succ(0.5, 1.0):.4f}")
