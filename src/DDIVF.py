import numpy as np
from numba import njit


def get_rho(deviations: np.ndarray, signs: np.ndarray) -> float:
    d = deviations - deviations.mean()
    s = signs - signs.mean()
    denom = len(d) * d.std() * s.std()
    if denom == 0:
        return 0.798
    return np.dot(d, s) / denom


@njit
def ddivf_core(V: np.ndarray, alphas: np.ndarray, min_window: int) -> float:
    best_fess = 1e18
    alpha_opt = alphas[0]
    for ai in range(len(alphas)):
        S    = V[0]
        fess = 0.0
        for t in range(1, len(V)):
            err = (V[t] - S) ** 2
            if t >= min_window:
                fess += err
            S = alphas[ai] * V[t] + (1 - alphas[ai]) * S
        if fess < best_fess:
            best_fess = fess
            alpha_opt = alphas[ai]
    return alpha_opt


@njit
def _reconstruct(V: np.ndarray, alpha: float) -> np.ndarray:
    """Reconstruct vol series with optimal alpha — JIT compiled."""
    S    = np.empty(len(V))
    S[0] = V[0]
    for t in range(1, len(V)):
        S[t] = alpha * V[t] + (1 - alpha) * S[t - 1]
    return S

def DDIVF(innovations: np.ndarray, min_window: int = 10) -> tuple:
    nu_bar     = np.mean(innovations)
    deviations = innovations - nu_bar
    signs      = np.sign(deviations)
    mask = signs != 0
    if mask.sum() < 2:
        rho = 0.798
    else:
        rho = get_rho(deviations[mask], signs[mask])
        if abs(rho) < 1e-6:
            rho = 0.798

    V         = np.abs(deviations) / rho
    alphas    = np.arange(0.01, 0.51, 0.01)
    alpha_opt = ddivf_core(V, alphas, min_window)
    S         = _reconstruct(V, alpha_opt)

    return alpha_opt, S, S[-1], rho


# ── JIT warmup — triggers compilation on import, not on first real call ──
_dummy_V      = np.ones(10, dtype=np.float64)
_dummy_alphas = np.arange(0.01, 0.51, 0.01)
ddivf_core(_dummy_V, _dummy_alphas, 5)
_reconstruct(_dummy_V, 0.1)
