"""
afm_modes.py
============
Toy-model solver for the linearized Landau-Lifshitz modes of a
two-sublattice antiferromagnet (easy-plane, with optional DMI,
in-plane anisotropy, cubic anisotropy and thin-film demagnetization).

Procedure:
    1. find the static equilibrium (numerical minimisation, then a Newton
       polish; this is the only unavoidably numerical step),
    2. build the *analytical* energy Hessian in the tangent plane of each
       sublattice,
    3. linearize the LLG equation and diagonalise a small 4x4 matrix.

Energy density (J/m^3), each sublattice fills the whole volume, same Ms:

    E = sum_i [ -Ms B.m_i
                + Kh  m_iz^2                       (easy-plane: z is hard)
                - Kip (m_i.u)^2                    (in-plane uniaxial)
                + K1 (mx^2 my^2 + my^2 mz^2 + mz^2 mx^2) ]   (cubic)
        + lam  m1.m2                               (AFM exchange, lam > 0)
        + D . (m1 x m2)                            (DMI, homogeneous)
        + 0.5 mu0 Ms^2 Nz (m1z + m2z)^2            (demagnetizing, Nz=1 film)
        + gradient terms:  A[(grad m1)^2 + (grad m2)^2] + A12 grad m1.grad m2

Linearized LLG for each sublattice:  dm_i/dt = -gamma m_i x B_eff,i,
B_eff,i = -(1/Ms) dE/dm_i. With delta m_i = a_i e1_i + b_i e2_i
(e1 x e2 = m_i) this becomes

        dq/dt = (gamma/Ms) J H q ,   q = (a1, b1, a2, b2),
        J = diag([[0,-1],[1,0]], [[0,-1],[1,0]]),

H = constrained (tangent-plane) energy Hessian. Positive eigenvalue
i*omega gives the mode e^{+i omega t}.

Simplifications / things NOT included (on purpose, for the toy model):
  * plane-wave (bulk-like) modes, no thickness-mode basis, no dynamic
    dipolar kernel (negligible for an AFM: no net moment). Nz only
    contributes the static, k-independent shape term.
  * no damping (conservative modes).
  * the Hessian depends on k only through |k|. With inhomogeneous
    (Lifshitz-type) DMI or dipolar terms, direction would matter.

Stiffness reading of the gradient terms (Hessian units):
    pure Neel-vector (delta m1 = -delta m2) mode:   (2A - A12) k^2
    pure net-moment (delta m1 = +delta m2) mode:    (2A + A12) k^2
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

from scipy.optimize import minimize, linear_sum_assignment


# ============================================================
# Constants
# ============================================================

MU0 = 4.0 * np.pi * 1e-7
GAMMA = 1.76085963023e11  # rad s^-1 T^-1  (magnitude)

LEVI = np.zeros((3, 3, 3))
LEVI[0, 1, 2] = LEVI[1, 2, 0] = LEVI[2, 0, 1] = 1.0
LEVI[0, 2, 1] = LEVI[2, 1, 0] = LEVI[1, 0, 2] = -1.0


# ============================================================
# Parameters
# ============================================================

@dataclass
class AFMParams:
    """
    Material parameters (SI). All energies in J/m^3, stiffnesses in J/m.

    Ms     : sublattice saturation magnetization (A/m)
    lam    : homogeneous inter-sublattice exchange, E = lam m1.m2
    A      : intra-sublattice exchange stiffness
    A12    : inter-sublattice gradient coupling, E = A12 grad m1.grad m2
    Kh     : hard-axis (easy-plane) anisotropy, E = Kh mz^2
    Kip    : in-plane uniaxial anisotropy, E = -Kip (m.u)^2,
             u = (cos phi_ip, sin phi_ip, 0)
    K1     : cubic anisotropy
    D      : DMI vector, E = D.(m1 x m2)   (J/m^3)
    Nz     : out-of-plane demag factor (0 = bulk, 1 = thin film)
    """
    Ms: float
    lam: float
    A: float = 0.0
    A12: float = 0.0
    Kh: float = 0.0
    Kip: float = 0.0
    phi_ip: float = 0.0
    K1: float = 0.0
    D: tuple = (0.0, 0.0, 0.0)
    Nz: float = 0.0

    @classmethod
    def from_fields(cls, Ms, BE, Bh=0.0, Bip=0.0, BD=0.0,
                    D_axis=(0.0, 0.0, 1.0), **kw):
        """
        Build parameters from characteristic fields in tesla:

            BE  = lam/Ms         exchange field
            Bh  = 2 Kh/Ms        hard-axis (easy-plane) anisotropy field
            Bip = 2 Kip/Ms       in-plane anisotropy field
            BD  = |D|/Ms         DMI field (D along D_axis, default z)

        Handy because textbook AFMR formulas are written with these.
        """
        D = Ms * BD * normalize(D_axis)
        return cls(Ms=Ms, lam=Ms * BE, Kh=Ms * Bh / 2.0,
                   Kip=Ms * Bip / 2.0, D=tuple(D), **kw)

    @property
    def BE(self):
        return self.lam / self.Ms

    @property
    def BD(self):
        return np.linalg.norm(self.D) / self.Ms


# ============================================================
# Geometry
# ============================================================

def normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("Cannot normalize a zero vector.")
    return v / n


def tangent_basis(m):
    """Orthonormal (e1, e2) with e1 x e2 = m."""
    m = normalize(m)
    ref = np.array([0.0, 0.0, 1.0])
    if abs(ref @ m) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    e1 = normalize(np.cross(ref, m))
    e2 = normalize(np.cross(m, e1))
    return e1, e2


def block_diag2(a, b):
    n, m = a.shape[0], b.shape[0]
    out = np.zeros((n + m, a.shape[1] + b.shape[1]), dtype=np.result_type(a, b))
    out[:n, :a.shape[1]] = a
    out[n:, a.shape[1]:] = b
    return out


# ============================================================
# Energy, gradient, Hessian (Cartesian, analytic)
# ============================================================

def sublattice_terms(m, p):
    """Single-sublattice anisotropy terms: E, grad (3), Hessian (3x3)."""
    x, y, z = m
    u = np.array([np.cos(p.phi_ip), np.sin(p.phi_ip), 0.0])

    E = 0.0
    g = np.zeros(3)
    Q = np.zeros((3, 3))

    # hard axis (easy plane)
    E += p.Kh * z**2
    g[2] += 2 * p.Kh * z
    Q[2, 2] += 2 * p.Kh

    # in-plane uniaxial
    mu = m @ u
    E += -p.Kip * mu**2
    g += -2 * p.Kip * mu * u
    Q += -2 * p.Kip * np.outer(u, u)

    # cubic
    s = m**2
    S = s.sum()
    E += p.K1 * (s[0] * s[1] + s[1] * s[2] + s[2] * s[0])
    g += 2 * p.K1 * m * (S - s)
    for i in range(3):
        Q[i, i] += 2 * p.K1 * (S - s[i])
        for j in range(3):
            if i != j:
                Q[i, j] += 4 * p.K1 * m[i] * m[j]

    return E, g, Q


def energy_grad_hessian(m1, m2, B, p):
    """
    Total energy density, gradient (6,) and Cartesian Hessian (6x6)
    with respect to (m1, m2), ignoring the unit-length constraint.
    """
    B = np.asarray(B, dtype=float)
    D = np.asarray(p.D, dtype=float)

    E1, g1, Q1 = sublattice_terms(m1, p)
    E2, g2, Q2 = sublattice_terms(m2, p)

    E = E1 + E2
    g = np.concatenate([g1, g2])
    Q = block_diag2(Q1, Q2)
    Q12 = np.zeros((3, 3))

    # Zeeman
    E += -p.Ms * B @ (m1 + m2)
    g[:3] += -p.Ms * B
    g[3:] += -p.Ms * B

    # AFM exchange
    E += p.lam * m1 @ m2
    g[:3] += p.lam * m2
    g[3:] += p.lam * m1
    Q12 += p.lam * np.eye(3)

    # DMI:  D.(m1 x m2) = m1.(m2 x D) = m2.(D x m1)
    E += D @ np.cross(m1, m2)
    g[:3] += np.cross(m2, D)
    g[3:] += np.cross(D, m1)
    Q12 += np.einsum("abc,a->bc", LEVI, D)

    # demag of the *net* moment (thin film)
    if p.Nz != 0.0:
        c = MU0 * p.Ms**2 * p.Nz
        mz = m1[2] + m2[2]
        E += 0.5 * c * mz**2
        g[2] += c * mz
        g[5] += c * mz
        Q[2, 2] += c
        Q[5, 5] += c
        Q12[2, 2] += c

    Q[:3, 3:] = Q12
    Q[3:, :3] = Q12.T
    return E, g, Q


def tangent_hessian(m1, m2, B, p):
    """Constrained 4x4 Hessian in tangent coordinates (a1,b1,a2,b2)."""
    e11, e12 = tangent_basis(m1)
    e21, e22 = tangent_basis(m2)
    T1 = np.column_stack([e11, e12])
    T2 = np.column_stack([e21, e22])
    T = block_diag2(T1, T2)

    _, g, Q = energy_grad_hessian(m1, m2, B, p)
    lam1 = g[:3] @ m1
    lam2 = g[3:] @ m2
    L = block_diag2(lam1 * np.eye(3), lam2 * np.eye(3))

    H = T.T @ (Q - L) @ T
    gt = T.T @ (g - np.concatenate([lam1 * m1, lam2 * m2]))
    return H, gt, T, T1, T2


# ============================================================
# Equilibrium (the one numerical step)
# ============================================================

def _newton_polish(m1, m2, B, p, n_iter=30):
    for _ in range(n_iter):
        H, gt, T, _, _ = tangent_hessian(m1, m2, B, p)
        if np.linalg.norm(gt) < 1e-13 * max(1.0, p.Ms):
            break
        step = -np.linalg.pinv(H, rcond=1e-10) @ gt
        # damp very large steps
        nrm = np.linalg.norm(step)
        if nrm > 0.3:
            step *= 0.3 / nrm
        d = T @ step
        m1 = normalize(m1 + d[:3])
        m2 = normalize(m2 + d[3:])
    return m1, m2


def find_equilibrium(B, p, n_starts=12, guess=None, follow=False, seed=1234):
    """
    Find the equilibrium (m1, m2).

    follow=False : global minimum over many starts (thermodynamic ground
                   state).
    follow=True  : local minimum nearest to `guess` (follows a metastable
                   branch, e.g. through a spin-flop hysteresis).

    Energy is divided by Ms internally (units of tesla) for conditioning.
    """
    B = np.asarray(B, dtype=float)

    def fun(x):
        E, g, _ = energy_grad_hessian(x[:3], x[3:], B, p)
        return E / p.Ms, g / p.Ms

    cons = [{
        "type": "eq",
        "fun": lambda x: np.array([x[:3] @ x[:3] - 1.0, x[3:] @ x[3:] - 1.0]),
        "jac": lambda x: np.array([
            np.concatenate([2 * x[:3], np.zeros(3)]),
            np.concatenate([np.zeros(3), 2 * x[3:]]),
        ]),
    }]

    # fast path for branch following: plain Newton from the previous state
    if follow and guess is not None:
        m1, m2 = _newton_polish(normalize(guess[0]), normalize(guess[1]), B, p)
        gt = tangent_hessian(m1, m2, B, p)[1]
        if np.linalg.norm(gt) < 1e-9 * p.Ms:
            return m1, m2

    starts = []
    if guess is not None:
        starts.append(np.concatenate([normalize(guess[0]), normalize(guess[1])]))

    if not (follow and guess is not None):
        axes = np.vstack([np.eye(3), -np.eye(3)])
        for a in axes:
            starts.append(np.concatenate([a, -a]))
        if np.linalg.norm(B) > 0:
            b = normalize(B)
            starts.append(np.concatenate([b, b]))
            # spin-flop-like: Neel vector perpendicular to B
            for a in axes:
                perp = a - (a @ b) * b
                if np.linalg.norm(perp) > 1e-6:
                    n = normalize(perp)
                    starts.append(np.concatenate([normalize(n + 0.3 * b),
                                                  normalize(-n + 0.3 * b)]))
        rng = np.random.default_rng(seed)
        for _ in range(n_starts):
            n = normalize(rng.normal(size=3))
            starts.append(np.concatenate([
                normalize(n + 0.3 * rng.normal(size=3)),
                normalize(-n + 0.3 * rng.normal(size=3)),
            ]))

    best = None
    for x0 in starts:
        r = minimize(fun, x0, jac=True, method="SLSQP", constraints=cons,
                     options={"ftol": 1e-12, "maxiter": 150})
        if not np.all(np.isfinite(r.x)):
            continue
        m1, m2 = normalize(r.x[:3]), normalize(r.x[3:])
        m1, m2 = _newton_polish(m1, m2, B, p)
        E = energy_grad_hessian(m1, m2, B, p)[0]
        if best is None or E < best[0] - 1e-12 * abs(p.Ms):
            best = (E, m1, m2)
        if follow and guess is not None:
            break

    if best is None:
        raise RuntimeError("Equilibrium minimisation failed.")
    return best[1], best[2]


# ============================================================
# Mode solver
# ============================================================

def solve_modes(k, B, p, equilibrium=None, guess=None, follow=False):
    """
    Solve the two AFM modes at wavevector k (scalar |k| in rad/m, or a
    3-vector) and field B (3-vector, tesla).

    Parameters
    ----------
    equilibrium : (m1, m2), optional
        Skip the minimisation (useful for f(k) sweeps).
    guess, follow : see find_equilibrium.

    Returns dict with
        frequency        (2,) Hz, ascending
        eigenvectors     (4,2) tangent coordinates (a1,b1,a2,b2)
        cartesian        (6,2) delta m1, delta m2 in Cartesian components
        equilibrium      (m1, m2)
        Hessian, dynamical_matrix, tangent (T1, T2), hessian_min_eig
    """
    B = np.asarray(B, dtype=float)
    k2 = float(np.dot(k, k)) if np.ndim(k) else float(k) ** 2

    def build_H(m1, m2):
        H, _, T, T1, T2 = tangent_hessian(m1, m2, B, p)
        # gradient (exchange) terms: analytic, plane wave
        I2 = np.eye(2)
        Hk = np.block([
            [2 * p.A * k2 * I2, p.A12 * k2 * (T1.T @ T2)],
            [p.A12 * k2 * (T2.T @ T1), 2 * p.A * k2 * I2],
        ])
        H = H + Hk
        H = 0.5 * (H + H.T)
        return H, T, T1, T2

    def unstable(H):
        return np.linalg.eigvalsh(H)[0] < -1e-8 * max(np.abs(H).max(), 1.0)

    if equilibrium is None:
        m1, m2 = find_equilibrium(B, p, guess=guess, follow=follow)
    else:
        m1, m2 = equilibrium

    H, T, T1, T2 = build_H(m1, m2)

    # a followed branch can turn into a saddle (end of metastability):
    # jump to the global minimum instead.
    if equilibrium is None and follow and unstable(H):
        m1, m2 = find_equilibrium(B, p)
        H, T, T1, T2 = build_H(m1, m2)

    hmin = np.linalg.eigvalsh(H)[0]
    if unstable(H):
        print(f"Warning: Hessian not positive semi-definite at B={B}, k={k} "
              f"(min eig {hmin:.3e}); equilibrium may be unstable.")

    J2 = np.array([[0.0, -1.0], [1.0, 0.0]])
    J = np.kron(np.eye(2), J2)
    Dm = (GAMMA / p.Ms) * J @ H

    w, V = np.linalg.eig(Dm)

    # eigenvalues come in +-i*omega pairs; keep the two largest Im parts.
    # (works also when a Goldstone mode sits at omega = 0)
    idx = np.argsort(np.imag(w))[::-1][:2]
    idx = idx[np.argsort(np.imag(w[idx]))]

    freq = np.maximum(np.imag(w[idx]), 0.0) / (2 * np.pi)
    vecs = V[:, idx].copy()

    for j in range(2):
        v = vecs[:, j]
        v = v / np.linalg.norm(v)
        v = v * np.exp(-1j * np.angle(v[np.argmax(np.abs(v))]))
        vecs[:, j] = v

    cart = T @ vecs

    return {
        "frequency": freq,
        "eigenvalues": w[idx],
        "eigenvectors": vecs,
        "cartesian": cart,
        "equilibrium": (m1, m2),
        "tangent": (T1, T2),
        "Hessian": H,
        "dynamical_matrix": Dm,
        "hessian_min_eig": hmin,
        "B": B,
        "k": k,
        "params": p,
    }


# ============================================================
# Mode tracking
# ============================================================

def track_modes(results):
    """
    Order branches continuously along a sweep using the overlap of the
    *Cartesian* eigenvectors (tangent-plane coordinates are not
    comparable between points because the tangent basis moves).
    """
    freqs = np.array([r["frequency"] for r in results])
    vecs = np.stack([r["cartesian"] for r in results])  # (n, 6, 2)

    for i in range(1, len(results)):
        overlap = np.abs(vecs[i - 1].conj().T @ vecs[i])
        row, col = linear_sum_assignment(-overlap)
        perm = np.zeros(len(col), dtype=int)
        perm[row] = col
        freqs[i] = freqs[i, perm]
        vecs[i] = vecs[i][:, perm]
    return freqs, vecs


# ============================================================
# Sweeps
# ============================================================

def sweep_field(fields, field_direction, p, k=0.0, follow=True):
    """
    f(B) at fixed k along a fixed field direction.

    follow=True : follow the metastable branch from the first field
                  (sweep direction matters -> hysteresis at spin-flop).
                  Pass fields in decreasing order for the down-sweep.
    follow=False: always use the global energy minimum.
    """
    d = normalize(field_direction)
    results = []
    guess = None
    for Bmag in fields:
        r = solve_modes(k, Bmag * d, p, guess=guess, follow=(follow and guess is not None))
        guess = r["equilibrium"]
        results.append(r)
    freqs, vecs = track_modes(results)
    return freqs, vecs, results


def sweep_field_angle(angles, B_mag, rotation_plane, p, k=0.0, follow=True):
    """
    f(theta) at fixed |B| and k, rotating B in a plane.
    rotation_plane = 'xy' (in-plane) or 'xz' or 'yz'; angle from the
    first axis towards the second.
    """
    ax = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}[rotation_plane]
    results = []
    guess = None
    for th in angles:
        d = np.zeros(3)
        d[ax[0]] = np.cos(th)
        d[ax[1]] = np.sin(th)
        r = solve_modes(k, B_mag * d, p, guess=guess, follow=(follow and guess is not None))
        guess = r["equilibrium"]
        results.append(r)
    freqs, vecs = track_modes(results)
    return freqs, vecs, results


def sweep_wavevector(k_values, B, p):
    """f(k) at fixed B (equilibrium is k-independent: computed once)."""
    B = np.asarray(B, dtype=float)
    eq = find_equilibrium(B, p)
    results = [solve_modes(k, B, p, equilibrium=eq) for k in k_values]
    freqs, vecs = track_modes(results)
    return freqs, vecs, results


# ============================================================
# Mode characterisation
# ============================================================

def describe_mode(result, mode_index):
    """
    Physical character of one mode.

    l  = (m1 - m2)/2 : Neel vector,   m = (m1 + m2)/2 : net moment.
    Returns equilibrium l0, m0 and the complex dynamic amplitudes dl, dm
    (Cartesian), the l-vs-m weight, and the precession sense of each
    sublattice about its own equilibrium direction
    (+1 = right-handed about m_i, -1 = left-handed).
    """
    m1, m2 = result["equilibrium"]
    v = result["cartesian"][:, mode_index]
    d1, d2 = v[:3], v[3:]
    dl = 0.5 * (d1 - d2)
    dm = 0.5 * (d1 + d2)

    def sense(m0, d):
        # delta(t) = Re[d e^{i w t}]: R = Re d, V = -w Im d
        return int(np.sign(-m0 @ np.cross(d.real, d.imag)))

    return {
        "frequency_GHz": result["frequency"][mode_index] / 1e9,
        "l0": 0.5 * (m1 - m2),
        "m0": 0.5 * (m1 + m2),
        "dl": dl,
        "dm": dm,
        "weight_l": np.linalg.norm(dl) ** 2 / (np.linalg.norm(dl) ** 2 + np.linalg.norm(dm) ** 2),
        "sense_1": sense(m1, d1),
        "sense_2": sense(m2, d2),
    }


# ============================================================
# Plotting
# ============================================================

def plot_frequency_sweep(x, frequencies, xlabel, ylabel="Frequency (GHz)",
                         title=None, labels=None, ax=None, **kw):
    """Plot tracked branches. Pass ax= to overlay e.g. up/down sweeps."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
    else:
        fig = ax.figure
    for n in range(frequencies.shape[1]):
        lab = labels[n] if labels is not None else f"Mode {n}"
        ax.plot(x, frequencies[:, n] / 1e9, lw=1.8, label=lab, **kw)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_equilibrium(x, results, xlabel):
    """In-plane angle of l and m, out-of-plane tilt, and canting vs sweep."""
    l0 = np.array([0.5 * (r["equilibrium"][0] - r["equilibrium"][1]) for r in results])
    m0 = np.array([0.5 * (r["equilibrium"][0] + r["equilibrium"][1]) for r in results])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(x, np.degrees(np.arctan2(l0[:, 1], l0[:, 0])), label=r"in-plane angle of $\mathbf{l}$")
    axes[0].plot(x, np.degrees(np.arcsin(np.clip(l0[:, 2] / np.linalg.norm(l0, axis=1), -1, 1))),
                 label=r"tilt of $\mathbf{l}$ out of plane")
    axes[0].set_ylabel("deg")
    axes[1].plot(x, m0[:, 0], label=r"$m_x$")
    axes[1].plot(x, m0[:, 1], label=r"$m_y$")
    axes[1].plot(x, m0[:, 2], label=r"$m_z$")
    axes[1].set_ylabel(r"net moment $(\mathbf{m}_1+\mathbf{m}_2)/2$")
    for ax in axes:
        ax.set_xlabel(xlabel)
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.tight_layout()
    return fig, axes


def plot_mode(result, mode_index=0):
    """
    Left : |components| of the dynamic Neel vector dl and net moment dm.
    Right: precession ellipse of each sublattice in its own tangent plane
           (dot = t=0, arrow shows rotation sense).
    """
    info = describe_mode(result, mode_index)
    scale = max(np.abs(info["dl"]).max(), np.abs(info["dm"]).max())

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    xs = np.arange(3)
    axes[0].bar(xs - 0.2, np.abs(info["dl"]) / scale, 0.4, label=r"$|\delta l|$")
    axes[0].bar(xs + 0.2, np.abs(info["dm"]) / scale, 0.4, label=r"$|\delta m|$")
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels(["x", "y", "z"])
    axes[0].legend()
    axes[0].set_title(f"f = {info['frequency_GHz']:.3f} GHz,  l-weight {info['weight_l']:.2f}")

    v = result["eigenvectors"][:, mode_index]
    t = np.linspace(0, 2 * np.pi, 200)
    for s, ax in enumerate(axes[1:]):
        a, b = v[2 * s], v[2 * s + 1]
        # delta(t) = Re[v e^{i w t}]
        X = np.real(a * np.exp(1j * t))
        Y = np.real(b * np.exp(1j * t))
        ax.plot(X, Y)
        ax.plot(X[0], Y[0], "o")
        ax.annotate("", xy=(X[8], Y[8]), xytext=(X[0], Y[0]),
                    arrowprops=dict(arrowstyle="->"))
        ax.set_aspect("equal")
        ax.set_title(f"sublattice {s + 1}: tangent plane (e1,e2)")
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig, axes


# ============================================================
# Analytical formulas (k = 0)
# ============================================================

def afmr_analytic_zero_field(p):
    """
    k = 0, B = 0, D = 0, Nz = 0, l along the in-plane easy axis.
    With B_E = lam/Ms, B_A = 2Kip/Ms, B_h = 2Kh/Ms:

        in-plane  (l rotates in the plane, Goldstone mode if Kip=0):
            f = gamma/2pi sqrt( B_A (2B_E + B_A + B_h) )
        out-of-plane (l tilts out of the plane):
            f = gamma/2pi sqrt( (B_A + B_h)(2B_E + B_A) )
    """
    BE = p.lam / p.Ms
    BA = 2 * p.Kip / p.Ms
    Bh = 2 * p.Kh / p.Ms
    f_ip = GAMMA / (2 * np.pi) * np.sqrt(BA * (2 * BE + BA + Bh))
    f_oop = GAMMA / (2 * np.pi) * np.sqrt((BA + Bh) * (2 * BE + BA))
    return f_ip, f_oop


def dmi_analytic_zero_field(p):
    """
    k = 0, B = 0, Kip = K1 = 0, Nz = 0, D = D z-hat (hematite-like).
    Weak ferromagnetism: moments cant in plane by tan(eta) = D/lam, and
        Lambda = sqrt(lam^2 + D^2)
        in-plane mode      f = 0                     (Goldstone)
        out-of-plane mode  f = gamma/(2 pi Ms) sqrt( 2 Lambda (Lambda - lam + 2 Kh) )
    """
    D = np.linalg.norm(p.D)
    Lam = np.hypot(p.lam, D)
    f_oop = GAMMA / (2 * np.pi * p.Ms) * np.sqrt(2 * Lam * (Lam - p.lam + 2 * p.Kh))
    return 0.0, f_oop


# ============================================================
# Validation
# ============================================================

def validate_afmr_limit():
    """Zero-field AFMR of an easy-plane AFM with in-plane easy axis, no DMI."""
    p = AFMParams.from_fields(Ms=2e5, BE=10.0, Bh=0.1, Bip=0.01)
    r = solve_modes(0.0, np.zeros(3), p)
    f_ip, f_oop = afmr_analytic_zero_field(p)
    f_calc = np.sort(r["frequency"])

    print("AFMR zero-field validation (no DMI)")
    print("-----------------------------------")
    print(f"  in-plane    expected {f_ip/1e9:9.5f} GHz")
    print(f"  out-of-pl.  expected {f_oop/1e9:9.5f} GHz")
    print(f"  calculated           {f_calc[0]/1e9:9.5f}, {f_calc[1]/1e9:9.5f} GHz")
    exp = np.sort([f_ip, f_oop])
    print("  relative error:", np.abs(f_calc - exp) / exp)
    return r


def validate_dmi_limit():
    """Easy-plane AFM with DMI along z, isotropic in plane, B = 0."""
    p = AFMParams.from_fields(Ms=2e5, BE=10.0, Bh=0.1, BD=0.5)
    r = solve_modes(0.0, np.zeros(3), p)
    f0, f_oop = dmi_analytic_zero_field(p)
    f_calc = np.sort(r["frequency"])

    m1, m2 = r["equilibrium"]
    cant = np.degrees(np.arccos(np.clip(m1 @ m2, -1, 1)))
    cant_exp = 180.0 - np.degrees(np.arctan(np.linalg.norm(p.D) / p.lam))

    print("DMI zero-field validation")
    print("-------------------------")
    print(f"  angle between sublattices: {cant:.5f} deg (expected {cant_exp:.5f})")
    print(f"  out-of-plane mode expected {f_oop/1e9:.5f} GHz, "
          f"calculated {f_calc[1]/1e9:.5f} GHz")
    print(f"  Goldstone mode: {f_calc[0]/1e9:.2e} GHz")
    return r


def validate_spin_flop_state():
    """
    Easy-plane AFM, isotropic in plane, B in plane: the Neel vector goes
    perpendicular to B and the net moment is B/(2 B_E).
    """
    p = AFMParams.from_fields(Ms=2e5, BE=10.0, Bh=0.1)
    B = np.array([2.0, 0.0, 0.0])
    r = solve_modes(0.0, B, p)
    m1, m2 = r["equilibrium"]
    m = 0.5 * (m1 + m2)
    print("Field-induced canting validation")
    print("--------------------------------")
    print(f"  |m| calculated {np.linalg.norm(m):.6f}, expected {2.0/(2*10.0):.6f}")
    print(f"  l . B = {0.5*(m1-m2) @ normalize(B):.2e} (should be 0)")
    return r
