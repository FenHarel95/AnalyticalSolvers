
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.special import roots_legendre
from scipy.optimize import linear_sum_assignment
from itertools import product


# ============================================================
# Constants
# ============================================================

MU0 = 4.0 * np.pi * 1e-7
GAMMA = 1.76085963023e11  # rad s^-1 T^-1


# ============================================================
# Geometry and anisotropy
# ============================================================

def normalize(v):
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v)
    if norm == 0:
        raise ValueError("Cannot normalize a zero vector.")
    return v / norm


def tangent_basis(m):
    """
    Construct two orthonormal vectors perpendicular to m.

    Returns
    -------
    e1, e2 : ndarray, shape (3,)
        Right-handed local tangent basis:
        e1 x e2 = m.
    """
    m = normalize(m)

    # Choose a reference vector not parallel to m
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(ref, m)) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])

    e1 = normalize(np.cross(ref, m))
    e2 = normalize(np.cross(m, e1))

    return e1, e2


def local_energy_gradient_hessian(
    m,
    B,
    Ms,
    Ku=0.0,
    K1=0.0,
):
    """
    Homogeneous magnetic energy density, gradient and Hessian.

    Energy:
        E = -Ms B.m
            + Ku (1 - mz^2)
            + K1 (mx^2 my^2 + my^2 mz^2 + mz^2 mx^2)
            + 0.5 mu0 Ms^2 mz^2

    The final term is the static demagnetizing energy of an
    infinitely extended thin film.

    Parameters
    ----------
    m : ndarray
        Unit magnetization vector.
    B : ndarray
        Applied magnetic flux density in T.
    Ms : float
        Saturation magnetization in A/m.
    Ku, K1 : float
        Anisotropy constants in J/m^3.

    Returns
    -------
    energy : float
    grad : ndarray, shape (3,)
    Hessian : ndarray, shape (3,3)
    """
    x, y, z = m
    Bx, By, Bz = B

    # Zeeman energy
    energy = -Ms * np.dot(B, m)
    grad = -Ms * np.asarray(B, dtype=float)
    Q = np.zeros((3, 3))

    # Perpendicular uniaxial anisotropy
    energy += Ku * (1.0 - z**2)

    grad[2] += -2.0 * Ku * z
    Q[2, 2] += -2.0 * Ku

    # Cubic anisotropy
    energy += K1 * (
        x**2 * y**2
        + y**2 * z**2
        + z**2 * x**2
    )

    grad[0] += 2.0 * K1 * x * (y**2 + z**2)
    grad[1] += 2.0 * K1 * y * (x**2 + z**2)
    grad[2] += 2.0 * K1 * z * (x**2 + y**2)

    Q[0, 0] += 2.0 * K1 * (y**2 + z**2)
    Q[1, 1] += 2.0 * K1 * (x**2 + z**2)
    Q[2, 2] += 2.0 * K1 * (x**2 + y**2)

    Q[0, 1] += 4.0 * K1 * x * y
    Q[1, 0] += 4.0 * K1 * x * y

    Q[0, 2] += 4.0 * K1 * x * z
    Q[2, 0] += 4.0 * K1 * x * z

    Q[1, 2] += 4.0 * K1 * y * z
    Q[2, 1] += 4.0 * K1 * y * z

    # Static thin-film demagnetizing energy
    energy += 0.5 * MU0 * Ms**2 * z**2

    grad[2] += MU0 * Ms**2 * z
    Q[2, 2] += MU0 * Ms**2

    return energy, grad, Q


def find_equilibrium(
    B,
    Ms,
    Ku=0.0,
    K1=0.0,
    n_starts=30,
):
    """
    Find a low-energy equilibrium magnetization direction.

    Uses multiple constrained minimizations on the unit sphere.
    This is useful when anisotropy produces multiple local minima.

    Returns
    -------
    m0 : ndarray, shape (3,)
    """
    B = np.asarray(B, dtype=float)

    def objective(m):
        return local_energy_gradient_hessian(
            m, B, Ms, Ku, K1
        )[0]

    def constraint(m):
        return np.dot(m, m) - 1.0

    constraints = [{"type": "eq", "fun": constraint}]

    candidates = [
        normalize(B) if np.linalg.norm(B) > 0 else np.array([1., 0., 0.]),
        np.array([1., 0., 0.]),
        np.array([-1., 0., 0.]),
        np.array([0., 1., 0.]),
        np.array([0., -1., 0.]),
        np.array([0., 0., 1.]),
        np.array([0., 0., -1.]),
    ]

    rng = np.random.default_rng(1234)

    for _ in range(n_starts):
        candidates.append(normalize(rng.normal(size=3)))

    results = []

    for guess in candidates:
        result = minimize(
            objective,
            guess,
            method="SLSQP",
            constraints=constraints,
            options={
                "ftol": 1e-12,
                "maxiter": 500,
            },
        )

        if result.success:
            results.append(result)

    if not results:
        raise RuntimeError("Equilibrium minimization failed.")

    best = min(results, key=lambda r: r.fun)

    return normalize(best.x)


# ============================================================
# Free-exchange thickness-mode basis
# ============================================================

class ThicknessBasis:
    """
    Analytical free-exchange basis for a film of thickness d.

    phi_0(z) = 1/sqrt(d)

    phi_n(z) = sqrt(2/d) cos[n*pi*(z+d/2)/d]

    These modes satisfy:
        d(phi_n)/dz = 0
        at z = +/- d/2.
    """

    def __init__(self, d, n_modes=3):
        self.d = float(d)
        self.n_modes = int(n_modes)

        if self.d <= 0:
            raise ValueError("Film thickness must be positive.")

        if self.n_modes < 1:
            raise ValueError("At least one mode is required.")

        self.n = np.arange(self.n_modes)
        self.qz = self.n * np.pi / self.d

    def phi(self, z):
        """
        Evaluate all basis functions.

        Parameters
        ----------
        z : ndarray
            Positions in metres.

        Returns
        -------
        phi : ndarray
            Shape (n_modes, len(z)).
        """
        z = np.atleast_1d(z)
        u = (z + self.d / 2.0) / self.d

        phi = np.zeros((self.n_modes, len(z)))

        phi[0] = 1.0 / np.sqrt(self.d)

        for n in range(1, self.n_modes):
            phi[n] = (
                np.sqrt(2.0 / self.d)
                * np.cos(n * np.pi * u)
            )

        return phi


# ============================================================
# Magnetostatic Green kernel
# ============================================================

def dipolar_matrix(
    kx,
    ky,
    basis,
    n_quad=100,
):
    """
    Calculate the full projected magnetostatic tensor.

    Fourier convention:
        exp[i(kx*x + ky*y - omega*t)]

    The magnetostatic field is:
        H_d = -N.M

    Tensor components:
        N_ab = ka*kb/(2k) exp(-k|z-z'|)
        N_az = -i ka/2 sign(z-z') exp(-k|z-z'|)
        N_za = +i ka/2 sign(z-z') exp(-k|z-z'|)
        N_zz = delta(z-z') - k/2 exp(-k|z-z'|)

    Returns
    -------
    N : ndarray, shape (n_modes, n_modes, 3, 3)

    The first two indices are thickness-mode indices.
    The last two are Cartesian components.
    """
    d = basis.d
    nm = basis.n_modes

    k = np.sqrt(kx**2 + ky**2)

    if k < 1e-12:
        N = np.zeros((nm, nm, 3, 3))

        # At k=0, only the perpendicular demagnetizing
        # component remains.
        for n in range(nm):
            N[n, n, 2, 2] = 1.0

        return N

    # Gauss-Legendre quadrature over the film thickness
    nodes, weights = roots_legendre(n_quad)

    z = nodes * d / 2.0
    w = weights * d / 2.0

    phi = basis.phi(z)

    # Pairwise z-z'
    dz = z[:, None] - z[None, :]

    abs_dz = np.abs(dz)
    sign_dz = np.sign(dz)

    exponential = np.exp(-k * abs_dz)

    # Integral weights, including basis functions
    weighted_phi = phi * w[None, :]

    # Four-index integration:
    # integral dz dz' phi_n(z) kernel(z,z') phi_m(z')
    N = np.zeros((nm, nm, 3, 3), dtype=complex)

    for n in range(nm):
        for m in range(nm):

            integrand = (
                weighted_phi[n, :, None]
                * weighted_phi[m, None, :]
            )

            # In-plane components
            for a, ka in enumerate([kx, ky]):
                for b, kb in enumerate([kx, ky]):
                    kernel = (
                        ka * kb / (2.0 * k)
                        * exponential
                    )

                    N[n, m, a, b] = np.sum(
                        integrand * kernel
                    )

                # In-plane / perpendicular components
                kernel_az = (
                    -1j * ka / 2.0
                    * sign_dz * exponential
                )

                kernel_za = (
                    +1j * ka / 2.0
                    * sign_dz * exponential
                )

                N[n, m, a, 2] = np.sum(
                    integrand * kernel_az
                )

                N[n, m, 2, a] = np.sum(
                    integrand * kernel_za
                )

            # Smooth part of Nzz
            kernel_zz = (
                -k / 2.0 * exponential
            )

            N[n, m, 2, 2] = np.sum(
                integrand * kernel_zz
            )

    # Delta-function part of Nzz:
    # integral phi_n(z) phi_m(z) dz = delta_nm
    for n in range(nm):
        N[n, n, 2, 2] += 1.0

    return N


# ============================================================
# Modal energy Hessian
# ============================================================

def build_energy_hessian(
    kx,
    ky,
    B,
    Ms,
    A,
    basis,
    Ku=0.0,
    K1=0.0,
    n_quad=100,
):
    """
    Build the 2*n_modes x 2*n_modes modal energy Hessian.

    State ordering:
        (a0, b0, a1, b1, a2, b2, ...)

    Dynamic magnetization:
        delta_m(z) = sum_n [a_n e1 + b_n e2] phi_n(z)

    Units of H:
        J/m^3 in the normalized modal coordinates.
    """
    nm = basis.n_modes

    # Equilibrium direction
    m0 = find_equilibrium(
        B, Ms, Ku=Ku, K1=K1
    )

    e1, e2 = tangent_basis(m0)
    T = np.column_stack([e1, e2])

    # Local energy derivatives, including static demagnetization
    _, grad, Q = local_energy_gradient_hessian(
        m0, B, Ms, Ku, K1
    )

    # Lagrange multiplier for the unit-vector constraint
    lam = np.dot(grad, m0)

    # Constrained tangent-plane Hessian
    Hlocal = T.T @ (Q - lam * np.eye(3)) @ T

    # Full dynamic dipolar tensor
    N = dipolar_matrix(
        kx, ky, basis, n_quad=n_quad
    )

    # k=0 reference tensor, already included in equilibrium
    N0 = dipolar_matrix(
        0.0, 0.0, basis, n_quad=n_quad
    )

    H = np.zeros((2 * nm, 2 * nm), dtype=complex)

    for n in range(nm):
        for m in range(nm):

            block = np.zeros((2, 2), dtype=complex)

            if n == m:
                # Local anisotropy and Zeeman contribution
                block += Hlocal

                # Exchange stiffness
                k2 = kx**2 + ky**2
                qz2 = basis.qz[n]**2

                block += (
                    2.0 * A * (k2 + qz2)
                    * np.eye(2)
                )

            # Dynamic dipolar correction relative to k=0
            delta_N = N[n, m] - N0[n, m]

            block += (
                MU0 * Ms**2
                * (T.T @ delta_N @ T)
            )

            H[
                2*n:2*n+2,
                2*m:2*m+2
            ] = block

    # Remove tiny numerical asymmetries
    H = 0.5 * (H + H.conj().T)

    return H, m0, T


# ============================================================
# Linearized conservative LLG
# ============================================================

def solve_modes(
    kx,
    ky,
    B,
    Ms,
    A,
    d,
    Ku=0.0,
    K1=0.0,
    n_modes=3,
    n_quad=100,
):
    """
    Solve the conservative linearized LLG eigenproblem.

    Returns the positive-frequency modes.

    The LLG equation is:
        dm/dt = -gamma * m x B_eff

    with gamma taken as a positive magnitude.

    The matrix convention here gives:
        dot(q) = (gamma/Ms) J H q

    where:
        J = [[0,-1],[1,0]]
    """
    basis = ThicknessBasis(d, n_modes)

    H, m0, T = build_energy_hessian(
        kx=kx,
        ky=ky,
        B=B,
        Ms=Ms,
        A=A,
        basis=basis,
        Ku=Ku,
        K1=K1,
        n_quad=n_quad,
    )

    nm = n_modes

    J2 = np.array([
        [0.0, -1.0],
        [1.0, 0.0]
    ])

    J = np.kron(np.eye(nm), J2)

    # Linearized dynamical matrix
    D = (GAMMA / Ms) * J @ H

    eigenvalues, eigenvectors = np.linalg.eig(D)

    # Keep positive-frequency modes
    positive = np.where(np.imag(eigenvalues) > 1e-6)[0]

    if len(positive) != nm:
        raise RuntimeError(
            f"Expected {nm} positive-frequency modes, "
            f"found {len(positive)}. "
            "Check stability, units and equilibrium."
        )

    # Sort by frequency
    positive = positive[
        np.argsort(np.imag(eigenvalues[positive]))
    ]

    eigvals = eigenvalues[positive]
    eigvecs = eigenvectors[:, positive]

    frequencies = np.imag(eigvals) / (2.0 * np.pi)

    # Normalize eigenvectors
    for j in range(nm):
        v = eigvecs[:, j]
        v /= np.linalg.norm(v)

        # Fix arbitrary complex phase:
        # largest component becomes real and positive.
        idx = np.argmax(np.abs(v))
        v *= np.exp(-1j * np.angle(v[idx]))

        eigvecs[:, j] = v

    return {
        "frequency": frequencies,
        "eigenvalues": eigvals,
        "eigenvectors": eigvecs,
        "equilibrium": m0,
        "tangent_basis": T,
        "Hessian": H,
        "dynamical_matrix": D,
        "basis": basis,
    }


# ============================================================
# Eigenmode tracking
# ============================================================

def track_modes(results):
    """
    Track eigenmodes between consecutive parameter points
    using maximum eigenvector overlap.

    Parameters
    ----------
    results : list of solve_modes dictionaries

    Returns
    -------
    frequencies : ndarray, shape (n_points, n_modes)
    vectors : ndarray, shape (n_points, 2*n_modes, n_modes)
    """
    n_points = len(results)

    frequencies = np.array([
        r["frequency"] for r in results
    ])

    vectors = np.stack([
        r["eigenvectors"] for r in results
    ])

    for i in range(1, n_points):

        previous = vectors[i - 1]
        current = vectors[i]

        overlap = np.abs(previous.conj().T @ current)

        # Hungarian assignment: maximize total overlap
        row, col = linear_sum_assignment(-overlap)

        permutation = np.zeros(len(col), dtype=int)
        permutation[row] = col

        frequencies[i] = frequencies[i, permutation]
        vectors[i] = vectors[i][:, permutation]

    return frequencies, vectors


# ============================================================
# Field and wavevector sweeps
# ============================================================

def sweep_field(
    fields,
    field_direction,
    kx,
    ky,
    Ms,
    A,
    d,
    Ku=0.0,
    K1=0.0,
    n_modes=3,
    n_quad=100,
):
    """
    Calculate f(H) for a fixed field direction and wavevector.
    """
    direction = normalize(field_direction)

    results = []

    for field in fields:
        B = field * direction

        results.append(
            solve_modes(
                kx, ky, B, Ms, A, d,
                Ku=Ku,
                K1=K1,
                n_modes=n_modes,
                n_quad=n_quad,
            )
        )

    frequencies, vectors = track_modes(results)

    return frequencies, vectors, results


def sweep_wavevector(
    k_values,
    propagation_angle,
    B,
    Ms,
    A,
    d,
    Ku=0.0,
    K1=0.0,
    n_modes=3,
    n_quad=100,
):
    """
    Calculate f(k) for a fixed propagation angle and field.
    """
    results = []

    for k in k_values:
        kx = k * np.cos(propagation_angle)
        ky = k * np.sin(propagation_angle)

        results.append(
            solve_modes(
                kx, ky, B, Ms, A, d,
                Ku=Ku,
                K1=K1,
                n_modes=n_modes,
                n_quad=n_quad,
            )
        )

    frequencies, vectors = track_modes(results)

    return frequencies, vectors, results


# ============================================================
# Plotting
# ============================================================

def plot_frequency_sweep(
    x,
    frequencies,
    xlabel,
    ylabel="Frequency (GHz)",
    title=None,
    labels=None,
):
    """
    Plot tracked eigenfrequency branches.
    """
    frequencies_GHz = frequencies / 1e9

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for n in range(frequencies.shape[1]):
        label = (
            labels[n] if labels is not None
            else f"Mode {n}"
        )

        ax.plot(
            x,
            frequencies_GHz[:, n],
            linewidth=1.8,
            label=label,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title:
        ax.set_title(title)

    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()

    return fig, ax


def reconstruct_mode(
    result,
    mode_index,
    z=None,
):
    """
    Reconstruct the transverse dynamic magnetization
    profile of one eigenmode.

    Returns
    -------
    z : ndarray
    m1 : complex ndarray
    m2 : complex ndarray
    """
    basis = result["basis"]

    if z is None:
        z = np.linspace(
            -basis.d / 2,
            basis.d / 2,
            300,
        )

    phi = basis.phi(z)

    v = result["eigenvectors"][:, mode_index]

    a = v[0::2]
    b = v[1::2]

    m1 = np.sum(a[:, None] * phi, axis=0)
    m2 = np.sum(b[:, None] * phi, axis=0)

    return z, m1, m2


def plot_mode_profile(
    result,
    mode_index=0,
    normalize_profile=True,
):
    """
    Plot real, imaginary and amplitude profiles
    of the two transverse components.
    """
    z, m1, m2 = reconstruct_mode(
        result, mode_index
    )

    z_nm = z * 1e9

    if normalize_profile:
        scale = max(
            np.max(np.abs(m1)),
            np.max(np.abs(m2)),
        )

        if scale > 0:
            m1 = m1 / scale
            m2 = m2 / scale

    fig, axes = plt.subplots(
        2, 2,
        figsize=(10, 6),
        sharex=True,
    )

    axes[0, 0].plot(z_nm, m1.real, label="Re")
    axes[0, 0].plot(z_nm, m1.imag, "--", label="Im")
    axes[0, 0].set_title(r"$m_1(z)$")
    axes[0, 0].legend()

    axes[0, 1].plot(z_nm, m2.real, label="Re")
    axes[0, 1].plot(z_nm, m2.imag, "--", label="Im")
    axes[0, 1].set_title(r"$m_2(z)$")
    axes[0, 1].legend()

    axes[1, 0].plot(z_nm, np.abs(m1))
    axes[1, 0].set_title(r"$|m_1(z)|$")

    axes[1, 1].plot(z_nm, np.abs(m2))
    axes[1, 1].set_title(r"$|m_2(z)|$")

    for ax in axes.flat:
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("z (nm)")

    fig.suptitle(
        f"Thickness profile: mode {mode_index}"
    )

    fig.tight_layout()

    return fig, axes


# ============================================================
# Analytical validation: in-plane Kittel limit
# ============================================================

def validate_kittel_limit():
    """
    Check the uniform in-plane mode for:
        Ku = K1 = 0
        k = 0
        B along x

    Expected:
        f = gamma/(2*pi) * sqrt[B*(B + mu0*Ms)]
    """
    Ms = 1.4e6       # A/m
    A = 2.1e-11      # J/m
    d = 100e-9       # m
    B0 = 0.1         # T

    result = solve_modes(
        kx=0.0,
        ky=0.0,
        B=np.array([B0, 0.0, 0.0]),
        Ms=Ms,
        A=A,
        d=d,
        Ku=0.0,
        K1=0.0,
        n_modes=3,
    )

    f_expected = (
        GAMMA / (2.0 * np.pi)
        * np.sqrt(B0 * (B0 + MU0 * Ms))
    )

    f_calculated = result["frequency"][0]

    print("Kittel-limit validation")
    print("-----------------------")
    print(f"Expected:   {f_expected/1e9:.6f} GHz")
    print(f"Calculated: {f_calculated/1e9:.6f} GHz")
    print(
        "Relative error:",
        abs(f_calculated - f_expected) / f_expected
    )

    return result
