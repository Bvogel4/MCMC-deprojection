"""
shape_inference.py - Core functionality for 3D shape inference from 2D projections.

This module implements the methodology from Kado-Fong et al. (2020) to infer
the intrinsic 3D shapes of ellipsoids from their 2D projections.
"""

import numpy as np
import emcee
import os
import time
from multiprocessing import Pool
from scipy.special import factorial as scipy_factorial
import pickle
from scipy import stats
#supress warning from emcee
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="emcee")

Q_BINS = np.arange(0, 1.0000001, 0.04)   # same 0.04 bins as the original
import numpy as np
from scipy.special import gammaln
from scipy.special import ndtr, ndtri
import numpy as np
from scipy import stats
from scipy.stats import norm


import fast_likelihood as fl

_LL7 = None

def _init_ll7(q_obs, a_obs, kwargs):
    global _LL7
    _LL7 = fl.LogLike7(q_obs, a_obs, **kwargs)

def _ll7(theta):
    return _LL7(theta)


def random_viewing_angles(n):
    """
    Generate n random viewing angles uniformly distributed on a sphere.

    Parameters:
        n (int): Number of viewing angles to generate

    Returns:
        tuple: (phi, theta) where phi is in [0, 2π] and theta is in [0, π]
    """
    phi = np.random.uniform(0, 2 * np.pi, n)
    nu = np.random.uniform(0, 1, n)
    theta = np.arccos(2 * nu - 1)
    return phi,theta


def projected_axis_ratio(phi, theta, B, C):
    """
    Calculate the projected axis ratio q for a triaxial ellipsoid with axis ratios B/A and C/A
    viewed from angles (phi, theta).

    Based on Simonneau et al. (1998) as used in Kado-Fong et al. (2020).

    Parameters:
        phi (float or array): Azimuthal viewing angle in radians
        theta (float or array): Polar viewing angle in radians
        B (float or array): B/A axis ratio (intermediate/major)
        C (float or array): C/A axis ratio (minor/major)

    Returns:
        float or array: Projected axis ratio q = b/a
    """
    # Calculate f and g as per equations in Simonneau et al. (1998)
    f = np.sqrt(
        (C * np.sin(theta) * np.cos(phi)) ** 2 +
        (B * C * np.sin(theta) * np.sin(phi)) ** 2 +
        (B * np.cos(theta)) ** 2
    )

    g = (
            np.cos(phi) ** 2 + np.cos(theta) ** 2 * np.sin(phi) ** 2 +
            B ** 2 * (np.sin(phi) ** 2 + np.cos(theta) ** 2 * np.cos(phi) ** 2) +
            (C * np.sin(theta)) ** 2
    )

    # Calculate h
    h = np.sqrt((g - 2 * f) / (g + 2 * f))

    # Calculate q
    q = (1 - h) / (1 + h)

    return q


# def generate_projections(B_A, C_A, n_samples=10000):
#     """
#     Generate n_samples random projections of an ellipsoid with semi-axes a, b, c.
#
#     Parameters:
#         a, b, c (float): Semi-axes of the ellipsoid (a >= b >= c)
#         n_samples (int): Number of random projections to generate
#
#     Returns:
#         array: Projected axis ratios q = b/a
#     """
#
#
#     # Generate random viewing angles
#     phi, theta = random_viewing_angles(n_samples)
#
#     # Calculate projected axis ratios
#     q = projected_axis_ratio(phi, theta, B_A, C_A)
#
#     return q


# def projected_semi_axes(phi, theta, B, C):
#     """
#     Semi-axes (alpha >= beta) of the projected ellipse, in units of the
#     intrinsic major axis A, for viewing angles (phi, theta).
#
#     Uses the same Simonneau et al. (1998) f, g as projected_axis_ratio:
#     the projected semi-axes satisfy alpha^2 * beta^2 = f^2 and
#     alpha^2 + beta^2 = g, so they are the roots
#
#         alpha = sqrt((g + sqrt(g^2 - 4 f^2)) / 2)
#         beta  = sqrt((g - sqrt(g^2 - 4 f^2)) / 2)
#
#     Consistency: beta/alpha equals the q returned by projected_axis_ratio.
#
#     Returns
#     -------
#     alpha, beta : arrays
#         Projected semi-major and semi-minor axes in units of A.
#         Physical sizes: a_proj = A * alpha, b_proj = A * beta.
#         Circularized radius: r_circ = A * sqrt(alpha * beta) = A * sqrt(f).
#     """
#     f = np.sqrt(
#         (C * np.sin(theta) * np.cos(phi)) ** 2 +
#         (B * C * np.sin(theta) * np.sin(phi)) ** 2 +
#         (B * np.cos(theta)) ** 2
#     )
#     g = (
#             np.cos(phi) ** 2 + np.cos(theta) ** 2 * np.sin(phi) ** 2 +
#             B ** 2 * (np.sin(phi) ** 2 + np.cos(theta) ** 2 * np.cos(phi) ** 2) +
#             (C * np.sin(theta)) ** 2
#     )
#     disc = np.sqrt(np.clip(g ** 2 - 4 * f ** 2, 0, None))  # clip fp noise
#     alpha = np.sqrt((g + disc) / 2)
#     beta = np.sqrt((g - disc) / 2)
#     return alpha, beta
from fast_likelihood import projected_semi_axes


def project_ellipticity_xu(theta,phi,a,b,c):
    alpha = (a*np.cos(theta)*np.cos(phi))**2 + (b*np.cos(theta)*np.sin(phi))**2 + (c*np.sin(theta))**2
    beta = (a*np.sin(phi))**2 + (b*np.cos(phi))**2
    gamma = (a**2 - b**2)*np.cos(theta)*np.sin(phi)*np.cos(phi)
    f = np.sqrt(4*gamma**2 + (alpha - beta)**2)
    ell = 1 - np.sqrt( 1 - (2*f)/(alpha+beta+f))
    return 1-ell




def _in_support(a, b, c):
    """Boolean mask for the physical region A>0, 0<C<B<1."""
    return (a > 0) & (c > 0) & (b > c) & (b < 1)


def _cov_from_theta(theta):
    """
    Build the 3x3 covariance matrix from theta.

    Column order is (a, b, c).  sigma_ab and sigma_bc are zero by
    construction; only a and c are correlated.
    """
    theta = np.asarray(theta, dtype=float)
    sig_a, sig_b, sig_c, rho_ac = theta[3:7]

    cov = np.diag([sig_a ** 2, sig_b ** 2, sig_c ** 2])
    cov[0, 2] = cov[2, 0] = rho_ac * sig_a * sig_c
    return cov


def project_to_attainable(theta, logL, tol=1e-3,
                           center=None):
    """
    Pull theta toward a safe interior point until it becomes attainable.

    Bisects on a single interpolation factor f applied to the full theta
    vector: theta(f) = center + f * (theta - center). At f=1 this is the
    original theta.

    By default, `center` is theta itself, locally nudged just far enough
    that mu_b and mu_c clear their boundaries by a 2-sigma margin (using
    theta's own sigma_b, sigma_c). This targets boundary-hugging means
    directly, rather than shrinking sigmas toward 0 -- a mu_b sitting at
    0.95 gets pushed toward the interior instead of relying on shrinking
    sigma_b alone to rescue it. Note the default center does NOT shrink
    sigma_a/sigma_b/sigma_c; if theta is still not attainable at f=0
    despite mu being well clear of its boundaries, the issue is likely
    sigma being too large in some other sense, not boundary proximity --
    pass a `center` with reduced sigmas too in that case.

    mu_a and rho_ac are left untouched by the default center construction.

    Returns (theta_proj, f). f == 1.0 means theta was already fine.
    """

    theta = np.asarray(theta, dtype=float)
    if center is None:
        center = theta.copy()
        if center[1] + 2 * center[4] > 1:
            d = center[1] + 2 * center[4] - 1
            center[1] = center[1] - 0.5 * d
            center[4]=center[4]/2
        if center[1] - 2 * center[4] < 0:
            d = center[1] - 2 * center[4]
            center[1] = center[1] - 0.5 * d
            center[4]=center[4]/2
        if center[2] + 2 * center[5] > center[1]:
            d = center[2] + 2 * center[5] - center[1]
            center[2] = center[2] - 0.5 * d
            center[5] = center[5]/2
        if center[2] - 2 * center[5] < 0:
            d = center[2] - 2 * center[5]
            center[2] = center[2] - 0.5 * d
            center[5] = center[5]/2

    center = np.asarray(center, dtype=float)
    #check that center is valid
    if not valid_theta(center):
        raise ValueError(
            f"center point is not valid ({center}); cannot project "
        )
    #check that it is attainable
    if not attainable(center):
        print(center)
        raise ValueError(
            f'center is not attainable ({center}); cannot project'
        )

    def theta_at(f):
        return center + f * (theta - center)
    if logL == None:
        logL = fl.LogLike7(np.ones(10)*0.5,np.ones(10),n_draw_pool = 30000,solve_pool=10000)
    crn = fl.get_crn(logL.solve_pool, logL.seed, logL.dtype)

    def ok(f):
        t = theta_at(f)
        return (valid_theta(t)
                and fl.solve_latent_fast(t, crn) is not None)

    if ok(1.0):
        return theta.copy(), 1.0

    if not ok(0.0):
        raise ValueError(
            f"center point is not attainable ({center}); cannot project "
            f"theta toward it -- pass a different `center`."
        )
    lo, hi = 0.0, 1.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return theta_at(lo), lo

def _ellipsoid_truncnorm(params, n_samples):
    mu_B, mu_C, sigma_B, sigma_C = params

    a_B = (0 - mu_B) / sigma_B
    b_B = (1 - mu_B) / sigma_B
    B_samples = stats.truncnorm.rvs(a_B, b_B, loc=mu_B, scale=sigma_B, size=n_samples)

    # Vectorized C sampling using inverse CDF method
    # For truncated normal: X = Φ⁻¹(Φ(a) + U·(Φ(b) - Φ(a))) · σ + μ
    a_C = (0 - mu_C) / sigma_C  # scalar lower bound (standardized)
    b_C = (B_samples - mu_C) / sigma_C  # array of upper bounds (standardized)

    # CDF values at bounds (ndtr is ~3-5x faster than stats.norm.cdf)
    Phi_a = ndtr(a_C)  # scalar
    Phi_b = ndtr(b_C)  # array

    # Uniform samples for inverse CDF transform
    u = np.random.random(n_samples)

    # Inverse CDF transform
    cdf_vals = Phi_a + u * (Phi_b - Phi_a)
    cdf_vals = np.clip(cdf_vals, 1e-12, 1 - 1e-12)  # Numerical stability

    C_standardized = ndtri(cdf_vals)  # ndtri is faster than stats.norm.ppf
    C_samples = C_standardized * sigma_C + mu_C

    # Handle edge cases where bounds are invalid (B_val < mu_C effectively)
    invalid = Phi_b <= Phi_a
    C_samples[invalid] = B_samples[invalid] * 0.99

    return  B_samples, C_samples, np.ones((len(B_samples)))




class EllipsoidDistribution:
    """
    (B, C) ~ N(mu_B, sigma_B) x N(mu_C, sigma_C) conditioned on 0 < C < B < 1,
    with A = 1.

    Building the object evaluates `acc`, the fraction of the unconditioned
    normal mass lying inside the physical region. `sample` then draws by
    rejection when acc is high and by inverse CDF when it is not; both routes
    target the same distribution, so acc selects the algorithm only and never
    changes the sampled density.

        dist = EllipsoidDistribution(params)
        B, C, A = dist.sample(n_samples)
        dist.acc
    """

    def __init__(self, params, rng=None, accept_thresh=0.95,
                 n_grid=4001, n_sig=8.0):
        self.mu_B, self.mu_C, self.sigma_B, self.sigma_C = map(float, params)
        self.rng = np.random.default_rng() if rng is None else rng
        self.accept_thresh = accept_thresh

        self._b_grid = None
        self._cdf = None
        self.acc = 0.0

        if self.sigma_B > 0 and self.sigma_C > 0:
            self._build_grid(n_grid, n_sig)

    def _build_grid(self, n_grid, n_sig):
        """
        Conditioned marginal of B, unnormalised:
            p(b) ~ phi(b; mu_B, sigma_B) * P(0 < C < b)
        Its integral over (0, 1) is acc.
        """
        lo = max(0.0, self.mu_B - n_sig * self.sigma_B)
        hi = min(1.0, self.mu_B + n_sig * self.sigma_B)
        if hi <= lo:
            return

        b = np.linspace(lo, hi, n_grid)
        w = ndtr((b - self.mu_C) / self.sigma_C) - ndtr(-self.mu_C / self.sigma_C)
        np.clip(w, 0.0, None, out=w)
        pdf = norm.pdf(b, self.mu_B, self.sigma_B) * w

        cdf = np.concatenate(([0.0],
                              np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(b))))

        self._b_grid, self._cdf = b, cdf
        self.acc = float(cdf[-1])

    def sample(self, n_samples):
        """Returns (B, C, A) with A = 1. Draws are always physical."""
        if self.acc <= 0.0:
            z = np.zeros(n_samples)
            return z, z, np.ones(n_samples)

        if self.acc >= self.accept_thresh:
            B, C = self._sample_rejection(n_samples)
        else:
            B, C = self._sample_inverse_cdf(n_samples)
        return B, C, np.ones(n_samples)

    def _sample_rejection(self, n, max_rounds=50):
        B_out, C_out, filled = np.empty(n), np.empty(n), 0
        for _ in range(max_rounds):
            if filled >= n:
                break
            need = n - filled
            draw = int(np.ceil(need / self.acc * 1.15)) + 32
            B = self.rng.normal(self.mu_B, self.sigma_B, draw)
            C = self.rng.normal(self.mu_C, self.sigma_C, draw)
            ok = (C > 0.0) & (C < B) & (B < 1.0)
            k = min(int(ok.sum()), need)
            if k:
                B_out[filled:filled + k] = B[ok][:k]
                C_out[filled:filled + k] = C[ok][:k]
                filled += k
        if filled < n:
            raise RuntimeError("rejection sampling stalled")
        return B_out, C_out

    def _sample_inverse_cdf(self, n):
        keep = np.concatenate(([True], np.diff(self._cdf) > 0.0))
        B = np.interp(self.rng.random(n) * self.acc,
                      self._cdf[keep], self._b_grid[keep])

        a = np.full(n, -self.mu_C / self.sigma_C)
        b = (B - self.mu_C) / self.sigma_C
        C = self._truncnorm_std(a, b) * self.sigma_C + self.mu_C
        return B, np.clip(C, 0.0, None)

    def _truncnorm_std(self, lo, hi):
        """Standard normal truncated to [lo, hi] elementwise, tail-stable."""
        lo, hi = np.broadcast_arrays(np.asarray(lo, float), np.asarray(hi, float))
        out = np.empty(lo.shape)
        u = self.rng.random(lo.shape)

        flip = hi <= 0.0                    # reflect out of the far left tail
        a = np.where(flip, -hi, lo)
        b = np.where(flip, -lo, hi)

        tiny = np.finfo(float).tiny
        right = a >= 0.0
        if np.any(right):                   # both bounds >= 0: survival function
            Qa, Qb = ndtr(-a[right]), ndtr(-b[right])
            out[right] = -ndtri(np.clip(Qb + u[right] * (Qa - Qb), tiny, 1.0))
        mid = ~right                        # straddles zero: CDF difference is fine
        if np.any(mid):
            Fa, Fb = ndtr(a[mid]), ndtr(b[mid])
            out[mid] = ndtri(np.clip(Fa + u[mid] * (Fb - Fa),
                                     tiny, 1.0 - np.finfo(float).epsneg))

        return np.clip(np.where(flip, -out, out), lo, hi)


def adaptive_ellipsoid_distribution(params, n_samples, rng=None):
    """Convenience wrapper returning (B, C, A, acc)."""
    dist = EllipsoidDistribution(params, rng=rng)
    B, C, A = dist.sample(n_samples)
    return B, C, A, dist.acc

def generate_ellipsoids(params,n):
    mu_b,mu_c,sigma_b,sigma_c = params
    rng = np.random.default_rng()
    b = rng.normal(loc=mu_b,scale=sigma_b,size=n)
    c = rng.normal(loc=mu_c,scale=sigma_c,size=n)
    #count number of valid ellipses
    info = {}
    valid = 0 < c < b < 1
    acc = np.sum(~valid)/n

    info['acc']=acc
    if acc > 0.7:
        info['ok']=True
    else:
        info['ok']=False
    return b,c,info


def containment_penalty(acc, acc_min=0.85, width=0.03, hard_floor=0.50):
    """
    Smooth log-prior penalty: ~0 well above acc_min, softly negative below,
    -inf below hard_floor. Returns a value to ADD to the log-likelihood.

    Softplus form so the gradient never vanishes -- a hard cut at acc_min
    clips the posterior when the truth sits close to the boundary.
    """
    if not np.isfinite(acc) or acc < hard_floor:
        return -np.inf
    x = (acc_min - acc) / width
    return 40*float(np.logaddexp(0.0, x) ** 2)

def _cov_from_sigmas(sig_a, sig_b, sig_c, rho_ac):
    cov = np.diag([sig_a ** 2, sig_b ** 2, sig_c ** 2])
    cov[0, 2] = cov[2, 0] = rho_ac * sig_a * sig_c
    return cov


# --------------------------------------------------------------------------
# common random numbers
# --------------------------------------------------------------------------
_POOL = {}


def _get_pool(n, seed=0):
    """
    Fixed pool of standard normal deviates, reused across every call.

    This is what makes the moment solve well-posed: with the deviates held
    fixed, the map latent -> measured moments is a deterministic, smooth
    function, so the fixed-point iteration converges to a definite answer
    and the MCMC likelihood surface has no evaluation noise.
    """
    key = (seed,)
    pool = _POOL.get(key)
    if pool is None or len(pool) < n:
        pool = np.random.default_rng(seed).standard_normal((max(n, 200_000), 3))
        _POOL[key] = pool
    return pool[:n]


# --------------------------------------------------------------------------
# measured moments of a truncated draw
# --------------------------------------------------------------------------
def _measure(samples):
    """Post-truncation moments in theta order."""
    mu = samples.mean(axis=0)
    sd = samples.std(axis=0, ddof=1)
    rho = np.corrcoef(samples[:, 0], samples[:, 2])[0, 1]
    return np.concatenate([mu, sd, [rho]])


def _draw_truncated(latent, pool):
    """Draw from the truncated normal with the given LATENT parameters."""
    cov = _cov_from_sigmas(*latent[3:7])
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        return None, 0.0
    samp = latent[:3] + pool @ L.T
    keep = _in_support(samp[:, 0], samp[:, 1], samp[:, 2])
    return samp[keep], keep.mean()


# --------------------------------------------------------------------------
# the moment solve
# --------------------------------------------------------------------------
def solve_latent(theta, n_pool=200_000, seed=0, tol=1e-4, max_iter=60,
                 damping=1.0):
    """
    Find latent parameters whose truncated distribution has moments `theta`.

    Fixed-point iteration: measure what the current latent guess actually
    produces, then push the latent parameters by the shortfall.  Truncation
    is a mild, monotone perturbation over the useful region, so this
    contracts quickly -- typically 4-8 iterations from the trivial start
    latent = theta.

    Returns (latent, acceptance) or (None, acc) if theta is not attainable
    by any truncated Gaussian on this support.
    """
    theta = np.asarray(theta, dtype=float)
    pool = _get_pool(n_pool, seed)

    latent = theta.copy()
    acc = 0.0
    for _ in range(max_iter):
        samp, acc = _draw_truncated(latent, pool)
        if samp is None or len(samp) < 1000:
            return None, acc

        m = _measure(samp)
        d_mu = theta[:3] - m[:3]
        ratio = theta[3:6] / m[3:6]
        d_rho = theta[6] - m[6]

        if (np.abs(d_mu).max() < tol
                and np.abs(ratio - 1).max() < tol
                and abs(d_rho) < tol):
            return latent, acc

        latent[:3] += damping * d_mu
        latent[3:6] *= 1.0 + damping * (ratio - 1.0)
        latent[6] = np.clip(latent[6] + damping * d_rho, -0.999, 0.999)
        if not np.isfinite(latent).all() or (latent[3:6] <= 0).any():
            return None, acc

    return None, acc


def attainable(theta, **kw):
    """True if theta is realizable as the moments of a truncated Gaussian."""
    latent, _ = solve_latent(theta, **kw)
    return latent is not None


# --------------------------------------------------------------------------
# validity, in moment space
# --------------------------------------------------------------------------
def valid_theta(theta):
    """
    Cheap necessary conditions on MEASURED moments.

    Note what is NOT here: no sigma_ratio_cap.  In moment space the cap is
    implied -- a distribution supported on (0,1) cannot have a standard
    deviation above 1/sqrt(12) ~ 0.289 without being more spread than the
    uniform, which no truncated Gaussian can manage.  The runaway ridge
    toward sigma_b ~ 0.5 that ate the old fit is simply not expressible.
    """
    t = np.atleast_2d(np.asarray(theta, dtype=float))
    mu_a, mu_b, mu_c, sig_a, sig_b, sig_c, rho_ac = t.T
    ok = (
            np.isfinite(t).all(axis=1)
            & (mu_a > 0)
            & (mu_c > 0) & (mu_c < mu_b) & (mu_b < 1.0)
            & (sig_a > 0) & (sig_b > 0) & (sig_c > 0)
            & (sig_b < 0.2887) & (sig_c < 0.2887)
            & (np.abs(rho_ac) < 1.0)
    )
    return ok if np.ndim(theta) > 1 else bool(ok[0])


def truncated_solved(theta, n_samples, rng=None,
                                    n_pool=400_000, solve_pool=50_000,
                                    seed=0):
    """
    Draw n_samples ellipsoids whose measured moments equal theta.

    rng=None (the default) reuses the fixed common-random-number pool, which
    is what you want inside a likelihood: no evaluation noise.  Pass an rng
    for independent realizations.

    Returns (b, c, a), matching the existing call signature.
    """
    theta = np.asarray(theta, dtype=float)
    if not valid_theta(theta):
        raise ValueError(f"theta outside allowed region: {theta}")

    latent, acc = solve_latent(theta, n_pool=solve_pool, seed=seed)
    if latent is None:
        #print(f'theta={theta}')
        #raise ValueError(f"theta not attainable by a truncated Gaussian")
        info = {"ok": False, "acceptance": acc, "latent": latent}
        return None, None, None, info

    if rng is None:
        samp, acc = _draw_truncated(latent, _get_pool(n_pool, seed))
    else:
        need = int(n_samples / max(acc, 1e-3) * 1.3) + 10_000
        samp, acc = _draw_truncated(latent, rng.standard_normal((need, 3)))

    if samp is None or len(samp) < n_samples:
        info = {"ok": False, "acceptance": acc, "latent": latent}
        return None, None, None, info
        #raise RuntimeError("not enough accepted draws; raise n_pool")

    samp = samp[:n_samples]
    info = {"ok": True, "acceptance": acc, "latent": latent}
    return samp[:, 1], samp[:, 2], samp[:, 0], info


# --------------------------------------------------------------------------
# measurement (the inverse operation)
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
def relax_a(theta, solve_pool=50_000, seed=0, step=0.25, max_shift=40.0):
    """
    Make theta attainable by increasing mu_a only.

    When sig_a is large relative to mu_a, truncation at a>0 clips the lower
    tail so hard that the requested sig_a is not achievable and the moment
    solve diverges.  Sliding mu_a away from the wall removes the clipping
    without touching sig_a, rho, or any b/c moment -- the least-damaging fix
    when a is not a quantity you compare against.

    Returns (theta_relaxed, shift).  shift == 0.0 means no change was needed.
    Raises if no shift up to max_shift makes theta attainable (which would
    indicate the b/c part of theta is itself infeasible).
    """
    theta = np.asarray(theta, dtype=float)
    if attainable(theta, n_pool=solve_pool, seed=seed):
        return theta.copy(), 0.0
    sig_a = theta[3]
    shift = 0.0
    while shift < max_shift:
        shift += step * sig_a
        t = theta.copy()
        t[0] += shift
        if attainable(t, n_pool=solve_pool, seed=seed):
            return t, shift
    raise ValueError(
        f"could not make theta attainable by shifting mu_a alone; the b/c "
        f"moments are probably infeasible: {theta}")


def cov_fit(a, b, c, verbose=True, fix_a=True, ratio_gate=1.9):
    """
    Measure theta from a population of ellipsoids.

    Exact and unbiased for every parameter EXCEPT mu_a.  When sig_a is large
    relative to mu_a the measured moments are not attainable by a truncated
    Gaussian -- the a>0 clipping inflates the surviving sig_a beyond what any
    latent mu_a/sig_a can reproduce -- so with fix_a=True the returned mu_a is
    raised just enough to restore attainability.  sig_a, rho, and every b/c
    moment are left exactly as measured.  Pass fix_a=False for the raw
    (possibly infeasible) moments; note the round-trip identity
    cov_fit(*generate(theta)) == theta then holds only away from the a>0 wall.

    ratio_gate is a cheap trigger, not the correction itself: below it the
    real solver (relax_a) finds the minimal shift; above it the attainability
    check is skipped.  It sits above the worst-case feasibility ratio (~1.79,
    reached near |rho| ~ 0.9) so no infeasible theta slips through unchecked.
    """
    a, b, c = map(np.asarray, (a, b, c))
    mask = _in_support(a, b, c)
    if not mask.all():
        if verbose:
            print(f"cov_fit(): dropping {(~mask).sum()} points outside the physical region")
        a, b, c = a[mask], b[mask], c[mask]
    theta = _measure(np.column_stack([a, b, c]))

    if fix_a and theta[0] < ratio_gate * theta[3]:
        theta_fixed, shift = relax_a(theta)
        if shift and verbose:
            print(f"cov_fit(): mu_a raised {theta[0]:.3f} -> {theta_fixed[0]:.3f} "
                  f"(sig_a/mu_a = {theta[3] / theta[0]:.2f} was infeasible)")
        return theta_fixed
    return theta


def generate_ellipsoid_distribution(params, n_samples, rng=None):
    """
    Sample (B, C) from an ellipsoid axis-ratio model.

    Parameters
    ----------
    params : sequence of length 4 or 7
        len 4 -> (mu_B, mu_C, sigma_B, sigma_C)
                 independent/conditional truncated-normal model.
        len 7 -> (mu_a, mu_b, mu_c, s_aa, s_bb, s_cc, s_ac)
                 trivariate normal + rejection sampling on the support.
    n_samples : int
    rng : np.random.Generator, optional

    Returns
    -------
    B_samples, C_samples : ndarray, each of length n_samples
    """
    params = np.asarray(params, dtype=float).ravel()
    if rng is None:
        rng = np.random.default_rng()

    if params.size == 4:
        B_samples, C_samples, A_samples,acc = adaptive_ellipsoid_distribution(params, n_samples)
        if acc > 0.6:
            info = {'ok': True, 'acc': acc}
        else:
            info = {'ok': False, 'acc': acc}
    elif params.size == 7:
        if n_samples > 1000:
            crn = fl._CRN(5 * n_samples, seed=0)
        if n_samples < 1000:
            crn = fl._CRN(50 * n_samples, seed=0)
        latent = fl.solve_latent_fast(params, crn, max_iter=50)
        if latent is None:
            raise ValueError(f"params not attainable: {params}")
        a, b, c, keep = fl._draw(latent, crn)
        idx = np.flatnonzero(keep)
        if idx.size < n_samples:
            print(idx.size,5*n_samples)
            raise RuntimeError(f"only {idx.size} of {n_samples} survived; "
                               f"enlarge the pool")
        idx = idx[:n_samples]
        A_samples, B_samples, C_samples = a[idx], b[idx], c[idx]
        n_good = int(keep.sum())  # inside the support
        n_bad = keep.size - n_good  # rejected
        acc = n_good / keep.size
        if acc > 0.6:
            info = {'ok':True,'acc':acc}
        else:
            info = {'ok':False,'acc':acc}
        
    else:
        raise ValueError(
            f"params must have length 4 or 7, got {params.size}."
        )

    return B_samples, C_samples, A_samples, info



def generate_model_projections(params, n_samples=10000):
    """
    Generate model projected axis ratios using truncated normal distributions.

    Vectorized version using inverse CDF method for conditional sampling.
    """

    B_samples, C_samples, A_samples,info = generate_ellipsoid_distribution(params, n_samples)

    if info is not None:
        if not info['ok']:
            return None, None, info

    # Generate random viewing angles
    phi, theta = random_viewing_angles(n_samples)
    if len(params)==4:
        # q_model = project_ellipticity_xu(A_samples,B_samples,C_samples,theta,phi)
        # return q_model, np.ones(len(q_model)), info
        alpha, beta = projected_semi_axes(phi, theta, B_samples, C_samples)
        q_model = beta / alpha
        a_model = A_samples * np.sqrt(alpha * beta)
        return q_model, a_model,info
    elif len(params)==7:
        # # Calculate projected axis ratios
        #q_model_temp = projected_axis_ratio(phi, theta, B_samples, C_samples)
        #
        # a_model = projected_axis_ratio(phi, theta, B_samples, C_samples) * A_samples
        alpha, beta = projected_semi_axes(phi, theta, B_samples, C_samples)
        q_model = beta / alpha
        a_model = A_samples * np.sqrt(alpha * beta)
        return q_model, a_model, info
    else:
        raise(f'Incorrect input for parameters, length must be 4 or 7, currently is {len(params)}')


def _binned_poisson_loglike(n_obs, n_model):
    """
    sum_i [ n_i ln(m_i) - m_i - ln(n_i!) ] over flattened bins, with the same
    three edge-case rules as your original implementation:
      * n_i>0, m_i=0  -> penalty -n_i
      * n_i=0, m_i>0  -> penalty -m_i
      * n_i=0, m_i=0  -> contributes 0
    ln(n_i!) uses gammaln(n_i + 1): exact, vectorized, and (being independent
    of the parameters) irrelevant to the MCMC — kept so logL values remain
    comparable to your existing runs.
    """
    ni = np.asarray(n_obs, dtype=float).ravel()
    mi = np.asarray(n_model, dtype=float).ravel()

    log_like = 0.0
    #
    both = (ni > 0) & (mi > 0)
    if np.any(both):
        log_like += np.sum(ni[both] * np.log(mi[both]) - mi[both]
                           - gammaln(ni[both] + 1))
    #
    obs_no_model = (ni > 0) & (mi == 0)
    if np.any(obs_no_model):
        log_like += -np.sum(ni[obs_no_model])
    #
    model_no_obs = (ni == 0) & (mi > 0)
    if np.any(model_no_obs):
        log_like += -np.sum(mi[model_no_obs])

    return log_like


def _a_bin_edges(a_obs, n_bins=8):
    """
    Quantile-based interior edges (equal data weight per A slice) with
    open-ended outer bins: edge[0]=0 and edge[-1] far beyond the data, so
    model draws outside the observed size range land in (and are penalized
    through) the outer bins instead of being dropped from the normalization.
    """
    interior = np.quantile(a_obs, np.linspace(0, 1, n_bins + 1)[1:-1])
    interior = np.unique(interior)  # guard against duplicate quantiles
    return np.concatenate([[0.0], interior, [max(a_obs.max() * 5, interior[-1] * 5)]])

def attainable_fast(params):
    #quickly see if params might be possible, want to toss quickly if we know it's not.
    #for 7 parameters
    mu_a,mu_b,mu_c,sigma_a,sigma_b,sigma_c,rho_ac = params
    #physical boundary checks
    if not 0 < mu_c < mu_b < 1:
        return False
    if not -1 < rho_ac < 1:
        return False
    if not (0 < sigma_b < 0.7 and 0 < sigma_c < 0.7):
        return False
    if (0 > mu_b-0.2*sigma_b) or (mu_b+0.2*sigma_b > 1):
        return False
    if (0 > mu_c-0.2*sigma_c) or (mu_c+0.2*sigma_c > mu_b):
        return False
    if not valid_theta(params):
        return False
    return True

# ----------------------------------------------------------------------
# The likelihood
# ----------------------------------------------------------------------


def old_log_likelihood(params, q_obs):
    """
    Calculate the log likelihood using equation (6) from Kado-Fong et al.

    ln p(q|μB,μC,σB,σC) = ∑(ni ln(mi) - mi - ln(ni!))

    where ni is the observed count where 0.04i < q ≤ 0.04(i+1)
    and mi is the predicted count in the same range.

    Parameters:
        params (list): [mu_B, mu_C, sigma_B, sigma_C]
        q_obs (array): Observed projected axis ratios

    Returns:
        float: Log likelihood
    """

    mu_B, mu_C, sigma_B, sigma_C = params

    # Enforce physical constraints
    # mu_B and mu_C must be in [0, 1]
    if not (0 < mu_C < mu_B <= 1):
        return -np.inf
    # sigma_B and sigma_C must be between 0 and 0.8
    if not (0 < sigma_B < 0.8 and 0 < sigma_C < 0.8):
        return -np.inf

    from scipy import stats

    #n_model_draws = min(len(q_obs),10000)
    n_model_draws = len(q_obs)# number of draws to approximate the model distribution

    q_model,_,_ = generate_model_projections(params, n_samples=n_model_draws)


    #bins from 0-1 in 0.04 increments
    bins = np.arange(0, 1, 0.04)

    # Count observed and model values in each bin
    n_obs, _ = np.histogram(q_obs, bins=bins)
    n_model, _ = np.histogram(q_model, bins=bins)

    # Normalize model counts to match observed counts
    n_model = n_model / np.sum(n_model) * np.sum(n_obs)

    # Vectorized stirling approximation
    def vectorized_stirling(n_array):
        result = np.zeros_like(n_array, dtype=float)
        small_mask = n_array < 30
        large_mask = ~small_mask

        # For small values, use scipy factorial
        if np.any(small_mask):
            small_n = n_array[small_mask]
            result[small_mask] = np.array([np.log(scipy_factorial(n)) for n in small_n])

        # For large values, use Stirling's approximation
        if np.any(large_mask):
            large_n = n_array[large_mask]
            result[large_mask] = (large_n + 0.5) * np.log(large_n) - large_n + 0.5 * np.log(2 * np.pi)

        return result

    # Vectorized log-likelihood calculation
    ni = n_obs
    mi = n_model

    # Create masks for different cases
    mask_both_positive = (ni > 0) & (mi > 0)
    mask_ni_positive_mi_zero = (ni > 0) & (mi == 0)
    mask_ni_zero_mi_positive = (ni == 0) & (mi > 0)

    # Initialize log likelihood
    log_like = 0

    # Case 1: Both positive
    if np.any(mask_both_positive):
        ni_pos = ni[mask_both_positive]
        mi_pos = mi[mask_both_positive]
        log_like += np.sum(ni_pos * np.log(mi_pos) - mi_pos - vectorized_stirling(ni_pos))

    # Case 2: Model incorrectly predicts impossible event
    if np.any(mask_ni_positive_mi_zero):
        ni_pos_mi_zero = ni[mask_ni_positive_mi_zero]
        log_like += -np.sum(ni_pos_mi_zero)

    # Case 3: Model predicts events that didn't occur
    if np.any(mask_ni_zero_mi_positive):
        mi_pos_ni_zero = mi[mask_ni_zero_mi_positive]
        log_like += -np.sum(mi_pos_ni_zero)

    # Case 4: Agreement on zero counts (adds nothing)

    assert loglike < 0
    return log_like


def log_likelihood(params, q_obs, a_obs=None):
    """
    Unified 4/7-parameter log likelihood.

    Parameters
    ----------
    params : array, length 4 or 7
        4: [mu_B, mu_C, sigma_B, sigma_C]
        7: [mu_a, mu_b, mu_c, sigma_a, sigma_b, sigma_c, rho_ac]
    q_obs : array
        Observed projected axis ratios.
    a_obs : array or None
        Observed (projected) effective radii, paired index-by-index with
        q_obs. Required for the 7-parameter model.

    Usage with emcee (both models):
        sampler = emcee.EnsembleSampler(n_walkers, ndim, log_likelihood,
                                        args=[q_obs, a_obs], pool=pool)
    """
    params = np.asarray(params, dtype=float)
    ndim = len(params)
    # n_draws = min(len(q_obs),10000)
    n_draws = len(q_obs)  # number of draws to approximate the model distribution

    # ---------------- constraints / priors ----------------
    # if ndim == 4:
    mu_B, mu_C, sigma_B, sigma_C = params
    if not (0 < mu_C < mu_B <= 1):
        return -np.inf
    if not (0 < sigma_B < 0.5 and 0 < sigma_C < 0.5):
        return -np.inf

    q_model, a_model, info = generate_model_projections(params, n_samples=n_draws)
    if info is not None:
        if not info["ok"]:
            return -np.inf
    acc = info['acc']

    n_obs, _ = np.histogram(q_obs, bins=Q_BINS)
    n_model, _ = np.histogram(q_model, bins=Q_BINS)


    total_model = n_model.sum()
    total_obs = n_obs.sum()
    if total_obs != total_model:
        n_model = n_model * total_obs / total_model

    loglike = _binned_poisson_loglike(n_obs, n_model)
    # min_q = min(q_obs)
    # i_q_model = q_model[q_model < min_q]
    # add a negative penalty for each model prediction under min_q, scaled by how far it is:
    # d_q = (i_q_model - min_q) / .04
    # p = np.sum(d_q)
    # loglike +=p
    loglike = loglike - containment_penalty(acc)
    assert loglike < 0
    return loglike


def theta0_from_4param(params_4, a_obs,logL):
    """Promote a (mu_B, mu_C, sigma_B, sigma_C) fit to the full 7-vector."""
    mu_B, mu_C, sigma_B, sigma_C = np.asarray(params_4, dtype=float)
    mu_a = np.mean(a_obs)
    sigma_a = np.std(a_obs, ddof=1)
    rho_ac = 0.0  # correlation, not covariance

    theta0 = np.array([mu_a, mu_B, mu_C, sigma_a, sigma_B, sigma_C, rho_ac])
    if not valid_theta(theta0):
        raise ValueError(f"theta0 is outside the allowed region: {theta0}")

    theta0,acc = project_to_attainable(theta0,logL)
    return theta0,acc

def initialize_walkers(theta0, n_walkers, rng, logL, frac=0.1, range_frac=0.1,
                       sigma_b_floor_frac=0.1, max_tries=1000):
    """
    Gaussian ball around theta0, sized to the physical scale of each
    parameter rather than a single fraction of theta0.
    - mu_a, sigma_a are positive and formally unbounded above, so their
      jitter is `frac * theta0` (relative jitter around the current value).
    - mu_c, sigma_c, rho_ac live inside known finite prior ranges, so their
      jitter is `range_frac * (hi - lo)` -- a fraction of the prior width.
    - mu_b and sigma_b are coupled: sigma_b's jitter scale is modulated by
      how much room mu_b has before hitting the (0, 1) boundary (max at
      mu_b=0.5, shrinking toward the edges). This avoids proposing walkers
      like mu_b=0.95, sigma_b=0.4, which will almost always be rejected.
      `sigma_b_floor_frac` keeps a small floor on the jitter so walkers
      near the edge aren't frozen entirely.

    emcee raises on a non-finite initial log-probability rather than
    resampling, so starting points are filtered here by evaluating `logL`
    directly and keeping only the finite ones (rejects -inf, +inf, nan).
    """
    theta0 = np.asarray(theta0, dtype=float)
    if rng is None:
        rng = np.random.default_rng()

    base_scale = np.array([
        frac * abs(theta0[0]),  # mu_a     (0, inf)
        range_frac * 1.0,  # mu_b     (0, 1)
        range_frac * 1.0,  # mu_c     (0, mu_b) subset of (0, 1)
        frac * abs(theta0[3]),  # sigma_a  (0, inf)
        range_frac * 0.5,  # sigma_b  (0, 0.5) soft, base cap before mu_b coupling
        range_frac * 0.5,  # sigma_c  (0, 0.5) soft
        range_frac * 2.0,  # rho_ac   (-1, 1)
    ])

    pos = np.empty((n_walkers, 7))
    filled = 0

    for _ in range(max_tries):
        n_try = n_walkers - filled
        trial = np.empty((n_try, 7))
        trial[:, 0] = theta0[0] + base_scale[0] * rng.standard_normal(n_try)  # mu_a
        trial[:, 1] = theta0[1] + base_scale[1] * rng.standard_normal(n_try)  # mu_b
        trial[:, 2] = theta0[2] + base_scale[2] * rng.standard_normal(n_try)  # mu_c
        trial[:, 3] = theta0[3] + base_scale[3] * rng.standard_normal(n_try)  # sigma_a
        # room_b: 0 at mu_b=0 or 1, 0.5 at mu_b=0.5. Normalize so the
        # multiplier is 1.0 at center and floors out near the edges instead
        # of hitting exactly zero.
        room_b = np.minimum(trial[:, 1], 1.0 - trial[:, 1]) / 0.5
        sigma_b_mult = np.clip(room_b, sigma_b_floor_frac, 1.0)
        trial[:, 4] = theta0[4] + base_scale[4] * sigma_b_mult * rng.standard_normal(n_try)  # sigma_b
        trial[:, 5] = theta0[5] + base_scale[5] * rng.standard_normal(n_try)  # sigma_c
        trial[:, 6] = theta0[6] + base_scale[6] * rng.standard_normal(n_try)  # rho_ac

        ll = np.array([logL(t) for t in trial], dtype=float)
        good = trial[np.isfinite(ll)]

        take = min(len(good), n_walkers - filled)
        pos[filled:filled + take] = good[:take]
        filled += take
        if filled == n_walkers:
            return pos

    raise RuntimeError(
        f"initialize_walkers: only found {filled}/{n_walkers} walkers with a "
        f"finite log-likelihood after {max_tries} attempts."
    )


def _init_walkers_4d(theta0, n_walkers, q_obs, rng=None,
                     mu_scale=0.05, sigma_scale=0.02,
                     oversample=8, min_walkers=None, verbose=True):
    """
    Draw oversample*n_walkers candidates, keep the valid ones, and return
    up to n_walkers of them. If fewer survive, run with the smaller (even)
    ensemble rather than retrying.
    """
    ndim = 4
    rng = np.random.default_rng() if rng is None else rng
    theta0 = np.asarray(theta0, float)
    scale = np.array([mu_scale, mu_scale, sigma_scale, sigma_scale])
    if min_walkers is None:
        min_walkers = 2 * ndim + 2          # 10 for ndim=4

    trial = theta0 + scale * rng.standard_normal((oversample * n_walkers, ndim))
    mu_B, mu_C, s_B, s_C = trial.T
    ok = ((s_B > 1e-3) & (s_C > 1e-3) & (s_B < 0.5) & (s_C < 0.5) &
          (mu_C > 0.02) & (mu_B < 0.98) & (mu_C < mu_B) &
          (mu_B + s_B < 1.0) & (mu_C - s_C > 0.0))
    cand = trial[ok]

    finite = np.array([np.isfinite(log_likelihood(t, q_obs)) for t in cand],
                      dtype=bool) if len(cand) else np.zeros(0, dtype=bool)
    good = cand[finite]

    n_keep = min(len(good), n_walkers)
    n_keep -= n_keep % 2                    # emcee needs an even count
    if n_keep < min_walkers:
        raise RuntimeError(
            f"Only {n_keep} valid walkers from {len(trial)} draws; "
            f"theta0={theta0} is probably on a boundary."
        )
    if verbose and n_keep < n_walkers:
        print(f"  reduced ensemble: {n_keep}/{n_walkers} walkers")
    return good[:n_keep]

def infer_intrinsic_shape(q_obs, n_walkers=128, n_steps=5000, burn_in=500,
                          n_cores=None, output_prefix=None, output_dir="results"):
    """
    Infer the intrinsic shape distribution from observed projected axis ratios.

    Parameters:
        q_obs (array): Observed projected axis ratios
        n_walkers (int): Number of MCMC walkers
        n_steps (int): Number of MCMC steps
        burn_in (int): Number of burn-in steps to discard
        initial_guess (list): Initial parameter guess [mu_B, mu_C, sigma_B, sigma_C]
        n_cores (int): Number of CPU cores to use
        output_prefix (str): Prefix for output files
        output_dir (str): Directory to save results

    Returns:
        tuple: (samples, max_prob_params, sampler) - MCMC samples, parameters with highest probability, and the sampler
    """
    # Set initial guess if not provided

    ndim = 4  # Number of parameters

    # # Initialize walkers in a small ball around the initial guess
    # pos = [initial_guess + 1e-2 * np.random.randn(ndim) for _ in range(n_walkers)]

    # intial walker positions uniformly distributed in the valid b/a vs c/a space
    #first guess of where to start based on peak of q_obs
    q_obs_peak = np.mean(q_obs)
    q_obs_std = np.std(q_obs)
    mu_B = q_obs_peak + 2/3*q_obs_std
    mu_C = q_obs_peak - 2/3*q_obs_std
    sigma_B = q_obs_std/2
    sigma_C = q_obs_std/2
    print(f'intial guess = {mu_B}, {mu_C}, {sigma_B}, {sigma_C}')
    #make sure intial guess is within physical limits
    if mu_B > 0.9:
        mu_B = .9
    if mu_C < 0.1:
        mu_C = 0.1
    if sigma_B > 0.35:
        sigma_B = 0.35
    if sigma_C > 0.35:
        sigma_C = 0.35


    #start all walkers at the same initial guess

    pos = _init_walkers_4d([mu_B, mu_C, sigma_B, sigma_C], n_walkers, q_obs)
    n_walkers = pos.shape[0]

    C = pos - pos.mean(axis=0)
    assert np.all(np.abs(C).max(axis=0) > 0), "a parameter column has zero spread"
    assert np.linalg.cond(C) < 1e8, f"ill-conditioned start: cond={np.linalg.cond(C):.2e}"


    for i in range(n_walkers):
        params = pos[i, :]
        ll = log_likelihood(params, q_obs)
        if ll == -np.nan or ll == -np.inf:
            print(f'{ll}, {params}')
            raise f'invalid walker'
        assert ll < 0, 'loglikelhood must be a real negative number!'
        

    # Set up the sampler with multiprocessing if requested
    if n_cores is not None and n_cores > 1:
        with Pool(processes=n_cores) as pool:
            sampler = emcee.EnsembleSampler(n_walkers, ndim, log_likelihood, args=[q_obs], pool=pool)

            # Run MCMC
            print(f"Running MCMC with {n_cores} processes...")
            start_time = time.time()
            sampler.run_mcmc(pos, n_steps, progress=True)
            end_time = time.time()
            print(f"MCMC completed in {end_time - start_time:.2f} seconds")
    else:
        # Run without multiprocessing
        sampler = emcee.EnsembleSampler(n_walkers, ndim, log_likelihood, args=[q_obs])

        # Run MCMC
        print("Running MCMC...")
        start_time = time.time()
        sampler.run_mcmc(pos, n_steps, progress=True)
        end_time = time.time()
        print(f"MCMC completed in {end_time - start_time:.2f} seconds")

    # Discard burn-in and get samples
    samples = sampler.get_chain(discard=burn_in, flat=True)
    log_probs = sampler.get_log_prob(discard=burn_in, flat=True)

    # Find the parameters with highest probability
    max_prob_idx = np.argmax(log_probs)
    max_prob_params = samples[max_prob_idx]

    # Save results if output_prefix is provided
    if output_prefix:
        os.makedirs(output_dir, exist_ok=True)

        # Save samples and parameters
        np.save(f"{output_dir}/{output_prefix}_samples.npy", samples)
        np.save(f"{output_dir}/{output_prefix}_max_prob_params.npy", max_prob_params)
        np.save(f"{output_dir}/{output_prefix}_log_probs.npy", log_probs)

        # Save full chain for diagnostics
        full_chain = sampler.get_chain()
        np.save(f"{output_dir}/{output_prefix}_full_chain.npy", full_chain)

        # Save observed q values
        np.save(f"{output_dir}/{output_prefix}_q_obs.npy", q_obs)

        # Save all results in a pickle file for easy loading
        results = {
            'samples': samples,
            'max_prob_params': max_prob_params,
            'log_probs': log_probs,
            'full_chain': full_chain,
            'q_obs': q_obs,
            'n_walkers': n_walkers,
            'n_steps': n_steps,
            'burn_in': burn_in,
        }

        with open(f"{output_dir}/{output_prefix}_results.pkl", 'wb') as f:
            pickle.dump(results, f)

    return samples, max_prob_params, sampler
    # pos = np.array([[mu_B, mu_C, sigma_B, sigma_C] for _ in range(n_walkers)])
    # params = [mu_B, mu_C, sigma_B, sigma_C]
    #
    # B,C,A = _ellipsoid_truncnorm(params,n_walkers)
    #
    # pos[:,0] = B
    # pos[:,1] = C
    #
    # for i in range(n_walkers):
    #     mu_B, mu_C, sigma_B, sigma_C = pos[i,:]
    #     if mu_B + 1 * sigma_B > 0.98:
    #         pos[i,0] += -0.02
    #         pos[i,2] = 0.01
    #     if mu_C - 1 * sigma_C < 0.02:
    #         pos[i,1] += 0.02
    #         pos[i, 3] = 0.01
    #     if mu_C + 1 * sigma_C > mu_B:
    #         pos[i, 3] = 0.01
    #         pos[i, 1] += -0.02

    # #add a little variation to each walker
    # # Define separate variation scales for means and sigmas
    # mu_variation = 0.25
    # sigma_variation = 0.01
    #
    # # Create random variations with appropriate scales for each parameter
    # variations = np.zeros((n_walkers, ndim))
    # variations[:, 0] = mu_variation * np.random.randn(n_walkers)  # mu_B
    # variations[:, 1] = mu_variation * np.random.randn(n_walkers)  # mu_C
    # variations[:, 2] = sigma_variation * np.random.randn(n_walkers)  # sigma_B
    # variations[:, 3] = sigma_variation * np.random.randn(n_walkers)  # sigma_C
    #
    # # Add variations to positions
    # pos += variations
    # #ensure walkers are within physical limits
    # for i in range(n_walkers):
    #     if pos[i, 0] > .9:
    #         pos[i, 0] = 0.9
    #     if pos[i, 0] < 0.2:
    #         pos[i, 0] = 0.2
    #     if pos[i, 1] < .15:
    #         pos[i, 1] = .15
    #     if pos[i, 1] > 0.8:
    #         pos[i, 1] = 0.8
    #     if pos[i, 2] > 0.5:
    #         pos[i, 2] = 0.4
    #     if pos[i, 3] > 0.5:
    #         pos[i, 3] = 0.4
    #     if pos[i, 2] < 0:
    #         pos[i, 2] = 0.01
    #     if pos[i, 3] < 0:
    #         pos[i, 3] = 0.01
    #     if pos[i, 0] < pos[i, 1]:
    #         pos[i,0],pos[i,1] = pos[i,1],pos[i,0]

    #make sure none of the walkers start in a bad region of the loglikelihood


def infer_intrinsic_shape_multivariate(q_obs, params_4,
                                       a_obs,
                                       n_walkers=128, n_steps=5000, burn_in=500,
                                       n_cores=None, rng=None,
                                       output_prefix=None, output_dir="results"):
    """
    Infer the 7-parameter intrinsic shape distribution, warm-started from a
    4-parameter result.

    Parameters
    ----------
    q_obs : array
        Observed projected axis ratios.
    log_likelihood : callable
        Your updated likelihood: log_likelihood(theta, q_obs, *likelihood_args).
        It will receive the 7-vector [mu_a, mu_b, mu_c, sigma_a, sigma_b,
        sigma_c, rho_ac] (std+corr form). If it works internally with the
        canonical covariance parameters, convert at the top with
        MultivariateShapeModel.std_corr_to_params(theta), and guard with
        is_valid_std_corr(theta) -> -inf.
    params_4 : array
        [mu_B, mu_C, sigma_B, sigma_C] from the 4-parameter run
        (e.g. its max_prob_params).
    a_obs : array, optional
        Observed sizes A, used to initialize mu_a / sigma_a.
    mu_a, sigma_a : float, optional
        Explicit A initialization (overrides a_obs).
    likelihood_args : tuple
        Extra positional args passed to log_likelihood after q_obs
        (e.g. your a-observable data if the likelihood uses it).
    ball : dict, optional
        Override walker-ball scales, keys: mu_bc, sigma_bc, mu_a_rel,
        sigma_a_rel, rho.
    Other parameters mirror infer_intrinsic_shape().

    Returns
    -------
    samples, max_prob_params, sampler
        samples/max_prob_params are in the std+corr parameterization; a
        canonical-covariance copy of max_prob_params is also saved to disk.
    """
    ndim = 7
    mu_B, mu_C, sigma_B, sigma_C = np.asarray(params_4, dtype=float)
    mu_a, sigma_a = np.mean(a_obs), np.std(a_obs)
    rho_ac = 0

    ll_kwargs = dict(solve_pool=20_000)  # add n_draw_pool=... to tune
    ll = fl.LogLike7(q_obs, a_obs, **ll_kwargs)

    theta0,f = theta0_from_4param(params_4, a_obs,ll)


    if f < 1.0:
        print(f"warning: theta0 dispersions shrunk by {f:.2f} to reach feasibility")

    print(f'initializing {n_walkers} walkers')
    pos = initialize_walkers(theta0, n_walkers, rng, ll)

    #check walkers
    for p in pos:
        log_like = ll.__call__(p)
        print(log_like)
        if np.isnan(log_like):
            print(f'Invalid Walker: {p}')

    n_walkers = pos.shape[0]

    #
    # logL = fl.LogLike7(q_obs, a_obs)
    # sampler = emcee.EnsembleSampler(n_walkers, 7, logL, pool=pool)

    if n_cores is not None and n_cores > 1:
        with Pool(processes=n_cores, initializer=_init_ll7,
                  initargs=(q_obs, a_obs, ll_kwargs)) as pool:
            sampler = emcee.EnsembleSampler(n_walkers, ndim, _ll7, pool=pool)
            print(f"Running 7-parameter MCMC with {n_cores} processes...")
            start_time = time.time()
            sampler.run_mcmc(pos, n_steps, progress=True)
            print(f"MCMC completed in {time.time() - start_time:.2f} seconds")
    else:
        _init_ll7(q_obs, a_obs, ll_kwargs)
        sampler = emcee.EnsembleSampler(n_walkers, ndim, _ll7)
        print("Running 7-parameter MCMC...")
        start_time = time.time()
        sampler.run_mcmc(pos, n_steps, progress=True)
        print(f"MCMC completed in {time.time() - start_time:.2f} seconds")
        
    samples = sampler.get_chain(discard=burn_in, flat=True)
    log_probs = sampler.get_log_prob(discard=burn_in, flat=True)

    max_prob_idx = np.argmax(log_probs)
    max_prob_params = samples[max_prob_idx]

    if output_prefix:
        os.makedirs(output_dir, exist_ok=True)

        np.save(f"{output_dir}/{output_prefix}_samples.npy", samples)
        np.save(f"{output_dir}/{output_prefix}_max_prob_params.npy", max_prob_params)
        np.save(f"{output_dir}/{output_prefix}_log_probs.npy", log_probs)

        full_chain = sampler.get_chain()
        np.save(f"{output_dir}/{output_prefix}_full_chain.npy", full_chain)
        np.save(f"{output_dir}/{output_prefix}_q_obs.npy", q_obs)

        # Canonical-covariance copy for downstream code that expects
        # [mu_a, mu_b, mu_c, sigma_aa, sigma_bb, sigma_cc, sigma_ac]

        results = {
            'samples': samples,
            'max_prob_params': max_prob_params,
            'parameterization': 'std_corr',
            'log_probs': log_probs,
            'full_chain': full_chain,
            'q_obs': q_obs,
            'params_4_start': np.asarray(params_4, dtype=float),
            'theta0': theta0,
            'n_walkers': n_walkers,
            'n_steps': n_steps,
            'burn_in': burn_in,
        }
        with open(f"{output_dir}/{output_prefix}_results.pkl", 'wb') as f:
            pickle.dump(results, f)

    try:
        tau = sampler.get_autocorr_time(quiet=True)
        print(f"Autocorrelation times: {np.round(tau, 1)}")
        print(f"Effective samples: {n_steps / np.max(tau):.0f}")
    except Exception as e:
        print(f"Could not compute autocorrelation time: {e}")

    return samples, max_prob_params, sampler

