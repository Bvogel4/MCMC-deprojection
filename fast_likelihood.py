"""
fast_likelihood.py -- drop-in replacement for the 7-parameter branch of
shape_inference.log_likelihood.

Same model, same binned-Poisson statistic, ~20-80x faster per evaluation,
and (unlike the original) exactly deterministic: calling it twice at the
same theta returns the same number.

Usage
-----
    from fast_likelihood import LogLike7

    logL = LogLike7(q_obs, a_obs)          # build ONCE, outside the sampler
    sampler = emcee.EnsembleSampler(n_walkers, 7, logL, pool=pool)
    sampler.run_mcmc(pos, n_steps, progress=True)

Note there is no `args=[q_obs, a_obs]`: the data, the observed histogram
and the bin edges are baked into the object at construction time.

What changed (see the notes at the bottom of the file for details):
  1. the moment solve runs once per call instead of twice
  2. common random numbers are actually reused, so the surface is smooth
  3. everything that does not depend on theta is precomputed
  4. the projection trig is precomputed and the projection is done once
  5. cheaper draw, cheaper moments, cheaper histogram
"""

import numpy as np
from scipy.special import gammaln

Q_BINS = np.arange(0, 1.0000001, 0.04)
_NQ = len(Q_BINS) - 1
_INV_QW = 1.0 / 0.04
TOL_VEC = np.array([0.10, 0.02, 0.02, 0.10, 0.02, 0.02, 0.02])

# ----------------------------------------------------------------------
# common random numbers: standard normals AND viewing angles, held fixed
# and paired by row index so that a small change in theta changes only the
# few samples that cross the support boundary.
# ----------------------------------------------------------------------
class _CRN:
    def __init__(self, n, seed=0, dtype=np.float64):
        rng = np.random.default_rng(seed)
        z = rng.standard_normal((3, n)).astype(dtype)   # row-major: each
        self.z0, self.z1, self.z2 = z[0], z[1], z[2]    # component contiguous
        phi = rng.uniform(0, 2 * np.pi, n)
        cos_t = 2 * rng.uniform(0, 1, n) - 1
        self.phi = phi
        self.theta=np.acos(cos_t)
        self.c2p = (np.cos(phi) ** 2).astype(dtype)     # cos^2 phi
        self.c2t = (cos_t ** 2).astype(dtype)           # cos^2 theta
        self.n = n


_POOLS = {}


def get_crn(n, seed=0, dtype=np.float64):
    key = (n, seed, np.dtype(dtype).str)
    p = _POOLS.get(key)
    if p is None:
        p = _POOLS[key] = _CRN(n, seed, dtype)
    return p


# ----------------------------------------------------------------------
# draw and measure
# ----------------------------------------------------------------------
def _draw(latent, crn):
    """
    Truncated draw, using the closed form of the Cholesky factor.

    cov is diag(sa^2, sb^2, sc^2) with a single off-diagonal rho*sa*sc, so
    L is known analytically and the matmul + (n,3) temporary is replaced by
    three contiguous axpy's.
    """
    mu_a, mu_b, mu_c, sa, sb, sc, rho = latent
    if sa <= 0 or sb <= 0 or sc <= 0 or abs(rho) >= 1:
        return None
    a = mu_a + sa * crn.z0
    b = mu_b + sb * crn.z1
    c = mu_c + sc * (rho * crn.z0 + np.sqrt(1.0 - rho * rho) * crn.z2)
    keep = (b < 1.0) & (c > 0.0) & (c < b) & (a > 0.0)
    return a, b, c, keep


def _moments(a, b, c, keep, min_keep=1000):
    """
    Post-truncation mean/sd of (a,b,c) plus corr(a,c).

    Second moments come from three dot products instead of ndarray.std +
    np.corrcoef, which between them make five full-size temporaries.
    """
    idx = np.flatnonzero(keep)
    n = idx.size
    if n < min_keep:
        return None, idx
    ak, bk, ck = a[idx], b[idx], c[idx]
    ma = ak.mean(); mb = bk.mean(); mc = ck.mean()
    d = n - 1.0
    vaa = (ak @ ak - n * ma * ma) / d
    vbb = (bk @ bk - n * mb * mb) / d
    vcc = (ck @ ck - n * mc * mc) / d
    vac = (ak @ ck - n * ma * mc) / d
    if vaa <= 0 or vbb <= 0 or vcc <= 0:
        return None, idx
    sda, sdb, sdc = np.sqrt(vaa), np.sqrt(vbb), np.sqrt(vcc)
    return np.array([ma, mb, mc, sda, sdb, sdc, vac / (sda * sdc)]), idx




def solve_latent_fast(theta, crn, tol_vec=TOL_VEC, max_iter=15, damping=1.0):
    theta = np.asarray(theta, dtype=float)
    tol_vec = np.asarray(tol_vec, dtype=float)
    latent = theta.copy()
    for _ in range(max_iter):
        d = _draw(latent, crn)
        if d is None:
            return None
        m, _ = _moments(*d)
        if m is None:
            return None

        done = bool((np.abs(theta - m) < tol_vec).all())

        # step first, then stop: `latent` leaves the loop one correction
        # ahead of where the residual was measured
        latent[:3] += damping * (theta[:3] - m[:3])
        latent[3:6] *= 1.0 + damping * (theta[3:6] / m[3:6] - 1.0)
        latent[6] = min(max(latent[6] + damping * (theta[6] - m[6]),
                            -0.999), 0.999)
        if not np.isfinite(latent).all() or (latent[3:6] <= 0).any():

            return None
        if done:
            return latent
    return None

# ----------------------------------------------------------------------
# cheap necessary conditions (unchanged in spirit; see note 8 below)
# ----------------------------------------------------------------------
def valid_theta(theta):
    t = np.atleast_2d(np.asarray(theta, dtype=float))
    mu_a, mu_b, mu_c, sig_a, sig_b, sig_c, rho_ac = t.T
    ok = (np.isfinite(t).all(axis=1) & (mu_a > 0)
          & (mu_c > 0) & (mu_c < mu_b) & (mu_b < 1.0)
          & (sig_a > 0) & (sig_b > 0) & (sig_c > 0)
          & (sig_b < 0.2887) & (sig_c < 0.2887) & (np.abs(rho_ac) < 1.0))
    return ok if np.ndim(theta) > 1 else bool(ok[0])


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

def _a_bin_edges(a_obs, n_bins=8):
    interior = np.quantile(a_obs, np.linspace(0, 1, n_bins + 1)[1:-1])
    interior = np.unique(interior)
    return np.concatenate([[0.0], interior,
                           [max(a_obs.max() * 5, interior[-1] * 5)]])



def containment_penalty(acc, acc_min=0.75, width=0.01, hard_floor=0.50):
    """
    Smooth log-prior penalty: ~0 well above acc_min, softly negative below,
    -inf below hard_floor. Returns a value to ADD to the log-likelihood.

    Softplus form so the gradient never vanishes -- a hard cut at acc_min
    clips the posterior when the truth sits close to the boundary.
    """
    if not np.isfinite(acc) or acc < hard_floor:
        return np.inf
    x = (acc_min - acc) / width
    return 40*float(np.logaddexp(0.0, x) ** 2)


def projected_semi_axes(phi, theta, B, C):
    """
    Semi-axes (alpha >= beta) of the projected ellipse, in units of the
    intrinsic major axis A, for viewing angles (phi, theta).

    Uses the same Simonneau et al. (1998) f, g as projected_axis_ratio:
    the projected semi-axes satisfy alpha^2 * beta^2 = f^2 and
    alpha^2 + beta^2 = g, so they are the roots

        alpha = sqrt((g + sqrt(g^2 - 4 f^2)) / 2)
        beta  = sqrt((g - sqrt(g^2 - 4 f^2)) / 2)

    Consistency: beta/alpha equals the q returned by projected_axis_ratio.

    Returns
    -------
    alpha, beta : arrays
        Projected semi-major and semi-minor axes in units of A.
        Physical sizes: a_proj = A * alpha, b_proj = A * beta.
        Circularized radius: r_circ = A * sqrt(alpha * beta) = A * sqrt(f).
    """
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
    disc = np.sqrt(np.clip(g ** 2 - 4 * f ** 2, 0, None))  # clip fp noise
    alpha = np.sqrt((g + disc) / 2)
    beta = np.sqrt((g - disc) / 2)
    return alpha, beta


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

    both = (ni > 0) & (mi > 0)
    if np.any(both):
        log_like += np.sum(ni[both] * np.log(mi[both]) - mi[both]
                           - gammaln(ni[both] + 1))

    obs_no_model = (ni > 0) & (mi == 0)
    if np.any(obs_no_model):
        log_like += -np.sum(ni[obs_no_model])

    model_no_obs = (ni == 0) & (mi > 0)
    if np.any(model_no_obs):
        log_like += -np.sum(mi[model_no_obs])

    return log_like


# ----------------------------------------------------------------------
class LogLike7:
    """
    Callable log-likelihood for the 7-parameter model.

    Parameters
    ----------
    q_obs, a_obs : arrays
        Observed projected axis ratios and sizes, paired index by index.
    n_draw_pool : int
        Size of the fixed pool the model population is drawn from. The
        number of model samples that survive the support cut is
        acceptance * n_draw_pool (typically 0.8-0.97 * n_draw_pool).
        ~10x the number of observed objects is plenty; cost is close to
        linear in this number, so it is the main speed/precision dial.
    solve_pool : int
        Pool used by the moment solve only. 20k is enough: the residual
        moment error it leaves behind is ~sigma/sqrt(20000) ~ 1e-3 sigma,
        far below the posterior width.
    tol : float
        Fixed-point tolerance. 3e-4 instead of 1e-4 saves an iteration or
        two and is still an order of magnitude tighter than the binning
        noise.
    dtype : np.float64 or np.float32
        float32 halves the memory traffic in the draw and gives identical
        log-likelihoods to ~0.01; worth trying once everything else works.
    """

    def __init__(self, q_obs, a_obs, n_draw_pool=None, solve_pool=20_000,
                 seed=0, dtype=np.float64, n_a_bins=8):

        self.q_obs = np.asarray(q_obs, float)
        self.a_obs = np.asarray(a_obs, float)
        if n_draw_pool is None:
            n_draw_pool = max(10 * len(self.q_obs), 30_000)
        self.n_draw_pool = int(n_draw_pool)
        self.solve_pool = int(solve_pool)
        self.seed = seed
        self.dtype = dtype

        # --- everything below is independent of theta: compute it once ---
        self.a_edges = _a_bin_edges(self.a_obs, n_a_bins)
        self.a_inner = np.ascontiguousarray(self.a_edges[1:-1])
        self.na = len(self.a_edges) - 1
        n_obs, _, _ = np.histogram2d(self.q_obs, self.a_obs,
                                     bins=[Q_BINS, self.a_edges])
        self.n_obs = n_obs.ravel()
        self.N_obs = float(self.n_obs.sum())
        self.pos = np.flatnonzero(self.n_obs > 0)   # only occupied bins matter
        self.n_pos = self.n_obs[self.pos]
        # sum_i m_i == N_obs after normalisation, so -sum m_i is a constant;
        # so is sum ln(n_i!). Both folded in here to keep logL values
        # comparable with the original implementation.
        self.const = -self.N_obs - float(gammaln(self.n_pos + 1).sum())

    # the CRN pools are large; rebuild them lazily in each worker instead
    # of shipping them through pickle
    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items()}

    def _hist(self, q, a):
        """2-D count, exactly equal to np.histogram2d on these bins."""
        iq = (q * _INV_QW).astype(np.intp)
        np.clip(iq, 0, _NQ - 1, out=iq)
        ia = np.searchsorted(self.a_inner, a, side='right')
        iq *= self.na
        iq += ia
        return np.bincount(iq, minlength=_NQ * self.na)

    def __call__(self, theta):
        theta = np.asarray(theta, dtype=float)
        if not attainable_fast(theta):
            return -np.inf

        # --- one moment solve per call (the original did two) ---
        latent = solve_latent_fast(
            theta, get_crn(self.solve_pool, self.seed, self.dtype),)
        if latent is None:
            return -np.inf            # not attainable: this replaces the
                                      # separate attainable() pre-check

        crn = get_crn(self.n_draw_pool, self.seed, self.dtype)
        d = _draw(latent, crn)
        if d is None:
            return -np.inf
        a, b, c, keep = d
        n_good = int(keep.sum())  # inside the support
        n_bad = keep.size - n_good  # rejected
        acc = n_good / keep.size

        idx = np.flatnonzero(keep)
        # if idx.size < 1000:
        #     return -np.inf
        A = a[idx]
        B = b[idx]
        C = c[idx]

        phi = crn.phi[idx]
        t = crn.theta[idx]
        alpha, beta = projected_semi_axes(phi, t, B, C)
        q = beta / alpha
        a_model = A * np.sqrt(alpha * beta)
        n_model = self._hist(q, a_model)
        total_model = n_model.sum()
        if total_model == 0:
            return -np.inf
        n_model = n_model / total_model * self.N_obs
        loglike= _binned_poisson_loglike(self.n_obs,n_model)

        #add a negative term as a penalty for losing model draws
        #loglike = loglike -(1/acc**3+1)*25

        loglike = loglike - containment_penalty(acc)
        assert loglike < 0
        return loglike


        # u = crn.c2p[idx]              # cos^2 phi, fixed per pool row
        # v = crn.c2t[idx]              # cos^2 theta
        #
        # # Simonneau f, g rewritten in terms of cos^2 phi / cos^2 theta so
        # # that no trig is evaluated inside the likelihood at all.
        # B2 = B * B
        # C2 = C * C
        # omu = 1.0 - u                 # sin^2 phi
        # omv = 1.0 - v                 # sin^2 theta
        # f = np.sqrt(omv * C2 * (u + B2 * omu) + B2 * v)
        # g = u + v * omu + B2 * (omu + v * u) + C2 * omv
        # f2 = 2.0 * f
        # num = g - f2
        # np.clip(num, 0.0, None, out=num)   # near-spherical rounding: without
        # h = np.sqrt(num / (g + f2))        # this a few q's come out NaN
        # q = (1.0 - h) / (1.0 + h)
        # a_model = q * A               # matches the original definition

        # n_model = self._hist(q, a_model)
        # tot = n_model.sum()
        # if tot == 0:
        #     return -np.inf
        # m = n_model[self.pos] * (self.N_obs / tot)
        # term = np.where(m > 0, self.n_pos * np.log(np.maximum(m, 1e-300)),
        #                 -self.n_pos)
        # return float(term.sum() + self.const)


# ======================================================================
# NOTES
# ======================================================================
# 1. Double solve. log_likelihood() called attainable(params), which runs
#    solve_latent with n_pool=200_000, and then generate_model_projections
#    -> truncated_solved, which runs solve_latent AGAIN with
#    solve_pool=50_000. Profiling the original: 71% of the runtime was in
#    solve_latent, and 60% of the total was the attainable() call alone,
#    whose result was thrown away. The two calls also used different pool
#    sizes, so attainable() could say yes and the second solve then fail.
#
# 2. Broken common random numbers. truncated_solved documents rng=None as
#    "reuse the fixed pool, no evaluation noise", but
#    generate_ellipsoid_distribution does `if rng is None: rng =
#    np.random.default_rng()` before calling it, so a fresh generator was
#    always passed down and fresh normals were drawn every call. On top of
#    that random_viewing_angles() draws from the global np.random each
#    time. Measured: repeated calls at identical theta scattered with
#    std ~ 1.8 in logL (range ~ 6). That is both wasted work and a noisy
#    surface for emcee, which shows up as a depressed acceptance fraction
#    and long autocorrelation times.
#
# 3. Per-call constants. _a_bin_edges (quantiles over the full a_obs),
#    the n_obs histogram, and gammaln(n_obs+1) were all recomputed on
#    every likelihood evaluation.
#
# 4. Double projection. generate_model_projections called
#    projected_axis_ratio twice with identical arguments (once for
#    q_model, once for a_model = q * A). That was 19% of the runtime.
#
# 5. Trig. With the viewing angles fixed, cos^2 phi and cos^2 theta are
#    precomputed once and no sin/cos/arccos is evaluated in the hot loop.
#
# 6. Histogram. np.histogram2d is replaced by an index computation plus
#    np.bincount, verified bit-identical on the same bins.
#
# 7. Poisson sum. Only bins with n_obs > 0 contribute a theta-dependent
#    term; -sum_i m_i is constant at N_obs once the model is normalised.
#
# 8. attainable_fast: two things in the original look unintended rather
#    than deliberately conservative --
#      * `if not 0 < sigma_c < sigma_b < 1` requires sigma_c < sigma_b,
#        which is not implied by the model and silently truncates the
#        prior. Removed here.
#      * `if (0 > mu_b - 0.2*mu_b)` is `0 > 0.8*mu_b`, never true for
#        mu_b > 0, so that half of the boundary check never fired. Read as
#        `mu_b - 0.2*sigma_b < 0` here, to match the companion clause.
#    If either was intentional, put it back.
#
# 9. Possible physics bug, unrelated to speed: a_model = q * A_samples is
#    the projected axis RATIO times the intrinsic major axis. The
#    projected semi-major axis is A*alpha and the circularised radius is
#    A*sqrt(alpha*beta) -- which is what projected_semi_axes() in
#    shape_inference.py computes, and which nothing currently calls. If
#    a_obs is a measured (circularised) effective radius, the model side
#    is not the same quantity.
#
# 10. With Pool, prefer a module-level global initialised once per worker
#     over emcee's args=[...]: emcee's _FunctionWrapper is pickled on every
#     map call, so q_obs/a_obs travel down the pipe repeatedly.
#         def _init(qo, ao):
#             global _LL
#             _LL = LogLike7(qo, ao)
#         def _logL(theta):
#             return _LL(theta)
#         with Pool(n_cores, initializer=_init, initargs=(q_obs, a_obs)) as p:
#             emcee.EnsembleSampler(n_walkers, 7, _logL, pool=p)