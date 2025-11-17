import numpy as np
import sympy as sp
from itertools import combinations

# ---------------------------------------------------------------------------
# Core geometry helpers
# ---------------------------------------------------------------------------

def find_conic_center(coeffs):
    '''
    Return the center ``(h, k)`` of a general conic
    ``Ax² + Bxy + Cy² + Dx + Ey + F = 0``.

    Raises
    ------
    ValueError
        If the denominator vanishes (parabolic or degenerate conic).
    '''
    A, B, C, D, E, F = coeffs
    denom = B * B - 4.0 * A * C
    if abs(denom) < 1e-15:
        raise ValueError('No finite center (parabolic or degenerate case).')

    h = -(B * E - 2.0 * C * D) / denom
    k = -(B * D - 2.0 * A * E) / denom
    return h, k

def ellipse_axis_ratio(coeffs):
    '''
    Compute the semi-major / semi-minor axis ratio of an ellipse given the
    general conic coefficients A,B,C,D,E,F.

    Parameters
    ----------
    A, B, C, D, E, F : float
        Coefficients of the conic section.

    Returns
    -------
    ratio : float
        a / b  ≥ 1, where a is the semi-major axis and b the semi-minor axis.
    '''
    A, B, C, D, E, F = coeffs
    
    # 2) Coordinates of the center (∂f/∂x = ∂f/∂y = 0)
    x0, y0 = find_conic_center(coeffs)

    # 3) Constant term after translation to the center
    F0 = F + A*x0*x0 + B*x0*y0 + C*y0*y0 + D*x0 + E*y0

    # 4) Quadratic form matrix, scaled so the right-hand side becomes 1
    Q = np.array([[A, B/2.0],
                  [B/2.0, C]], dtype=float) / (-F0)

    # Its positive eigen-values are 1/a² and 1/b²
    lambda1, lambda2 = np.linalg.eigvalsh(Q)          # sorted ascending
    if lambda1 <= 0 or lambda2 <= 0:
        raise ValueError('Semi-axis lengths are not real and positive.')

    # 5) Axis ratio a / b  (≥ 1)
    return np.sqrt(lambda2 / lambda1)

def ellipse_major_axis(coeffs):
    '''
    Compute the length of the major axis of an ellipse given the conic
    coefficients A,B,C,D,E,F.
    Parameters
    ----------
    A, B, C, D, E, F : float
        Coefficients of the conic section.
    Returns
    -------
    length : float
        Length of the major axis.
    '''
    A, B, C, D, E, F = coeffs
    # Quadratic form and linear term
    Q = np.array([[A, B/2],
                  [B/2, C]], dtype=float)
    b = np.array([D, E], dtype=float)

    # Check ellipse conditions: positive definite Q and positive k
    if np.linalg.det(Q) <= 0:
        raise ValueError("Not an ellipse: quadratic form is not positive definite.")

    # Center of the ellipse: solve 2 Q x_c + b = 0
    xc = -0.5 * np.linalg.solve(Q, b)

    # Constant after translation: x'^T Q x' = k
    k = xc @ Q @ xc - F

    # Eigenvalues of Q (principal curvatures)
    eigvals = np.linalg.eigvalsh(Q)          # sorted ascending
    a_sq = k / eigvals                       # squared semi-axes (a1^2, a2^2)
    if np.any(a_sq <= 0):
        raise ValueError("Not an ellipse or degenerate: semi-axis squared <= 0.")

    a, b_sem = np.sqrt(a_sq)                 # semi-axis lengths (a <= b or vice versa)
    major = max(a, b_sem)

    return 2 * major  # full major axis length

def ellipse_tangent_slope(coeffs, center, tol=1e-4):
    '''
    Return the slope dy/dx of the tangent line to the ellipse
    A x² + B x y + C y² + D x + E y + F = 0
    at the point (x, y).

    Parameters
    ----------
    coeffs : list[float]
        Conic coefficients (B² - 4AC < 0 for a real ellipse).
    center : tuple[float, float]
        Coordinates of a point lying on the ellipse.
    tol : float, optional
        Tolerance for point-on-curve and vertical-tangent checks.

    Returns
    -------
    m : float
        Slope of the tangent line.  `np.inf` if the tangent is vertical.

    Raises
    ------
    ValueError
        If (x, y) is not on the ellipse within `tol`.
    '''
    A, B, C, D, E, F = coeffs
    x, y = center

    # 1) Verify the point is (numerically) on the ellipse
    # if np.abs(A*x**2 + B*x*y + C*y**2 + D*x + E*y + F) > tol:
    #     raise ValueError('(x,y) does not satisfy the ellipse equation.')

    # 2) Partial derivatives (gradient of f)
    fx = 2*A*x + B*y + D          # ∂f/∂x
    fy = B*x + 2*C*y + E          # ∂f/∂y

    # 3) Slope from implicit-function theorem: dy/dx = –(∂f/∂x)/(∂f/∂y)
    if np.abs(fy) < tol:             # vertical tangent
        return np.inf           # or np.inf

    return -fx / fy

def get_asymptotes(coeffs):
    '''
    Compute slopes ``m₁, m₂`` and intercepts ``b₁, b₂`` of the two asymptotes
    of a hyperbola.

    Returns
    -------
    tuple[tuple, tuple]
        ``((m₁, b₁), (m₂, b₂))``
    '''
    h, k = find_conic_center(coeffs)
    A, B, C, D, E, F = coeffs
    disc = B * B - 4.0 * A * C
    denom = 2.0 * C
    m1 = (-B + np.sqrt(disc)) / denom
    m2 = (-B - np.sqrt(disc)) / denom
    return (m1, k - m1 * h), (m2, k - m2 * h)

def get_major_side_flag(images, m1, b1, m2, b2):
    '''
    Determine on which half-planes (relative to each asymptote) the *major*
    branch of a two-branched hyperbola lies.

    Parameters
    ----------
    images : (4, 2) ndarray
        Quad-image coordinates.
    m1, b1, m2, b2 : float
        Slopes and intercepts of the asymptotes.

    Returns
    -------
    list[bool]
        Two Boolean flags, one per asymptote.  ``True`` means the relevant
        hyperbola branch lies above the asymptote in question.
    '''
    above1 = images[:, 1] > (m1 * images[:, 0] + b1)
    above2 = images[:, 1] > (m2 * images[:, 0] + b2)
    ab_mat = np.vstack((above1, above2)).T

    # Partition quad images by matching (above/below) signatures
    uniq, idx = np.unique(ab_mat, axis=0, return_inverse=True)
    pairs = [np.where(idx == i)[0] for i in range(len(uniq))]
    p1, p2 = images[pairs[0]], images[pairs[1]]

    if np.linalg.norm(p1[0] - p1[1]) > np.linalg.norm(p2[0] - p2[1]):
        return ab_mat[pairs[0][0]].tolist()
    return ab_mat[pairs[1][0]].tolist()

def create_side_flag_mask(xs, ys, ms, bs, side_flags):
    '''
    Build a Boolean mask selecting points that lie on a specified side of two
    lines (the hyperbola asymptotes).

    Returns
    -------
    (N,) ndarray of bool
    '''
    mask = np.ones(xs.shape[0], bool)
    for m, b, flag in zip(ms, bs, side_flags):
        mask &= (ys > m * xs + b) == flag
    return mask

def point_to_hyperbola_distance(x0, y0, coeffs,
                                return_nearest_point=False, tol=1e-12):
    '''
    Shortest Euclidean distance from a point (x0, y0) to a general conic
    that is a hyperbola.  Uses the Lagrange-multiplier formulation.

    Parameters
    ----------
    x0, y0 : float
        Coordinates of the external point.
    coeffs : iterable
        Conic coefficients (A, B, C, D, E, F) for  A x² + B xy + C y² + D x + E y + F = 0.
    return_nearest_point : bool, default False
        If True, return both (distance, (x_near, y_near));
        else return only the distance.
    tol : float, default 1e-12
        Numerical tolerance for discarding spurious roots.

    Returns
    -------
    float or (float, tuple)
        Either the minimal distance, or `(distance, (x_nearest, y_nearest))`
        depending on *return_nearest_point*.
    '''
    A, B, C, D, E, F = coeffs

    lam = sp.symbols('lam')
    detM = (1 - 2*A*lam)*(1 - 2*C*lam) - (B*lam)**2
    x_num = (x0 + lam*D)*(1 - 2*C*lam) + lam*B*(y0 + lam*E)
    y_num = (1 - 2*A*lam)*(y0 + lam*E) + lam*B*(x0 + lam*D)
    x_expr = sp.simplify(x_num / detM)
    y_expr = sp.simplify(y_num / detM)

    g_expr = A*x_expr**2 + B*x_expr*y_expr + C*y_expr**2 + D*x_expr + E*y_expr + F
    numer = sp.expand(sp.together(g_expr).as_numer_denom()[0])
    poly = sp.Poly(numer, lam)

    # Real candidate λ values
    roots = [r.real for r in np.roots([float(c) for c in poly.all_coeffs()])
             if abs(r.imag) < tol]

    d2_pts = []   # list of (d², x, y)
    for lam_val in roots:
        det_val = (1 - 2*A*lam_val)*(1 - 2*C*lam_val) - (B*lam_val)**2
        if abs(det_val) < tol:
            continue
        x_val = ((x0 + lam_val*D)*(1 - 2*C*lam_val)
                 + lam_val*B*(y0 + lam_val*E)) / det_val
        y_val = ((1 - 2*A*lam_val)*(y0 + lam_val*E)
                 + lam_val*B*(x0 + lam_val*D)) / det_val
        d2_pts.append(((x_val - x0)**2 + (y_val - y0)**2, x_val, y_val))

    if not d2_pts:
        raise RuntimeError('No valid stationary point found; verify B²−4AC > 0.')

    # Choose the smallest-distance candidate
    d2_min, x_near, y_near = min(d2_pts, key=lambda t: t[0])
    dist = np.sqrt(d2_min)

    if return_nearest_point:
        return dist, (float(x_near), float(y_near))
    return dist


# ---------------------------------------------------------------------------
# Conic construction helpers
# ---------------------------------------------------------------------------

def generate_hyperbola(points):
    '''
    Return coefficients ``(A, B, C, D, E, F)`` of the hyperbola passing
    through four *quad* images (`F` is fixed to ``+1``).

    Raises
    ------
    RuntimeError
        If the Vandermonde matrix is singular (collinear or ill-posed input).
    '''
    M = np.column_stack([
        points[:, 0] ** 2 - points[:, 1] ** 2,
        points[:, 0] * points[:, 1],
        points[:, 0],
        points[:, 1],
    ])
    b = -np.ones(4)
    try:
        A, B, D, E = np.linalg.solve(M, b)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError('generate_hyperbola: singular matrix.') from exc

    C = -A
    return np.array([A, B, C, D, E, 1.0])

# ---------------------------------------------------------------------
# helper 1: asymptote slopes of a general‑position hyperbola
# ---------------------------------------------------------------------
def _asymptote_slopes(h_coeffs, tol=1e-12):
    """
    Return the two slopes (m1, m2) of the asymptotes of
        A x² + B x y + C y² + … = 0.
    """
    A, B, C = h_coeffs[:3]
    Δ = B * B - 4 * A * C                      # discriminant  (Δ > 0 → hyperbola)
    # if Δ <= tol:
    #     raise ValueError("Quadratic part is not a hyperbola (Δ ≤ 0).")

    if abs(C) > tol:                           # generic case
        root = np.sqrt(Δ)
        m1 = (-B + root) / (2 * C)
        m2 = (-B - root) / (2 * C)
    else:                   
        # C ≈ 0  → one asymptote vertical
        print('Warning: C ≈ 0, using vertical asymptote.')
        if abs(B) < tol:
            raise ValueError("Degenerate quadratic part: A≈C≈0.")
        m1 = -A / B                            # finite‑slope branch  (B m + A = 0)
        m2 = np.inf                            # vertical branch  (x = const)
    return (m1, m2)


# ---------------------------------------------------------------------
# helper 2: ellipse through 4 points with axis slope = m
# ---------------------------------------------------------------------
def _ellipse_from_slope(points, m, normalization='F'):
    """Return (coeff_dict, centre_xy) for the unique ellipse that
       passes through four points and has a principal axis of slope m."""
    pts = np.asarray(points, dtype=float)
    if pts.shape != (4, 2):
        raise ValueError("Exactly four points (shape (4,2)) are required.")

    # (i) point‑through linear constraints
    M = [[x**2, x*y, y**2, x, y, 1.0] for x, y in pts]

    # (ii) orientation constraint  (B - k (A-C) = 0,  with  k = tan 2θ)
    if np.isinf(m):                                # vertical axis  → B = 0
        orient = [0, 1, 0, 0, 0, 0]
    else:
        denom = 1.0 - m * m
        if abs(denom) < 1e-12:                  # θ ≈ 45° → A − C = 0
            orient = [1, 0, -1, 0, 0, 0]
        else:
            k = 2.0 * m / denom
            orient = [-k, 1, k, 0, 0, 0]
    M.append(orient)
    M = np.array(M, dtype=float)                # 5×6 matrix

    # homogeneous solution via SVD
    U, S, Vh = np.linalg.svd(M)
    if (S > 1e-10).sum() != 5:
        raise ValueError("Constraints are degenerate – no unique ellipse.")

    p = Vh[-1, :]                               # null‑space basis

    # scale
    if normalization == 'F':
        p /= p[-1] if abs(p[-1]) > 1e-12 else np.linalg.norm(p)
    else:
        p /= np.linalg.norm(p)

    coeffs = p

    # centre (x₀,y₀) solves ∇Q = 0 → [2A  B;  B  2C] [x₀; y₀] = -[D; E]
    A, B, C, D, E, _ = p
    Mcent = np.array([[2*A, B], [B, 2*C]])
    centre = np.linalg.solve(Mcent, -np.array([D, E]))
    return coeffs, centre


# ---------------------------------------------------------------------
# helper 3: normal‑direction distance from a point to a conic
# ---------------------------------------------------------------------
def _distance_point_to_conic(point, h_coeffs):
    """First‑order (normal) Euclidean distance from `point` to the conic
       Q_h(x,y)=0 defined by h_coeffs."""
    x, y = point
    A, B, C, D, E, F = h_coeffs
    Q = A*x*x + B*x*y + C*y*y + D*x + E*y + F
    Gx = 2*A*x + B*y + D                      # ∂Q/∂x
    Gy = B*x + 2*C*y + E                      # ∂Q/∂y
    grad_norm = np.sqrt(Gx*Gx + Gy*Gy)
    return abs(Q) / grad_norm if grad_norm > 0 else np.inf


# ---------------------------------------------------------------------
# main façade
# ---------------------------------------------------------------------
def ellipse_coefficients(points, hyperbola_coeffs, normalization='F', return_meta=False):
    """
    Determine the ellipse that
       • passes through 4 given points, and
       • has one of its principal axes aligned with *either* asymptote
         of the supplied hyperbola,
    then chooses the alignment whose ellipse centre is *closer to the
    hyperbola* (normal distance).  That ellipse’s coefficients are returned.

    Parameters
    ----------
    points : array‑like, shape (4,2) – the interpolation points.
    hyperbola_coeffs : array‑like, length 6 – (A_h, B_h, C_h, D_h, E_h, F_h).
    normalization : {'F','norm'}, default 'F'
        Scaling convention for the ellipse coefficients.
    return_meta : bool, default False
        If True, also return diagnostics:
            {'coeffs': … , 'centre': (x,y), 'asymptote_slope': m,
             'distance': d, 'other': {...}}.

    Returns
    -------
    coeffs : dict   (or meta dict if return_meta=True)
        (A,B,C,D,E,F) of the selected ellipse.
    """
    h_coeffs = np.asarray(hyperbola_coeffs, dtype=float)
    slopes = _asymptote_slopes(h_coeffs)

    candidates = []
    for m in slopes:
        coeffs, centre = _ellipse_from_slope(points, m, normalization)
        dist = _distance_point_to_conic(centre, h_coeffs)
        candidates.append({'coeffs': coeffs,
                           'centre': centre,
                           'slope': m,
                           'distance': dist})

    best = min(candidates, key=lambda d: d['distance'])
    if return_meta:
        best['other'] = [c for c in candidates if c is not best]
        return best
    return best['coeffs']

def ellipse_from_triplet(points, hyperbola_coeffs, residual_tol=1e-6,
                         angle_window=np.deg2rad(25.0), samples=241,
                         center_point=None, return_meta=False):
    """
    Construct an ellipse centred on the hyperbola, whose principal axes are
    approximately aligned with the hyperbola asymptotes, and which interpolates
    the supplied triplet.  The solver searches over small angular deviations
    around each asymptote to find a rotation that yields a consistent ellipse.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError('Input points must be an (N, 2) array.')
    if pts.shape[0] < 3:
        raise ValueError('At least three points are required to determine the ellipse.')

    if center_point is None:
        centre = np.array(find_conic_center(hyperbola_coeffs), dtype=float)
    else:
        centre = np.array(center_point, dtype=float)
    (m1, _), (m2, _) = get_asymptotes(hyperbola_coeffs)
    slopes = (m1, m2)

    def _slope_to_angle(m):
        if np.isinf(m):
            return 0.5 * np.pi
        return np.arctan(m)

    def _line_angle(theta):
        return (theta + 0.5 * np.pi) % np.pi - 0.5 * np.pi

    def _slope_from_vec(vec):
        if abs(vec[0]) < 1e-12:
            return np.inf
        return vec[1] / vec[0]

    def _build_candidate(theta, primary_idx, pair):
        u = np.array([np.cos(theta), np.sin(theta)])
        v = np.array([-np.sin(theta), np.cos(theta)])
        R = np.column_stack((u, v))
        coords = (pts - centre) @ R
        u2 = coords[:, 0] ** 2
        v2 = coords[:, 1] ** 2

        i, j = pair
        det = u2[i] * v2[j] - u2[j] * v2[i]
        if abs(det) < 1e-12:
            return None

        alpha = (v2[j] - v2[i]) / det
        beta = (u2[i] - u2[j]) / det
        if alpha <= 0 or beta <= 0:
            return None

        residuals = alpha * u2 + beta * v2 - 1.0
        residuals[i] = residuals[j] = 0.0
        residual = float(np.max(np.abs(residuals)))

        Lambda = np.diag([alpha, beta])
        M = R @ Lambda @ R.T

        A = M[0, 0]
        B = 2.0 * M[0, 1]
        C = M[1, 1]
        D = -2.0 * (M[0, 0] * centre[0] + M[0, 1] * centre[1])
        E = -2.0 * (M[0, 1] * centre[0] + M[1, 1] * centre[1])
        F = centre @ (M @ centre) - 1.0

        a = 1.0 / np.sqrt(alpha)
        b = 1.0 / np.sqrt(beta)

        primary_slope = _slope_from_vec(u)
        secondary_slope = _slope_from_vec(v)
        other_slope = slopes[1 - primary_idx]
        if np.isinf(other_slope) and np.isinf(secondary_slope):
            slope_mismatch = 0.0
        elif np.isinf(other_slope) or np.isinf(secondary_slope):
            slope_mismatch = np.inf
        else:
            slope_mismatch = abs(other_slope - secondary_slope)

        third_idx = next(iter({0, 1, 2} - set(pair)))

        return {
            'coeffs': np.array([A, B, C, D, E, F]),
            'axes': (a, b),
            'centre': centre,
            'rotation': R,
            'theta': theta,
            'primary_slope': primary_slope,
            'secondary_slope': secondary_slope,
            'slope_mismatch': slope_mismatch,
            'residual': residual,
            'residuals': residuals,
            'third_index': third_idx,
            'third_residual': residuals[third_idx],
            'approximate': residual > residual_tol,
        }

    def _refine(theta_lo, theta_hi, val_lo, val_hi, primary_idx, pair):
        best = None
        for _ in range(60):
            theta_mid = 0.5 * (theta_lo + theta_hi)
            cand = _build_candidate(theta_mid, primary_idx, pair)
            if cand is None:
                break
            val_mid = cand['third_residual']
            if abs(val_mid) < residual_tol:
                cand['approximate'] = False
                return cand
            if best is None or abs(val_mid) < abs(best['third_residual']):
                best = cand
            if val_lo * val_mid < 0:
                theta_hi, val_hi = theta_mid, val_mid
            else:
                theta_lo, val_lo = theta_mid, val_mid
        return best

    theta1 = _slope_to_angle(slopes[0])
    theta2 = _slope_to_angle(slopes[1])
    angle_guesses = [
        theta1,
        theta2,
        _line_angle(0.5 * (_line_angle(theta1) + _line_angle(theta2 - 0.5 * np.pi))),
        _line_angle(0.5 * (_line_angle(theta2) + _line_angle(theta1 - 0.5 * np.pi))),
    ]

    best_candidate = None
    for guess_idx, theta_guess in enumerate(angle_guesses):
        primary_idx = 0 if guess_idx in (0, 2) else 1
        theta_vals = np.linspace(theta_guess - angle_window,
                                 theta_guess + angle_window,
                                 samples)
        for pair in combinations(range(pts.shape[0]), 2):
            prev_theta = None
            prev_val = None
            for theta in theta_vals:
                cand = _build_candidate(theta, primary_idx, pair)
                if cand is None:
                    continue
                val = cand['third_residual']
                if abs(val) < residual_tol and cand['residual'] <= residual_tol:
                    cand['approximate'] = False
                    return cand if return_meta else cand['coeffs']
                if prev_val is not None and val * prev_val < 0:
                    refined = _refine(prev_theta, theta, prev_val, val,
                                      primary_idx, pair)
                    if refined is not None:
                        if refined['residual'] <= residual_tol:
                            return refined if return_meta else refined['coeffs']
                        if (best_candidate is None or
                                refined['residual'] < best_candidate['residual']):
                            best_candidate = refined
                if best_candidate is None or cand['residual'] < best_candidate['residual']:
                    best_candidate = cand
                prev_theta, prev_val = theta, val

    if best_candidate is None:
        raise ValueError('No valid ellipse configuration found (insufficient point geometry).')

    return best_candidate if return_meta else best_candidate['coeffs']

# ---------------------------------------------------------------------------
# Intersection / transform utilities
# ---------------------------------------------------------------------------

def conic_intersections(coeffs1, coeffs2,
                        upper_tol=1e-3, lower_tol=1e-14):
    '''
    Return real intersection points (N×2 ndarray) of two conics.
    '''
    A1, B1, C1, D1, E1, F1 = coeffs1
    A2, B2, C2, D2, E2, F2 = coeffs2

    def resultant_at_x(x):
        p1 = np.array([C1, B1 * x + E1, A1 * x * x + D1 * x + F1])
        p2 = np.array([C2, B2 * x + E2, A2 * x * x + D2 * x + F2])
        S = np.array([
            [p1[0], p1[1], p1[2], 0.0],
            [0.0, p1[0], p1[1], p1[2]],
            [p2[0], p2[1], p2[2], 0.0],
            [0.0, p2[0], p2[1], p2[2]],
        ])
        return np.linalg.det(S)

    xs_sample = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    coeffs = np.polyfit(xs_sample,
                        [resultant_at_x(x) for x in xs_sample], 4)
    x_roots = np.roots(coeffs)

    candidates = []
    for x in x_roots:
        a, b = C1, B1 * x + E1
        c = A1 * x * x + D1 * x + F1
        if abs(a) > lower_tol:
            disc = b * b - 4.0 * a * c
            y_roots = [(-b + np.sqrt(disc + 0j)) / (2.0 * a),
                       (-b - np.sqrt(disc + 0j)) / (2.0 * a)]
        elif abs(b) > lower_tol:
            y_roots = [-c / b]
        else:
            continue
        for y in y_roots:
            if abs(A2 * x * x + B2 * x * y + C2 * y * y
                   + D2 * x + E2 * y + F2) < upper_tol:
                candidates.append((x, y))

    uniq = []
    for p in candidates:
        if not any(np.allclose(p, q, atol=lower_tol) for q in uniq):
            uniq.append(p)

    pts = np.array(uniq, complex).reshape(-1, 2)
    return pts[np.all(np.isclose(pts.imag, 0.0, atol=upper_tol), axis=1)].real

# ---------------------------------------------------------------------------
# Plot / graph helpers
# ---------------------------------------------------------------------------

def get_graph_limits(points, pad=0.5):
    '''
    Get appropriate limits for linespace and diagrams.
    '''
    mins = np.min(points, axis=0)
    ranges = np.ptp(points, axis=0)
    limits = np.stack((mins - pad * ranges,
                        mins + (1 + pad) * ranges))
    return limits

def generate_conic_linspaces(coeffs, n=5000,
                             x_min=-2.0, x_max=2.0, tol=1e-20):
    '''
    Generate *x-y* lower/upper envelopes of a conic section for plotting.
    '''
    A, B, C, D, E, F = coeffs
    xs = np.linspace(x_min, x_max, n)
    a = C
    b = B * xs + E
    c = A * xs ** 2 + D * xs + F
    disc = b ** 2 - 4.0 * a * c
    mask = disc >= -tol
    disc[~mask] = np.nan
    sqrt_d = np.sqrt(np.clip(disc, 0.0, None))
    denom = 2.0 * a
    y_low = np.full_like(xs, np.nan)
    y_up = np.full_like(xs, np.nan)
    y_low[mask] = (-b[mask] - sqrt_d[mask]) / denom
    y_up[mask] = (-b[mask] + sqrt_d[mask]) / denom
    return xs, y_low, xs, y_up

import numpy as np

def ellipse_linspace(A, B, C, D, E, F, num=512, endpoint=True, tol=None):
    """
    Return x, y arrays that trace the ellipse defined by
        A x^2 + B x y + C y^2 + D x + E y + F = 0
    as a smooth, closed polyline. Robust to overall sign flips and tiny
    numerical negatives in the quadratic form.

    Parameters
    ----------
    A, B, C, D, E, F : float
        Conic coefficients (must define a real, non-degenerate ellipse).
    num : int
        Number of segments around the ellipse (default 512).
    endpoint : bool
        If True, repeats the first point at the end (closed loop).
    tol : float or None
        Numerical tolerance; default scales with coefficient magnitudes.

    Returns
    -------
    x, y : ndarray
        Coordinates parameterized monotonically in angle (no jumps).
    """
    # Build initial quadratic form and linear term
    M = np.array([[A, B/2.0],
                  [B/2.0, C]], dtype=float)
    d = np.array([D, E], dtype=float)

    # Choose a scale-aware tolerance if not provided
    if tol is None:
        s = max(1.0, np.linalg.norm(M, ord=2))
        tol = 1e-12 * s

    # If M is negative definite, flip the entire conic (it’s the same ellipse)
    w0, _ = np.linalg.eigh(M)
    if (w0 < 0).all():
        A, B, C, D, E, F = -A, -B, -C, -D, -E, -F
        M = -M
        d = -d
        w0 = -w0

    # Recompute eigendecomposition after potential flip
    w, R = np.linalg.eigh(M)  # eigenvalues ascending
    # If clearly indefinite (one pos, one neg beyond tolerance) -> not an ellipse
    if w.min() < -tol and w.max() > tol:
        raise ValueError("Coefficients describe an indefinite quadratic form (not an ellipse).")
    # Clip tiny negatives to zero for numerical robustness
    if w.min() < tol:
        w = np.maximum(w, tol)

    # Ellipse center solves (2M)c + d = 0
    c = -0.5 * np.linalg.solve(M, d)

    # Constant after translating to the center: u^T M u + F_c = 0
    F_c = F + float(c @ (M @ c)) + float(d @ c)
    rhs = -F_c
    if rhs < -tol:
        raise ValueError("No real ellipse (rhs < 0 after centering).")
    if rhs < tol:
        raise ValueError("Degenerate ellipse (collapses to a point).")

    # Semi-axes in principal frame: (u1^2)/(rhs/w1) + (u2^2)/(rhs/w2) = 1
    a = np.sqrt(rhs / w[0])
    b = np.sqrt(rhs / w[1])

    # Make rotation a proper rotation (det=+1) for consistent orientation
    if np.linalg.det(R) < 0:
        R[:, 1] *= -1

    # Sample uniformly in angle, close the loop if requested
    theta = np.linspace(0.0, 2.0*np.pi, num + int(endpoint), endpoint=endpoint)

    # Points in principal axes, then rotate and translate to original frame
    V = np.vstack((a * np.cos(theta), b * np.sin(theta)))  # (2, N)
    U = R @ V
    x = c[0] + U[0]
    y = c[1] + U[1]

    return x, y

def generate_linear_linspaces(m, b, n=5000, x_min=-2.0, x_max=2.0):
    '''
    Simple helper: return ``(x, y)`` points for the line ``y = m x + b``.
    '''
    xs = np.linspace(x_min, x_max, n)
    return xs, m * xs + b
