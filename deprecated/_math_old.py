import numpy as np
import sympy as sp

def find_conic_center(coeffs):
    '''
    Find the center of a conic.
    '''
    A, B, C, D, E, F = coeffs
    denom = B * B - 4 * A * C
    
    if np.abs(denom) < 1e-15:
        raise ValueError('No finite center (parabolic or degenerate case).')

    h = -(B * E - 2 * C * D) / denom
    k = -(B * D - 2 * A * E) / denom
    
    return h, k

def get_asymptotes(coeffs):
    '''
    Get the values for m and b for asymptotes of a hyperbola.
    '''
    h, k = find_conic_center(coeffs)
    A, B, C, D, E, F = coeffs
    
    disc = B**2 - 4 * A * C
    denom = 2 * C
    
    m1 = (-B + np.sqrt(disc)) / denom
    m2 = (-B - np.sqrt(disc)) / denom
    b1 = k - m1 * h
    b2 = k - m2 * h
    
    return (m1, b1), (m2, b2)

def get_major_side_flag(images, m1, b1, m2, b2):
    '''
    Determine which side of the asymptotes the major branch of the hyperbola is on.
    '''
    above_asymptote1 = images[:, 1] > (m1 * images[:, 0] + b1)
    above_asymptote2 = images[:, 1] > (m2 * images[:, 0] + b2)
    above_below_matrix = np.vstack((above_asymptote1, above_asymptote2)).T
    
    # Find indices of rows that are identical in the above_below_matrix
    unique_rows, indices = np.unique(above_below_matrix, axis=0, return_inverse=True)
    pair_indices = [np.where(indices == i)[0] for i in range(len(unique_rows))]
    pair1 = images[pair_indices[0]]
    pair2 = images[pair_indices[1]]
    
    pair1_dist = np.linalg.norm(pair1[0] - pair1[1])
    pair2_dist = np.linalg.norm(pair2[0] - pair2[1])
    
    if pair1_dist > pair2_dist:
        return above_below_matrix[pair_indices[0][0]]
    else:
        return above_below_matrix[pair_indices[1][0]]

def create_side_flag_mask(xs, ys, ms, bs, side_flags):
    '''
    Create a mask for points based on their position relative to the asymptotes.
    '''
    total_mask = np.ones(xs.shape[0], dtype=bool)
    for i in range(len(ms)):
        mask = (ys > ms[i] * xs + bs[i]) == side_flags[i]
        total_mask &= mask
    return total_mask

def point_to_hyperbola_distance(x0, y0, coeffs, tol=1e-12):
    '''
    Find the distance from a point (x0, y0) to a hyperbola defined by the coefficients.
    '''

    A, B, C, D, E, F = coeffs
    # Step 1: build x(λ), y(λ) in closed form
    lam = sp.symbols('lam')
    detM = (1 - 2*A*lam)*(1 - 2*C*lam) - (B*lam)**2
    x_num = (x0 + lam*D)*(1 - 2*C*lam) + lam*B*(y0 + lam*E)
    y_num = (1 - 2*A*lam)*(y0 + lam*E) + lam*B*(x0 + lam*D)
    x_expr = sp.simplify(x_num / detM)
    y_expr = sp.simplify(y_num / detM)

    # Step 2: quartic in λ
    g_expr   = A*x_expr**2 + B*x_expr*y_expr + C*y_expr**2 + D*x_expr + E*y_expr + F
    numer    = sp.expand(sp.together(g_expr).as_numer_denom()[0])   # clear denom.
    poly     = sp.Poly(numer, lam)                                  # quartic
    coeffs   = [float(c) for c in poly.all_coeffs()]                # a₀…a₄

    # Step 3: solve p(λ)=0 numerically
    cand_d2  = []
    for root in np.roots(coeffs):
        if abs(root.imag) > tol:        # keep real roots only
            continue
        lam_val = root.real
        det_val = (1 - 2*A*lam_val)*(1 - 2*C*lam_val) - (B*lam_val)**2
        if abs(det_val) < tol:          # singular matrix → skip
            continue
        x_val = ((x0 + lam_val*D)*(1 - 2*C*lam_val)
                 + lam_val*B*(y0 + lam_val*E)) / det_val
        y_val = ((1 - 2*A*lam_val)*(y0 + lam_val*E)
                 + lam_val*B*(x0 + lam_val*D)) / det_val
        d2    = (x_val - x0)**2 + (y_val - y0)**2
        cand_d2.append(d2)

    if not cand_d2:
        raise RuntimeError("No valid stationary point found; "
                           "check that the conic is a hyperbola "
                           "and B² - 4AC > 0.")

    return np.sqrt(min(cand_d2))

def generate_hyperbola(points):
    '''
    Find conic coefficients for hyperbola passing through 4 points.
    '''
    # Build the 4×4 basis functions matrix 
    M = np.column_stack([
        points[:, 0]**2 - points[:, 1]**2,  # x^2 − y^2
        points[:, 0] * points[:, 1],        # x·y
        points[:, 0],                       # x
        points[:, 1]                        # y
    ])
    
    # Assumes F = 1
    b = -np.ones(4)

    # Solve M · [A, B, D, E]^T = b
    try:
        coeffs = np.linalg.solve(M, b)
    except np.linalg.LinAlgError as exc:
        print(f'{M = }')
        print(f'{b = }')
        raise RuntimeError(
            'Singularity Found'
        ) from exc
        
    coeffs = np.insert(coeffs, 2, -coeffs[0])  # C = -A
    coeffs = np.append(coeffs, 1.0)  # F = 1
        
    return coeffs

def generate_ellipse(points, k):
    '''
    Given three points (xi, yi) that lie on
        (x - xs)^2/a^2 + (y - (k/xs))^2/b^2 = 1
    return all real (a, b, xs) triplets satisfying the curve.
    '''
    (x1, y1), (x2, y2), (x3, y3) = points

    A12 = np.poly1d([-2*(x1 - x2),  x1**2 - x2**2])     # degree 1
    A32 = np.poly1d([-2*(x3 - x2),  x3**2 - x2**2])

    B32 = np.poly1d([y3**2 - y2**2,  -2*k*(y3 - y2),  0.0])
    B12 = np.poly1d([y1**2 - y2**2,  -2*k*(y1 - y2),  0.0])

    # determinant polynomial (already multiplied by x_s²)
    det_poly = np.polysub(np.polymul(A12, B32), np.polymul(A32, B12))
    det_poly = np.poly1d(det_poly)          # strips any leading zeros

    # if x_s = 0 appears, discard that root (center cannot be at x = 0)
    coeffs = det_poly.c
    while abs(coeffs[0]) < 1e-12 and len(coeffs) > 1:
        coeffs = coeffs[1:]
    det_poly = np.poly1d(coeffs)

    # 3. solve Δ̂(x_s) = 0   (real roots only, x_s ≠ 0)
    xs_roots = [r.real for r in det_poly.r
                if abs(r.imag) < 1e-10 and abs(r.real) > 1e-12]

    # 4. back-substitute for u = 1/a², v = 1/b²  →  a, b
    solutions = []
    xi = np.array([x1, x2, x3])
    yi = np.array([y1, y2, y3])

    for xs in xs_roots:
        A = (xi - xs)**2
        B = (yi - k / xs)**2
        # solve two of the three linear equations  A·u + B·v = 1
        u, v = np.linalg.solve(np.column_stack([A[:2], B[:2]]), np.ones(2))
        if u > 0 and v > 0:                       # real semi-axes only
            a = 1 / np.sqrt(u)
            b = 1 / np.sqrt(v)
            solutions.append((a, b, xs))

    return solutions

def generate_optimal_ellipse(points,
                             hyperbola_coeffs,
                             align_weight=1e10,
                             nsamples=7200):
    '''
    Gives the general conic coefficients for ellipse which passes through 4 points.
    While optimizing have its center near hyperbola and axes aligned with hyperbola asymptotes.
    '''

    # ---------------- 1. pencil through the four points ---------------------
    P        = np.asarray(points, float)
    X, Y     = P[:, 0], P[:, 1]
    M        = np.column_stack([X**2, X*Y, Y**2, X, Y, np.ones_like(X)])
    *_, vh   = np.linalg.svd(M)
    N        = vh[-2:].T                          # 6×2 basis

    # ---------------- 2. hyperbola asymptote angles -------------------------
    Ah, Bh, Ch, Dh, Eh, Fh = hyperbola_coeffs
    m1, m2 = np.roots([Ch, Bh, Ah])               # slopes
    theta1, theta2 = np.arctan(m1), np.arctan(m2) # directions

    def angle_err(phi):
        """smallest |phi - theta_k| mod π"""
        diffs = [abs((phi - th + np.pi/2) % np.pi - np.pi/2)
                 for th in (theta1, theta2)]
        return min(diffs)

    # ---------------- 3. helpers for any conic in the pencil ----------------
    def coeffs(t):        # t = (cosθ, sinθ)
        return N @ t

    def is_ellipse(c, eps=1e-12):
        A, B, C = c[:3]
        return (B*B - 4*A*C < -eps) and (A*C > 0)

    def center(c):
        A,B,C,D,E,_ = c
        den = B*B - 4*A*C
        return ((2*C*D - B*E)/den, (2*A*E - B*D)/den)

    def H(x, y):
        return Ah*x*x + Bh*x*y + Ch*y*y + Dh*x + Eh*y + Fh

    # ---------------- 4. objective to minimise ------------------------------
    def cost(t):
        c = coeffs(t)
        # if not is_ellipse(c):
        #     return np.inf
        h, k   = center(c)
        phi_e  = 0.5*np.arctan2(c[1], c[0]-c[2])   # ellipse axis angle
        return H(h, k)**2 + align_weight*angle_err(phi_e)**2

    # ---------------- 5. coarse sweep for a good seed -----------------------
    angles  = np.linspace(0, 2*np.pi, nsamples, endpoint=False)
    best_t, best_f = None, np.inf
    for θ in angles:
        t = np.array([np.cos(θ), np.sin(θ)])
        f = cost(t)
        if f < best_f:
            best_t, best_f = t, f
    if best_t is None:
        raise RuntimeError("No ellipse passes through the four points.")

    # ---------------- 6. golden-section refine that single angle ------------
    def f_theta(theta):
        return cost(np.array([np.cos(theta), np.sin(theta)]))

    step = 2*np.pi/nsamples
    a = np.arctan2(best_t[1], best_t[0]) - step
    b = a + 2*step
    φg = (np.sqrt(5)-1)/2
    c1, c2 = b - φg*(b-a), a + φg*(b-a)
    f1, f2 = f_theta(c1), f_theta(c2)
    for _ in range(60):
        if f1 < f2:
            b, c2, f2 = c2, c1, f1
            c1, f1    = b - φg*(b-a), f_theta(b - φg*(b-a))
        else:
            a, c1, f1 = c1, c2, f2
            c2, f2    = a + φg*(b-a), f_theta(a + φg*(b-a))
    θ_opt = 0.5*(a + b)
    c_opt = coeffs(np.array([np.cos(θ_opt), np.sin(θ_opt)]))

    # ---------------- 7. scale so F = –1 and return -------------------------
    if abs(c_opt[-1]) > 1e-12:
        c_opt /= -c_opt[-1]
        
    return tuple(c_opt)

def conic_intersections(coeffs1, coeffs2, upper_tol=1e-3, lower_tol=1e-14):
    '''
    Find intersection points of two conics given by their coefficients.
    '''
    A1, B1, C1, D1, E1, F1 = coeffs1
    A2, B2, C2, D2, E2, F2 = coeffs2
    
    def resultant_at_x(x):
        p1 = np.array([C1, B1*x + E1, A1*x*x + D1*x + F1], float)
        p2 = np.array([C2, B2*x + E2, A2*x*x + D2*x + F2], float)
        S = np.array([
            [p1[0], p1[1], p1[2],    0.],
            [   0., p1[0], p1[1], p1[2]],
            [p2[0], p2[1], p2[2],    0.],
            [   0., p2[0], p2[1], p2[2]]
        ], float)
        return np.linalg.det(S)

    # 1) Fit the quartic in x via 5 samples
    xs = np.array([-2., -1., 0., 1., 2.])
    vals = np.array([resultant_at_x(x) for x in xs])
    coeffs = np.polyfit(xs, vals, 4)
    x_roots = np.roots(coeffs)

    # 2) Back-substitute each x_i to solve for y
    candidates = []
    for x in x_roots:
        a = C1
        b = B1*x + E1
        c = A1*x*x + D1*x + F1

        if abs(a) > lower_tol:
            disc = b*b - 4*a*c
            # force a complex sqrt so we don't get NaNs
            sqrt_disc = np.sqrt(disc.astype(complex))
            ys = [(-b + sqrt_disc) / (2*a),
                  (-b - sqrt_disc) / (2*a)]
        elif abs(b) > lower_tol:
            ys = [ -c / b ]
        else:
            continue

        for y in ys:
            if abs(A2*x*x + B2*x*y + C2*y*y + D2*x + E2*y + F2) < upper_tol:
                candidates.append((x, y))

    # 3) Deduplicate
    unique = []
    for p in candidates:
        if not any(np.allclose(p, q, atol=lower_tol) for q in unique):
            unique.append(p)

    # 4) Build an (N×2) complex array — even if N=0
    pts = np.array(unique, dtype=complex).reshape(-1, 2)

    # 5) Filter to only those whose imaginary parts are ≈ 0
    mask = np.all(np.isclose(pts.imag, 0.0, atol=upper_tol), axis=1)
    real_pts = pts[mask].real

    return real_pts

def hyperbola_xy_transform(hyperbola_coeffs, tol=1e-14):
    '''
    Find M and t for transformation: X, Y = M @ np.array([x, y]) + t
    '''
    A, B, C, D, E, F = hyperbola_coeffs
    
    # 1.  translate to the center  (kills the linear terms)
    det = 4 * A * C - B * B
    if abs(det) < tol:
        raise ValueError('Degenerate quadratic part (determinant ≈ 0).')

    x0 = (B * E - 2 * C * D) / det
    y0 = (B * D - 2 * A * E) / det
    F_star = A * x0**2 + B * x0 * y0 + C * y0**2 + D * x0 + E * y0 + F
    if abs(F_star) < tol:
        raise ValueError('The conic degenerates into two intersecting lines.')

    # 2.  rotate into principal axes  ——  use eigenvectors of Q
    Q = np.array([[A, B / 2.0],
                  [B / 2.0, C]])

    eigvals, eigvecs = np.linalg.eigh(Q)      # orthonormal eigen-vectors
    # eigvals sorted ascending; exactly one is positive and one negative
    pos_idx, neg_idx = (0, 1) if eigvals[0] > 0 else (1, 0)

    lambda_pos = eigvals[pos_idx]      # positive eigen-value
    lambda_neg = eigvals[neg_idx]      # negative eigen-value

    if lambda_pos * lambda_neg > 0:
        raise ValueError('Not a hyperbola (eigenvalues have same sign).')

    V = eigvecs[:, [pos_idx, neg_idx]]        # columns:  v_pos  v_neg

    # make it a proper rotation (det = +1), not a reflection
    if np.linalg.det(V) < 0:
        V[:, 0] = -V[:, 0]                    # flip one axis

    R_theta = V.T                             # 2×2 rotation matrix
    theta = np.arctan2(R_theta[1, 0], R_theta[0, 0])  # angle of first axis

    # 3.  fixed 45° rotation   (u,v) → (X,Y)
    c45 = np.sqrt(0.5)
    R45 = np.array([[ c45, -c45],
                    [ c45,  c45]])

    # 4.  assemble full affine map
    M = R45 @ R_theta                     # linear part
    t = -M @ np.array([x0, y0])               # translation

    # net rotation undergone by the original x-axis
    theta45 = (theta + np.pi / 4.0) % (2.0 * np.pi)

    return M, t, theta45

def XY_ellipse_to_xy_conic(a, b, xc, yc, x0, y0, theta):
    '''
    Convert ellipse parameters in XY space to conic coefficients in xy space.
    '''
    c, s = np.cos(theta), np.sin(theta)
    A = c**2 / a**2 + s**2 / b**2
    B = 2 * s * c * (1 / b**2 - 1 / a**2)
    C = s**2 / a**2 + c**2 / b**2
    D = 2 * ((x0 - xc) * c / a**2 + (y0 - yc) * s / b**2)
    E = 2 * ((xc - x0) * s / a**2 + (y0 - yc) * c / b**2)
    F = (x0 - xc)**2 / a**2 + (y0 - yc)**2 / b**2 - 1
    return np.array([A, B, C, D, E, F])

def transform_curve(xs, ys, M, t):
    '''
    Transform curve points using the matrix M and translation vector t.
    '''
    coords = np.column_stack((xs, ys))
    coords = M @ coords.T + t[:, np.newaxis]
    return coords[0], coords[1]

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
                             x_min=-2.0, x_max=2.0,
                             tol=1e-20):
    '''
    Generate x, y lower and upper bounds for a conic section
    '''
    A, B, C, D, E, F = coeffs
    xs = np.linspace(x_min, x_max, n)
    
    # Treat equation as quadratic in y:   C y² + (B x + E) y + (A x² + D x + F) = 0
    a = C
    b = B*xs + E
    c = A*xs**2 + D*xs + F

    disc = b**2 - 4.0*a*c                 # discriminant Δ
    mask = disc >= -tol                        # keep only "real" columns
    disc[~mask] = np.nan                       # avoid negative roots
    sqrt_d = np.sqrt(np.clip(disc, 0.0, None))

    y_lower = np.full_like(xs, np.nan)
    y_upper = np.full_like(xs, np.nan)
    denom = 2.0 * a
    y_lower[mask] = (-b[mask] - sqrt_d[mask]) / denom
    y_upper[mask] = (-b[mask] + sqrt_d[mask]) / denom

    return xs, y_lower, xs, y_upper

def generate_linear_linspaces(m, b, n=5000,
                             x_min=-2.0, x_max=2.0):
    '''
    Generate x, y lower and upper bounds for a linear section
    '''
    xs = np.linspace(x_min, x_max, n)
    ys = m * xs + b
    return xs, ys
