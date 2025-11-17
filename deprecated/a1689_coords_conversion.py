import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Fixed reference position (centre of the tie-set) --------------------------
_RA0  = 197.8770137254902    # deg
_DEC0 =  -1.3410323529411765 # deg

# Affine coefficients:  x = a₁ Δα + a₂ Δδ + x₀   (Δα, Δδ in arcsec)
#                       y = b₁ Δα + b₂ Δδ + y₀
_A1, _A2, _X0 =  8.49065623,  18.10783399, 2165.61764706
_B1, _B2, _Y0 = 18.10876590,  -8.49118067, 2412.73529412

_M     = np.array([[_A1, _A2],
                   [_B1, _B2]])
_INV_M = np.linalg.inv(_M)   # pre-compute inverse once

# ---------------------------------------------------------------------------
# Angle-parsing helpers
# ---------------------------------------------------------------------------

def _parse_ra_dec(ra, dec):
    '''
    Convert mixed (string or numeric) RA/Dec inputs to decimal degrees.

    Parameters
    ----------
    ra, dec : str or float
        Examples of accepted strings:
            RA  : '13h11m29.45s'
            Dec : '-01d20m28.1s'
        Numeric inputs are interpreted as *degrees* for Dec and *hours* for RA.

    Returns
    -------
    (ra_deg, dec_deg) : tuple of float
        RA and Dec in decimal degrees.
    '''
    ra_deg  = _ra_to_deg_scalar(ra)
    dec_deg = _dec_to_deg_scalar(dec)
    return ra_deg, dec_deg


def _ra_to_deg_scalar(ra):
    '''Helper: scalar RA (string or float) → decimal degrees.'''
    if isinstance(ra, str):
        h, m, s = map(float, ra.replace('h', ' ').replace('m', ' ')
                               .replace('s', ' ').split())
        return 15.0 * (h + m / 60.0 + s / 3600.0)
    return float(ra) * 15.0           # numeric assumed in *hours*


def _dec_to_deg_scalar(dec):
    '''Helper: scalar Dec (string or float) → decimal degrees.'''
    if isinstance(dec, str):
        sign = -1.0 if dec.strip().startswith('-') else 1.0
        d, m, s = map(float, dec.replace('+', '').replace('-', '')
                                .replace('d', ' ').replace('m', ' ')
                                .replace('s', ' ').split())
        return sign * (d + m / 60.0 + s / 3600.0)
    return float(dec)

def _deg_to_ra_scalar(deg):
    '''Helper: scalar decimal degrees → RA (string in 'hh mm ss.ss' format).'''
    deg = float(deg)
    if deg < 0.0:
        deg += 360.0
    h = int(deg // 15.0)
    m = int((deg % 15.0) * 4.0)
    s = (deg % 15.0 - m / 4.0) * 240.0
    return f'{h:02d}h{m:02d}m{s:.4f}s'

def _deg_to_dec_scalar(deg):
    '''Helper: scalar decimal degrees → Dec (string in '±dd mm ss.ss' format).'''
    deg = float(deg)
    sign = '-' if deg < 0.0 else '+'
    deg = abs(deg)
    d = int(deg)
    m = int((deg - d) * 60.0)
    s = (deg - d - m / 60.0) * 3600.0
    return f'{sign}{d:02d}d{m:02d}m{s:.4f}s'

# Public wrappers -----------------------------------------------------------

def ra_to_deg(ra):
    '''Vectorised RA → degrees wrapper.'''
    ra_arr = np.atleast_1d(ra)
    out    = np.array([_ra_to_deg_scalar(r) for r in ra_arr])
    return out if out.ndim else out.item()


def dec_to_deg(dec):
    '''Vectorised Dec → degrees wrapper.'''
    dec_arr = np.atleast_1d(dec)
    out     = np.array([_dec_to_deg_scalar(d) for d in dec_arr])
    return out if out.ndim else out.item()

def deg_to_ra(ra_deg):
    '''Vectorised degrees → RA wrapper.'''
    ra_deg_arr = np.atleast_1d(ra_deg)
    out        = np.array([_deg_to_ra_scalar(rd) for rd in ra_deg_arr])
    return out if out.ndim else out.item()

def deg_to_dec(dec_deg):
    '''Vectorised degrees → Dec wrapper.'''
    dec_deg_arr = np.atleast_1d(dec_deg)
    out          = np.array([_deg_to_dec_scalar(dd) for dd in dec_deg_arr])
    return out if out.ndim else out.item()

# ---------------------------------------------------------------------------
# Coordinate conversions
# ---------------------------------------------------------------------------

def radec_to_xy(ra, dec):
    '''
    Convert (RA, Dec) to detector pixels (x, y).

    Parameters
    ----------
    ra, dec : array-like or scalar
        RA may be string(s) in 'hh mm ss' format or numeric hours.
        Dec may be string(s) in '±dd mm ss' format or numeric degrees.

    Returns
    -------
    (x, y) : ndarray or float
        Pixel coordinates in the original ACS frame (not recentred).
    '''
    ra_arr, dec_arr = np.atleast_1d(ra), np.atleast_1d(dec)
    ra_deg, dec_deg = [], []

    for r, d in zip(ra_arr, dec_arr):
        rd, dd = _parse_ra_dec(r, d)
        ra_deg.append(rd)
        dec_deg.append(dd)

    ra_deg  = np.asarray(ra_deg)
    dec_deg = np.asarray(dec_deg)

    # Offsets (small-angle approximation) in arcseconds
    cos_dec0 = np.cos(np.radians(_DEC0))
    dra  = (ra_deg  - _RA0) * cos_dec0 * 3600.0  # eastward
    ddec = (dec_deg - _DEC0) * 3600.0           # northward

    # Affine mapping to pixels
    x = _A1 * dra + _A2 * ddec + _X0
    y = _B1 * dra + _B2 * ddec + _Y0
    return x.squeeze(), y.squeeze()


def xy_to_radec(x, y):
    '''
    Convert detector pixels (x, y) back to (RA, Dec).

    Returns
    -------
    (ra_deg, dec_deg) : float or ndarray
        Positions in decimal degrees.
    '''
    x = np.atleast_1d(x).astype(float)
    y = np.atleast_1d(y).astype(float)

    # Pixel ➜ arcsec offsets
    dxy         = np.vstack((x - _X0, y - _Y0))       # shape (2, N)
    dra, ddec   = _INV_M @ dxy                        # east, north (arcsec)

    # Tangent-plane inversion
    dec_deg     = _DEC0 + ddec / 3600.0
    cos_dec     = np.cos(np.radians(dec_deg))
    ra_deg      = _RA0 + dra / (3600.0 * cos_dec)

    return ra_deg.squeeze(), dec_deg.squeeze()

# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # ---------------------------------------------------------------------
    # Example 1: verify RA/Dec string ⇄ degree conversion
    # ---------------------------------------------------------------------
    actual_ra  = '13h11m29.45s'
    actual_dec = '-01d20m28.1s'
    print('Actual RA  = {:.7f} deg'.format(ra_to_deg(actual_ra)))
    print('Actual Dec = {:.7f} deg'.format(dec_to_deg(actual_dec)))

    # ---------------------------------------------------------------------
    # Example 2: SZE position from literature → pixels (re-centred)
    # ---------------------------------------------------------------------
    ra_ex  = '13h11m29.57s'
    dec_ex = '-01d20m29.87s'
    x, y   = radec_to_xy(ra_ex, dec_ex)
    x -= 2048.0
    y -= 2048.0
    print(f'Pixel coordinates (recentred): [{x}, {y}]')
