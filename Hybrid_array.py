import math
import numpy as np

def compute_tile_contrib(gaussian, tile_x, tile_y, mode='full'):
    """Compute the contribution of one Gaussian to a 13x13 tile.
    mode='full' for high-precision (compute every pixel via RE),
    mode='interp' for high-performance (compute sparse RE and interpolate IE)."""
    cx, cy = gaussian['center']         # Gaussian center (projected 2D coords)
    cov_inv = gaussian['Cov_inv']       # 2x2 inverse covariance matrix
    intensity = gaussian['intensity']   # Gaussian color/intensity coefficient
    # Tile pixel coordinate range
    px0 = tile_x * 13
    py0 = tile_y * 13
    contrib = np.zeros((13, 13), dtype=np.float32)
    
    if mode == 'full':
        # High-precision: compute alpha for all 169 pixels (RE for every pixel)
        for i in range(13):
            py = py0 + i
            dy = py - cy
            for j in range(13):
                px = px0 + j
                dx = px - cx
                # Compute exponent = -0.5 * [dx,dy] * Cov_inv * [dx,dy]^T
                expo = -0.5 * (cov_inv[0,0]*dx*dx + 2*cov_inv[0,1]*dx*dy + cov_inv[1,1]*dy*dy)
                # LUT-based exponential (simulated by math.exp) on FP16 exponent
                alpha = math.exp(expo) if expo > -50 else 0.0   # clamp for underflow
                contrib[i, j] = intensity * alpha
    else:
        # High-performance: compute 49 RE positions and interpolate 120 IE positions
        # 1. Compute RE contributions at even indices (0,2,4,...,12)
        for i in range(0, 13, 2):
            py = py0 + i
            dy = py - cy
            for j in range(0, 13, 2):
                px = px0 + j
                dx = px - cx
                expo = -0.5 * (cov_inv[0,0]*dx*dx + 2*cov_inv[0,1]*dx*dy + cov_inv[1,1]*dy*dy)
                alpha = math.exp(expo) if expo > -50 else 0.0
                contrib[i, j] = intensity * alpha
        # 2. Uni-directional IE: interpolate horizontally (even row, odd col) and vertically (odd row, even col)
        for i in range(0, 13, 2):       # horizontal interpolation for pixels between two REs
            for j in range(1, 13, 2):
                left_val  = contrib[i, j-1]    # RE at (i,j-1)
                right_val = contrib[i, j+1]    # RE at (i,j+1)
                contrib[i, j] = 0.5 * (left_val + right_val)
        for i in range(1, 13, 2):       # vertical interpolation for pixels between two REs
            for j in range(0, 13, 2):
                top_val    = contrib[i-1, j]   # RE at (i-1,j)
                bottom_val = contrib[i+1, j]   # RE at (i+1,j)
                contrib[i, j] = 0.5 * (top_val + bottom_val)
        # 3. Bi-directional IE: bilinear interpolate the remaining odd-row, odd-col pixels from four diagonal REs
        for i in range(1, 13, 2):
            for j in range(1, 13, 2):
                a = contrib[i-1, j-1]   # top-left RE
                b = contrib[i-1, j+1]   # top-right RE
                c = contrib[i+1, j-1]   # bottom-left RE
                d = contrib[i+1, j+1]   # bottom-right RE
                contrib[i, j] = 0.25 * (a + b + c + d)
    return contrib
