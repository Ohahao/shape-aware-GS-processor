import numpy.linalg as LA

def analyze_gaussian_shape(gaussian):
    # Pre-compute eigenvalues (for shape unit use)
    cov = gaussian['Cov']            # 2x2 covariance matrix
    vals, vecs = LA.eigh(cov)        # eigen decomposition (vals sorted low->high)
    gaussian['min_radius'] = math.sqrt(vals[0])   # minimum std. deviation (sqrt of smallest eigenvalue)
    gaussian['max_radius'] = math.sqrt(vals[1])   # maximum std. deviation
    gaussian['e_min_vec'] = vecs[:,0]             # unit eigenvector for min axis (for projections)

def should_skip_tile(gaussian, tile_x, tile_y, alpha_thresh=1e-4):
    """Decide whether to skip this Gaussian's contribution on the given tile."""
    cx, cy = gaussian['center']
    # 1. Corner unit: check Gaussian contribution at tile corners
    px0 = tile_x * 13
    py0 = tile_y * 13
    corners = [(px0, py0), (px0+12, py0), (px0, py0+12), (px0+12, py0+12)]
    corner_alphas = []
    for (px, py) in corners:
        dx, dy = px - cx, py - cy
        expo = -0.5 * (gaussian['Cov_inv'][0,0]*dx*dx + 
                       2*gaussian['Cov_inv'][0,1]*dx*dy + 
                       gaussian['Cov_inv'][1,1]*dy*dy)
        alpha = math.exp(expo) if expo > -50 else 0.0
        corner_alphas.append(alpha * gaussian['intensity'])
    if max(corner_alphas) < alpha_thresh:
        return True   # all corners have negligible alpha
    
    # 2. Position unit: check distance of tile from Gaussian center
    tile_center_x = px0 + 6    # center of 13x13 tile = offset + 6
    tile_center_y = py0 + 6
    dist = math.hypot(tile_center_x - cx, tile_center_y - cy)
    # If tile is farther than 3*max_radius (minus half-diagonal) – no overlap
    tile_half_diag = math.hypot(6, 6)
    if dist - tile_half_diag > 3 * gaussian['max_radius']:
        return True   # tile is well beyond Gaussian reach
    
    # 3. Shape unit: check alignment relative to Gaussian's narrow axis
    # Project tile center distance onto Gaussian's smallest radius direction:
    d_center = np.array([tile_center_x - cx, tile_center_y - cy])
    proj_len = abs(d_center.dot(gaussian['e_min_vec']))
    if proj_len - tile_half_diag > 3 * gaussian['min_radius']:
        return True   # tile lies outside 3σ of the minor axis
    
    # 4. Opacity unit: skip if overall intensity (opacity) is extremely low
    if gaussian['intensity'] < 1e-6:
        return True
    
    return False  # no skipping; tile has significant contribution
