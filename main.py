# Prepare scene Gaussians (from EDINA data)
gaussians = load_scene_gaussians(EDINA_frame0)
for g in gaussians:
    analyze_gaussian_shape(g)  # precompute eigen info for SESC

# Baseline vs Shape-aware rendering for each frame
baseline_metrics = {'ops':0, 'mem_access':0}
shape_metrics    = {'ops':0, 'mem_access':0}
baseline_images = []
shape_images = []
cache = GaussianCache()
for frame_idx, frame_data in enumerate(EDINA_frames):
    if frame_idx > 0:
        update_gaussians_positions(gaussians, frame_data)  # egocentric motion
    
    # Render this frame
    baseline_image = np.zeros((H, W))
    shape_image    = np.zeros((H, W))
    for tile in traverse_tiles(order=('row' if baseline else 'z')):
        for g_id, g in enumerate(gaussians_in_tile(tile, gaussians)):
            # Baseline computation
            # (no skip, always full precision)
            contrib = compute_tile_contrib(g, tile.x, tile.y, mode='full')
            baseline_image[tile.region] += contrib
            baseline_metrics['ops'] += 169   # 169 pixels computed
            baseline_metrics['mem_access'] += 1  # assume gauss fetched from memory

            # Shape-aware computation
            if should_skip_tile(g, tile.x, tile.y):
                continue  # skip this Gaussian for this tile
            # Cache access (may hit or miss)
            hit = cache.access(g_id)
            if not hit:
                shape_metrics['mem_access'] += 1
            # Choose high-precision or high-performance mode based on shape
            mode = 'full' if g['min_radius'] < shape_threshold else 'interp'
            contrib = compute_tile_contrib(g, tile.x, tile.y, mode)
            shape_image[tile.region] += contrib
            if mode == 'full':
                shape_metrics['ops'] += 169
            else:
                shape_metrics['ops'] += 49  # RE ops
                shape_metrics['ops'] += 120 # IE ops (counted as simpler int ops)
    # Store rendered images
    baseline_images.append(baseline_image)
    shape_images.append(shape_image)
