import math
import numpy as np
import torch
import torchvision.utils as vutils
from typing import NamedTuple
import os # For file operations, though simple open might suffice



class Tile(NamedTuple):
    x0: int
    x1: int
    y0: int
    y1: int


def create_exp_lut(z_min: float, z_max: float, num_segments: int, 
                   device: torch.device, dtype=torch.float32):

    # 1. Define segment boundaries
    edges = torch.linspace(z_min, z_max, num_segments + 1,
                           device=device, dtype=dtype)
    
    # 2. Compute true exp(z) at boundaries
    f_vals = torch.exp(edges)
      
    
    # 3. Compute slopes (a_k) and intercepts (b_k)
    a = (f_vals[1:] - f_vals[:-1]) / (edges[1:] - edges[:-1])
    b = f_vals[:-1] - a * edges[:-1]
    
    return a, b



def compute_alpha(
    centers:   torch.Tensor,   # (K,2)
    cov_inv:   torch.Tensor,   # (K,2,2)
    intensity: torch.Tensor,   # (K,)
    colors:    torch.Tensor,   # (K,3)
    N_total:   int,            # 전체 gaussian 개수
    N:         int,            # 타일별 gaussian 개수 
    tile_w:    int,            # tile width
    tile_h:    int,            # tile height
    tile_x0:   int,            # 타일 시작 x 좌표
    tile_y0:   int,            # 타일 시작 y 좌표
    N_tiles:   int,            # number of tiles
    sp:        torch.Tensor,   # indices of spiky gaussians, (sp_N,)
    sm:        torch.Tensor,   # indices of smooth gaussians, (sm_N,)
    xs:        torch.Tensor,   # (tile_w,)
    ys:        torch.Tensor,   # (tile_h,)
    device:    torch.device    # Explicitly require device
) -> torch.Tensor:

    NUM_SEGMENTS_HW = 40 
    Z_MIN_FIXED = -15.0
    Z_MAX_FIXED = 0.0

    a_lut, b_lut = create_exp_lut(Z_MIN_FIXED, Z_MAX_FIXED, NUM_SEGMENTS_HW, device=device)

    def _process(idxs: torch.Tensor, P_x: torch.Tensor, P_y: torch.Tensor):
        N_gauss = idxs.numel()
        N_pix   = P_x.numel() 

        if N_gauss == 0:
            return torch.empty((0, N_pix), device=device, dtype=torch.float32) # Return empty alpha
        
        # Means
        M = centers[idxs].to(device)           # (N_gauss,2)
        M_x = M[:,0].unsqueeze(1)               # (N_gauss,1)
        M_y = M[:,1].unsqueeze(1)               # (N_gauss,1)

        # distances
        dx = M_x - P_x.unsqueeze(0)             # (N_gauss, N_pix)
        dy = M_y - P_y.unsqueeze(0)             # (N_gauss, N_pix)
        #print(f"\n\n distances in _process: {dx[0]}, {dy[0]}")
        #print(f"\n\n pixel coords in _process: {P_x[0]}, {P_y[0]}")

        # Mahalanobis
        cov = cov_inv[idxs].to(device)          # (N_gauss,2,2)
        cx = cov[:,0,0].unsqueeze(1)
        cy = cov[:,0,1].unsqueeze(1)
        cz = cov[:,1,1].unsqueeze(1)
        #print(f"inverse 2d cov values in tile: {cov[0]}")
        power = -0.5*(cx*dx*dx + cz*dy*dy) - cy*dx*dy
        
        # power 값을 고정된 범위로 clamp
        clamped_power = torch.clamp(power, min=Z_MIN_FIXED, max=Z_MAX_FIXED)
        '''
        flat = clamped_power.flatten()

        if flat.numel() > 0:
            scale = (flat - Z_MIN_FIXED) / (Z_MAX_FIXED - Z_MIN_FIXED)
            # 구간 수에 맞게 인덱스 계산
            idx_lut = (scale * (NUM_SEGMENTS_HW - 1)).round().long().clamp(0, NUM_SEGMENTS_HW - 1)
            
            # 미리 생성된 LUT 사용
            expo = (a_lut[idx_lut] * flat + b_lut[idx_lut]).view(power.shape)
            
        
        #power = torch.clamp(power, max=0.0)
        
        # exponential lookup
        flat = power.flatten() # Shape (N_gauss * N_pix,)

        if flat.numel() == 0:
            print(f"Warning: flat.numel() is 0 in _process. N_gauss={N_gauss}, N_pix={N_pix}, power.shape={power.shape}")
            expo = torch.zeros_like(power) 
        else:
            z_min = float(flat.min())
            z_max = float(flat.max())

            if z_min == z_max:
                expo = torch.exp(power)
            else:
                scale = (flat - z_min) / (z_max - z_min)
                idx_lut = (scale*16).floor().long().clamp(0,15)
                expo   = (a_lut[idx_lut]*flat + b_lut[idx_lut]).view(power.shape)
                 
        '''       
        # alpha
        I_g = intensity[idxs].to(device).unsqueeze(1)  # (N_gauss,1)

        #exponential 함수 사용
        expo = torch.exp(clamped_power)

        thresh = torch.full_like(expo, 0.99, device=device)
        alpha  = torch.min(thresh, I_g * expo)         # (N_gauss, N_pix)
   
        #print(f"intensity: {I_g}, before unsqueeze: {intensity[idxs]}")
        '''
        print(f"[DEBUG] power stats: min={power.min():.4f}, max={power.max():.4f}, mean={power.mean():.4f}, std={power.std():.4f}")
        print(f"[DEBUG] dx.mean={dx.mean():.2f}, dy.mean={dy.mean():.2f}")
        print(f"[DEBUG] cx range: {cx.min().item():.2e} ~ {cx.max().item():.2e}")
        print(f"[DEBUG] cz range: {cz.min().item():.2e} ~ {cz.max().item():.2e}")
        print(f"[DEBUG] expo mean: {expo.mean():.2e}")
        print(f"[DEBUG] intensity mean: {I_g.mean().item():.4f}")
        print(f"[DEBUG] alpha mean: {alpha.mean().item():.6f}")
        '''
        
        return alpha 

    #entire alpha 
    alpha = torch.zeros(N_total, tile_w*tile_h).to(device)  #(N, 169)

    # --- spiky: full grid --- Px_full, Py_full 생성은 xs, ys (이미 w,h 반영) 기반이므로 그대로 둠
    if sp.numel() > 0:
        Px_full_mesh, Py_full_mesh = torch.meshgrid(xs, ys, indexing="xy")
        Px_full_flat = Px_full_mesh.flatten().to(device)
        Py_full_flat = Py_full_mesh.flatten().to(device)
        alpha_sp_processed = _process(sp, Px_full_flat, Py_full_flat)
        alpha[sp] = alpha_sp_processed
        #print(f"  \n\nspiky alpha: {alpha}")

    # --- smooth: even×even sub-grid --- xs_e, ys_e 생성은 xs, ys (이미 w,h 반영) 기반이므로 그대로 둠
        if sm.numel() > 0:
            xs_e = xs[0::2].to(device)
            ys_e = ys[0::2].to(device)
            
            Px_e_mesh, Py_e_mesh = torch.meshgrid(xs_e, ys_e, indexing="xy")
            Px_e_flat = Px_e_mesh.flatten().to(device)
            Py_e_flat = Py_e_mesh.flatten().to(device)
            
            alpha_sm_processed = _process(sm, Px_e_flat, Py_e_flat)

            even_w_indices = torch.arange(0, tile_w, 2, device=device)
            even_h_indices = torch.arange(0, tile_h, 2, device=device)

            subgrid_local_rows, subgrid_local_cols = torch.meshgrid(even_h_indices, even_w_indices, indexing='ij')
            
            target_indices_in_full_grid_flat = (subgrid_local_rows.flatten().long() * tile_w + 
                                                subgrid_local_cols.flatten().long()).to(device)


            if alpha_sm_processed.shape[1] == target_indices_in_full_grid_flat.numel():
                for i_sm_local, original_idx_sm in enumerate(sm):
                    alpha[original_idx_sm, target_indices_in_full_grid_flat] = alpha_sm_processed[i_sm_local, :]
        #else:
        #    print("NOTICE! there's no smooth gaussians")

    return alpha



def compute_tile_contrib_batch(
    centers:    torch.Tensor,  # (K, 2)
    cov2d:      torch.Tensor,  # (K, 2, 2)
    intensity:  torch.Tensor,  # (K,)
    color:      torch.Tensor,  # (K, 3, 1, 1)
    cov_inv:    torch.Tensor,
    N_total:    int,           # 전체 gaussian 개수 
    N:          int,           # tile 별 gaussian 개수 
    idxs:       torch.Tensor,  # tile 별 gaussian id
    sp:         torch.Tensor,
    sm:         torch.Tensor,
    tile:       Tile,
    N_tiles:    int,
    device:     torch.device = torch.device('cuda:0')
):

        
    # Calculate tile size (fixed as 13)
    w = tile.x1 - tile.x0
    h = tile.y1 - tile.y0

    
    if N == 0:
        print("skip computation cause there's no gaussian")
        return torch.zeros((13,13), device=device, dtype=torch.float32)
  
    centers = centers.to(device)  #(N,2)
    cov_inv = cov_inv.to(device)  
    cov2d   = cov2d.to(device)    #(N,2,2)
    colors  = color.to(device)    #(N,3)
    intensity = intensity.to(device)
    
    #print("\nDBG: sorted_ids:", idxs[:5].tolist())
    #print("DBG: color[0..4]:", color[:5].tolist())
    #print("DBG: intensity[0..4]:", intensity[:5].tolist())
    
    sp_N = sp.numel()  # number of spiky gaussian 
    sm_N = sm.numel()  # number of smooth gaussian 
    idx_N = idxs.numel()  # number of gaussian 
  
    #print(f"\n\n number of spiky gaussians: {sp_N} / number of smooth gaussians: {sm_N} / number of entire gaussians: {idx_N}")
    # set coordinate in each Tile
    px0 = tile.x0  #start point
    py0 = tile.y0
    xs = px0 + torch.arange(w, device=device, dtype=torch.float32)  # (13,) 
    ys = py0 + torch.arange(h, device=device, dtype=torch.float32)  # (13,)
    
    
    # alpha (N, tile*tile)
    alpha = compute_alpha(centers, cov_inv, intensity, colors, N_total, N, w, h, px0, py0, N_tiles, sp, sm, xs, ys, device)

    # --- Interpolating Element ---
    if sm_N > 0: # Only perform interpolation if there are smooth Gaussians
        even_h = torch.arange(0, h, 2, device=device)
        odd_h  = torch.arange(1, h, 2, device=device)
        even_w = torch.arange(0, w, 2, device=device)
        odd_w  = torch.arange(1, w, 2, device=device)

        out = alpha[sm].view(sm_N, h, w)   # (N_sm, 13, 13)

        # 1) 짝수 행(even_h), 홀수 열(odd_w): left/right 이방향 보간
        ew = odd_w[(odd_w > 0) & (odd_w < w - 1)]
        EH, EW = torch.meshgrid(even_h, ew, indexing='ij')  # both (len(even_h), len(ew))
        left  = out[:, EH, EW-1]                         # shape (N, Eh, Ew, 3)
        right = out[:, EH, EW+1]
        out[:, EH, EW] = 0.5 * (left + right)
            
        # 2) 홀수 행(odd_h), 짝수 열(even_w): up/down 이방향 보간
        oh = odd_h[(odd_h > 0) & (odd_h < h-1)]
        EH2, EW2 = torch.meshgrid(oh, even_w, indexing='ij') # shape (len(oh), len(even_w))
        up   = out[:, EH2-1, EW2]
        down = out[:, EH2+1, EW2]
        out[:, EH2, EW2] = 0.5 * (up + down)
        
        # 3) 홀수 행(odd_h), 홀수 열(odd_w): 대각선 교차 보간
        oh2 = odd_h[(odd_h > 0) & (odd_h < h-1)]
        ow2 = odd_w[(odd_w > 0) & (odd_w < w-1)]
        EH3, EW3 = torch.meshgrid(oh2, ow2, indexing='ij')
        ul = out[:, EH3-1, EW3-1]   # up-left
        ur = out[:, EH3-1, EW3+1]   # up-right
        dl = out[:, EH3+1, EW3-1]   # down-left
        dr = out[:, EH3+1, EW3+1]   # down-right
        out[:, EH3, EW3] = 0.25 * (ul + ur + dl + dr)  #out: (N_sm, tile, tile)
        alpha[sm] = out.reshape(sm_N, -1)
        

    # blending
    C      = torch.zeros((3, h*w), device=device)
    test_T = torch.ones(h*w, device=device)
    T      = torch.ones(h*w, device=device)
    
    for i in idxs.tolist():
        alpha_i = alpha[i]
        if (alpha_i.max() < 1.0/255.0):
          continue
        test_T = T * (1-alpha[i])
        if test_T.max() < 1e-4:
          break       
        #compute color
        C += colors[i].view(3,1) * (alpha_i * T).view(1, h*w)  #(N,3) * (N,h*w)
        T = test_T
    final_C = C.reshape(3, h, w)
    '''
    centers_tile = centers[idxs]
    colors_tile = colors[idxs]
    
    # for 루프를 통해 깊이 순으로 정렬된 가우시안을 순서대로 블렌딩
    for i in range(centers_tile.shape[0]):
        alpha_i = alpha[i] # 이미 정렬된 alpha

        # 조기 종료 조건
        if alpha_i.max() < 1.0 / 255.0:
            continue

        T_new = T * (1.0 - alpha_i)
        if T.max() < 1e-4:
            break

        # 색상 기여도 누적
        color_i = colors_tile[i] # 이미 정렬된 color
        C += color_i.view(3, 1) * (alpha_i * T).view(1, h * w)

        # 투과율 업데이트
        T = T_new
    C.reshape(3, h, w)   
    '''
    return final_C