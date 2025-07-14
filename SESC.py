import math
import numpy as np
import numpy.linalg as LA
import torch

def analyze_gaussian_shape(gaussian):
    lam1 = gaussian['lam_min']
    lam2 = gaussian['lam_max'] 
    R1 = gaussian['min_radius']  
    R2 = gaussian['max_radius'] 
    Rmin = min(R1, R2)
    
    Rmin_low  = 1.0 / math.sqrt(lam1 + lam2)
    Rmin_high = 1.0 / ((lam1 * lam2) ** 0.25)
    
    shape_idx = (Rmin - Rmin_low) / (Rmin_high - Rmin_low)
    shape_idx = min(max(shape_idx, 0.0), 1.0)

    if shape_idx < 0.5:
        mode = 'smooth'
    else:
        mode = 'spiky'

    return shape_idx, mode



def should_skip_tile(gaussian, tile_x, tile_y, alpha_thresh=1e-8):
    """Decide whether to skip this Gaussian's contribution on the given tile."""
    cx, cy = gaussian['center2d']
    # 1. Corner unit: check Gaussian contribution at tile corners
    px0 = tile_x * 13
    py0 = tile_y * 13
    corners = [(px0, py0), (px0+12, py0), (px0, py0+12), (px0+12, py0+12)]
    corner_alphas = []
    for (px, py) in corners:
        dx, dy = px - cx, py - cy
        expo = -0.5 * (gaussian['cov_inv'][0,0]*dx*dx + 
                       2*gaussian['cov_inv'][0,1]*dx*dy + 
                       gaussian['cov_inv'][1,1]*dy*dy)
        expo = min(0.0, max(expo, -50.0))
        alpha = math.exp(expo)
        corner_alphas.append(alpha * gaussian['intensity'])
    if max(corner_alphas) < alpha_thresh:    #alpha threshold 값 설정근거: 인지적으로 무시 가능한 기준(ChatGPT)
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
    if gaussian['intensity'] < 1e-8:
        return True
    
    return False  # no skipping; tile has significant contribution
    
    
def should_skip_batch(
    idxs, centers, cov_inv, radii, intensity,
    tile_x, tile_y, R1, R2, device,
    tile_size=13, alpha_thresh=1e-5
):
    """
    idxs:       (M,) 남길 Gaussian 인덱스
    centers:    (N,2) 픽셀 좌표계상의 (u,v)
    cov_inv:    (N,2,2) 픽셀 공분산의 역행렬
    intensity:  (N,)
    R1, R2:     (N,) shape 관련 스칼라
    returns:    skip mask (M,) -> True 이면 스킵
    """
    with torch.no_grad():
        # 1) 선택된 항목만 뽑아서 M 배치로
        c     = centers[idxs]      # (M,2)
        inv   = cov_inv[idxs]      # (M,2,2)
        inten = intensity[idxs]    # (M,)
        r1    = R1[idxs]           # (M,)
        r2    = R2[idxs]           # (M,)
        radii = radii[idxs]


        # 2) 타일 경계 & 상수
        px0 = tile_x * tile_size
        py0 = tile_y * tile_size
        corners = torch.tensor(
            [[px0, py0], [px0+tile_size-1, py0],
             [px0, py0+tile_size-1], [px0+tile_size-1, py0+tile_size-1]],
            device=device, dtype=c.dtype
        )  # (4,2)

        tile_center = torch.tensor(
            [px0 + tile_size/2, py0 + tile_size/2],
            device=device, dtype=c.dtype
        )  # (2,)

        # --- 1) corners α 검사 ---
        # (M,4,2)
        diffs = corners.unsqueeze(0) - c.unsqueeze(1)
        dx, dy = diffs[...,0], diffs[...,1]              # (M,4)
        ix0 = inv[:,0,0]  # inv[0,0]
        ix1 = inv[:,0,1]  # inv[0,1]
        ix2 = inv[:,1,1]  # inv[1,1]
        power = -0.5*(ix0.unsqueeze(1)*dx*dx
                      + 2*ix1.unsqueeze(1)*dx*dy
                      + ix2.unsqueeze(1)*dy*dy)
        alpha_c = (inten.unsqueeze(1) * torch.exp(power)).clamp_max(0.99)
        alpha_c_max = alpha_c.max(dim=1).values            # (M,)
        mask1 = alpha_c_max < 1e-8

        # --- 2) 중심 거리 검사 ---
        x0, x1 = tile_x*tile_size, (tile_x+1)*tile_size
        y0, y1 = tile_y*tile_size, (tile_y+1)*tile_size
        
        cx, cy, r = c[:,0], c[:,1], radii
        mask2 = (
            (cx + r < x0) |
            (cx - r >= x1) |
            (cy + r < y0) |
            (cy - r >= y1)
        )


        # --- 3) shape ratio 검사(기준 정하기 애매해서 일단 패스)---
        # trace = inv[0,0] + inv[1,1]
        #T_sh = 0.2
        #mask3 = (ix0 + ix2) > T_sh

        # --- 4) intensity 검사 ---
        mask4 = inten < alpha_thresh
        
        #mask5 = radii > 512

        # 합쳐서 skip mask
        skip = mask1 | mask2 | mask4   # (M,)

    return skip
    
    

def should_skip_batch_chunk(
    idxs, centers, cov_inv, intensity,
    tile_x, tile_y, R1, R2, device, tile_size=13, alpha_thresh=1e-2,
    chunk_size=65536
):

  with torch.no_grad():
      N = idxs.numel()  #number of gaussians
      skip = torch.zeros(N, dtype=torch.bool, device=device)  
  
      # precompute tile data
      px0 = tile_x * tile_size
      py0 = tile_y * tile_size
      corners = torch.tensor(
        [[px0, py0], [px0+tile_size-1, py0],
         [px0, py0+tile_size-1], [px0+tile_size-1, py0+tile_size-1]],
        device=device, dtype=centers.dtype
      )  # (4,2)
      tile_center = torch.tensor([px0+tile_size//2, py0+tile_size//2],
                                 device=device, dtype=centers.dtype)
      tile_half_diag = math.hypot(tile_size/2, tile_size/2)
  
      # --- 디버깅 코드 시작 ---
      print_debug_info = True # True로 설정하면 첫 번째 청크의 첫 번째 가우시안에 대한 정보 출력
      # --- 디버깅 코드 끝 ---
      
      for i in range(0, N, chunk_size):
          j = min(N, i + chunk_size)

          c = centers[idxs][i:j]       # (M,2)
          inv = cov_inv[idxs][i:j]     # (M,2,2)
          inten = intensity[idxs][i:j] # (M,)
          #emin = e_min_vec[i:j]  # (M,2)
          R1_ch = R1[idxs][i:j]        # (M,)
          R2_ch = R2[idxs][i:j]        # (M,)
          M = c.shape[0]
          
          
          # 1) corners test → mask1 (M,)
          #print(f"corners: {corners}, centers: {c}")
          diffs = corners.unsqueeze(0) - c.unsqueeze(1)      # (M,4,2)
          dx, dy = diffs[...,0], diffs[...,1]    #(M,4)
          #print(f"dx: {dx}, dy: {dy}")
          c_x = inv[:,0,0]   # (M,)
          c_y = inv[:,0,1]
          c_z = inv[:,1,1]
          power = -0.5*(c_x.unsqueeze(1)*dx*dx + c_z.unsqueeze(1)*dy*dy) - c_y.unsqueeze(1)*dx*dy  #(M,4)
          #power = power.clamp(min=-20.0, max=0.0) 
  
          alpha_c = (inten.unsqueeze(1) * torch.exp(power)).clamp_max(0.99) 
          alpha_c_max = alpha_c.max(dim=1).values  #4개의 corner alpha 중 max value 추출 
          #print(f"alpha_c: {alpha_c}")
            
          alpha_mean = alpha_c_max.mean()   # 평균값 (스칼라 텐서)
          zero_count = (alpha_c_max == 0).sum().item()
          small_count = (alpha_c_max < 1e-7).sum().item()

          mask1 = alpha_c_max < 1e-7 #(M,)
   
  
  
          # 2) dist test → mask2
          #pixels = torch.zeros_like(centers)
          #pixels = torch.stack([px0:px0+tile_size-1, py0+py0+tile_size-1], dim=1)
          #mask2 = centers[idxs] - pixels[idxs] > 
          dist = torch.hypot(*(tile_center - c).unbind(-1))  # (M,)
          mask2 = dist - tile_half_diag > 3 * R2_ch
  
          
          # 3) shape test → mask3
          #dctr = tile_center - c                           # (M,2)
          #proj_len = (dctr * emin).abs().sum(dim=1)        # (M,)
          #mask3 = proj_len - tile_half_diag > 3 * R1_ch
          T_sh = 0.2
          mask3 = (c_x + c_z) > T_sh
  
          # 4) intensity test → mask4
          mask4 = inten < alpha_thresh                     # (M,)
  
          skip[i:j] = mask1 | mask2 | mask3 | mask4  
          
      num_skipped = skip.sum().item()
      #print(f"Number of skipped Gaussians: {num_skipped} / {N}")
  return skip