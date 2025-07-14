import numpy as np
import torch
import pandas as pd
import sys
#import matplotlib.pyplot as plt
import torch.nn.functional as F
from typing import Tuple
from gaussian_splatting.utils.sh_utils import eval_sh
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.scene.dataset_readers import get_center
from gaussian_splatting.scene.cameras import Camera



def quaternion_to_rotation_matrix(q):
    r, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    #norm = torch.sqrt(r**2 + x**2 + y**2 + z**2 + 1e-8)
    #r, x, y, z = r/norm, x/norm, y/norm, z/norm

    R = torch.zeros((q.shape[0], 3, 3), device=q.device)
    R[:, 0, 0] = 1 - 2 * (y**2 + z**2)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x**2 + z**2)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x**2 + y**2)
    
    return R
    

def compute_cov3d(scale, modifier, quat):
    #print(f"\n\n quarternion: {quat}")
    S = torch.zeros((scale.shape[0], 3, 3), device=scale.device)
    S[:, 0, 0] = modifier * scale[:, 0]
    S[:, 1, 1] = modifier * scale[:, 1]
    S[:, 2, 2] = modifier * scale[:, 2]

    R = quaternion_to_rotation_matrix(quat)
    M = torch.bmm(R, S)
    Sigma = torch.bmm(M, M.transpose(1, 2))
    cov3D = torch.stack([
        Sigma[:, 0, 0], Sigma[:, 0, 1], Sigma[:, 0, 2],
        Sigma[:, 1, 1], Sigma[:, 1, 2], Sigma[:, 2, 2]
    ], dim=1)
    return cov3D



def in_frustum(orig_points: torch.Tensor,       # (N, 3) or flattened (N*3,)
               viewmatrix: torch.Tensor,        # (3, 4) or (4, 4)
               projmatrix: torch.Tensor,        # (4, 4)
               prefiltered: bool = False) -> (bool, torch.Tensor):


    N = orig_points.shape[0]

    # 1. Homogeneous coordinates: (N, 4)
    ones = torch.ones((N, 1), device=orig_points.device)
    p_orig_h = torch.cat([orig_points, ones], dim=-1)  # (N, 4)

    # 2. Projection → homogeneous screen space
    p_hom = p_orig_h @ projmatrix.T  # (N, 4)
    p_w = 1.0 / (p_hom[:, 3] + 1e-7)
    p_proj = p_hom[:, :3] * p_w.unsqueeze(1)  # (N, 3) — Not used in check, but available
    


    # 3. View transform → camera space
    if viewmatrix.shape == (3, 4):
        R = viewmatrix[:, :3]   # (3, 3)
        t = viewmatrix[:, 3]    # (3,)
        p_view = orig_points @ R.T + t  # (N, 3)
    else:  # (4, 4)
        p_view_h = p_orig_h @ viewmatrix  # (N, 4)
        p_view = p_view_h[:, :3] / (p_view_h[:, 3:] + 1e-7)  # (N, 3)

    # 4. Frustum condition: z > 0.2
    frustum_mask = p_view[:, 2] > 0.2  # (N,)

    if prefiltered and not torch.all(frustum_mask):
        raise RuntimeError("Some points failed frustum check even though prefiltered=True.")

    return frustum_mask, p_view



def ndc2Pix(p_proj: torch.Tensor, W: int, H: int) -> torch.Tensor:
    """
    p_proj: (N, 3) tensor in NDC space (-1 ~ 1), from projection step
    W: image width
    H: image height
    Returns:
        point_image: (N, 2) tensor of pixel coordinates (x, y)
    """
    x_ndc = p_proj[:, 0]
    y_ndc = p_proj[:, 1]

    x_pix = ((x_ndc + 1) * W - 1) * 0.5
    y_pix = ((y_ndc + 1) * H - 1) * 0.5

    return torch.stack([x_pix, y_pix], dim=1)  # (N, 2)


# in preprocessing.py
# project_3d_to_2d 함수를 아래 코드로 교체하세요.
def project_3d_to_2d(cov3d, cam_space_points, camera, intrinsics, device, global_ids):
        
    # 디버깅할 가우시안 ID 설정
    #TARGET_GAUSSIAN_IDX = 15
    #is_target_in_batch = cov3d.shape[0] > TARGET_GAUSSIAN_IDX


    viewmatrix = camera.world_view_transform.to(device)
    fx = intrinsics['fx']
    fy = intrinsics['fy']

    t = cam_space_points

    
    
    t_x, t_y, t_z = t[:, 0], t[:, 1], t[:, 2]

    J = torch.zeros((t.shape[0], 3, 3), device=device, dtype=t.dtype)
    J[:, 0, 0] = fx / t_z
    J[:, 0, 2] = -fx * t_x / (t_z * t_z)
    J[:, 1, 1] = fy / t_z
    J[:, 1, 2] = -fy * t_y / (t_z * t_z)

    Vrk = torch.zeros((cov3d.shape[0], 3, 3), device=device, dtype=cov3d.dtype)
    Vrk[:, 0, 0] = cov3d[:, 0]
    Vrk[:, 0, 1] = cov3d[:, 1]; Vrk[:, 1, 0] = cov3d[:, 1]
    Vrk[:, 0, 2] = cov3d[:, 2]; Vrk[:, 2, 0] = cov3d[:, 2]
    Vrk[:, 1, 1] = cov3d[:, 3]
    Vrk[:, 1, 2] = cov3d[:, 4]; Vrk[:, 2, 1] = cov3d[:, 4]
    Vrk[:, 2, 2] = cov3d[:, 5]

    W = viewmatrix[:3, :3]

    #cov_cam = W @ Vrk @ W.transpose(-2, -1) 
    #cov2d = J @ cov_cam @ J.transpose(-2, -1)
    T = W @ J 
    cov2d = T.transpose(1, 2) @ Vrk @ T 
    #T = J @ W
    #cov2d = T @ Vrk @ T.transpose(1, 2)

    cov2d = cov2d[:, :2, :2]
    cov2d[:, 0, 0] += 0.3
    cov2d[:, 1, 1] += 0.3

    return cov2d
    
    


def preprocess_from_model(model: GaussianModel,
                          intrinsics: dict,
                          view,      
                          device,                    
                          cam_pose: np.ndarray = None,
                          debug_gaussian_global_id_preprocess: int = 440 # Default to 440 for easy use
                          ):
    """
    GaussianModel 인스턴스에서 직접 가져온 파라미터로 전처리.
    Returns: list of dict with keys:
      - id       : int (global ID)
      - center   : (2,) pixel coords
      - cov2d    : (2,2) projected covariance
      - Cov      : same as cov2d
      - cov_inv  : inverse of cov2d
      - intensity: float (opacity)
      - color    : (3,) RGB
      - min_radius, max_radius: floats for SESC & mode decision
      - e_min_vec, lam_min, lam_max: shape analysis info
      
    """
 
    # 1) 필요한 값 할당 
    with torch.no_grad():  
      xyz3D_t    = model.get_xyz.to(device)                       # (N,3) Tensor
      cov3D_t    = model.get_covariance().to(device)              # (N,3,3) Tensor
      sh_all_t   = model.get_features.to(device)                  # (N,B,3) Tensor
      opacity_t  = model.get_opacity.squeeze().to(device)         # (N,) Tensor
      world_view = view.world_view_transform.to(device)
      projmatrix = view.full_proj_transform.to(device)            # view와 projection matrix를 미리 곱한 transform(cameras.py에 정의)
      cam_center = view.camera_center.to(device)
      scales_orig_t = model.get_scaling.to(device)
      rotations = model.get_rotation.to(device)
      modifier = 1.0
      
      original_indices = torch.arange(xyz3D_t.shape[0], device=device)

      # compute cov3D
      rotations_normalized_t = F.normalize(rotations)
      cov3D_t = compute_cov3d(scales_orig_t, modifier, rotations_normalized_t)
      
      # 2) frustum culling 
      valid, p_view = in_frustum(xyz3D_t, world_view, projmatrix, prefiltered=False)
      
      depths_t = p_view[:, 2]
      total = valid.shape[0]
      dropped = int((~valid).sum().item())
      if dropped:
          print(f"[preprocess] Dropped {dropped}/{total} gaussians with z ≤ {0.2}")
      
      # mask 적용
      xyz3D_t   = xyz3D_t[valid]
      cov3D_t   = cov3D_t[valid]
      sh_all_t  = sh_all_t[valid]
      opacity_t = opacity_t[valid]
      depths_t  = depths_t[valid]
      p_view_t  = p_view[valid]
      N = xyz3D_t.shape[0]
      valid_original_ids = original_indices[valid]
      print(f"[preprocess] Remaining gaussians: {N}")
  
      
      # 3) SH-based Gaussian color computation  
      shs_view = model.get_features.transpose(1, 2).view(-1, 3, (model.max_sh_degree+1)**2)
      shs_view = shs_view.to(device)
      shs_view  = shs_view[valid]
      dir_pp = xyz3D_t - cam_center.repeat(sh_all_t.shape[0],1)                   
      dir_pp_norm = dir_pp / dir_pp.norm(dim=1,keepdim=True) 
      dir_pp_norm = dir_pp_norm.to(device)
      sh2rgb_t = eval_sh(model.active_sh_degree, shs_view, dir_pp_norm)  # (N,3) Tensor
      colors_precomp_t = torch.clamp_min(sh2rgb_t + 0.5, 0.0)            # (N,3)
      colors_precomp_t = torch.clamp(colors_precomp_t, 0.0, 1.0) 
  
      out = []
      
      # 4) 3D-to-2D coordinate projection
      # p_view_t (카메라 공간 좌표)를 사용하여 순수하게 2D 중심을 계산합니다.
      w, h = intrinsics["image_size"]
      fx, fy = intrinsics["fx"], intrinsics["fy"]
      cx, cy = intrinsics["cx"], intrinsics["cy"]
      
      # 카메라 공간 좌표(t_x, t_y, t_z) -> 픽셀 좌표
      x = p_view_t[:, 0]
      y = p_view_t[:, 1]
      z = p_view_t[:, 2]
      center2d_t = torch.stack([
          fx * x / z + cx,
          fy * y / z + cy
      ], dim=1)
      
      # 5) 3D-to-2D covariance matrix transformation
      # 이제 이 함수는 cov2d만 계산하여 반환합니다.
      cov2d_t = project_3d_to_2d(cov3D_t, p_view_t, view, intrinsics, device, valid_original_ids)
         
      out = {
                "id":         torch.arange(N, device=device, dtype=torch.long),   #(N, )
                "center2d":   center2d_t,    # (N,2)
                "cov2d":      cov2d_t,       # (N,2,2)
                "cov3D":      cov3D_t,       # (N, 6)
                "intensity":  opacity_t,     # (N,)
                "color":      colors_precomp_t,      # (N,3)
                "depth":      depths_t
            }

    return out
