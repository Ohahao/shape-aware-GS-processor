import time
import os
import re
import sys
import numpy as np
import torch
import math
import json
import lpips
import torchvision.utils as vutils
from typing import Tuple
import cv2  # OpenCV for image loading
from argparse import ArgumentParser
from tqdm import tqdm
from collections import defaultdict
from torch.cuda.amp import autocast
from SESC import analyze_gaussian_shape, should_skip_tile, should_skip_batch
from gaussian_reuse_cache import GaussianCache, traverse_tiles
from hybrid_array import compute_tile_contrib_batch
from preprocessing import preprocess_from_model
from gaussian_splatting.arguments import ModelParams, get_combined_args
from gaussian_splatting.scene import Scene, GaussianModel
from gaussian_splatting.scene.dataset_readers import sceneLoadTypeCallbacks, SceneInfo
from gaussian_splatting.utils.graphics_utils import fov2focal


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:24"
torch.set_grad_enabled(False) 

def compute_psnr(reference, estimate):

    ref = reference.astype(np.float32)
    est = estimate.astype(np.float32)
    
    
    # 2) normalize x ( normalized input 사용)
    '''
    max_val = ref.max()
    if max_val > 1.0:
        ref = ref / max_val
        est = est / max_val
    '''
    # 3) MSE 계산
    mse = np.mean((ref - est) ** 2)
    if mse == 0:
        return float('inf')  # 완벽 일치
    
    psnr = 20 * np.log10(1.0 / np.sqrt(mse))
    return psnr

def compute_lpips(gt_tensor: torch.Tensor, img_tensor: torch.Tensor, device: torch.device):
    """
    Compute LPIPS (v0.1, spatial=False) between two images given as torch.Tensors.
    
    Args:
        gt_tensor: torch.Tensor of shape (3, H, W), (H, W, 3), (1, 3, H, W) or (N, 3, H, W) or (N, H, W, 3)
        img_tensor: same as gt_tensor
        device: torch.device, e.g., torch.device('cuda:0') or torch.device('cpu')
    
    Returns:
        LPIPS distance tensor of shape (N, 1) if spatial=False.
    """
    def _prep(tensor):
        tensor = tensor.float()
        # Handle tensor dimensions
        if tensor.ndim == 3:
            C, H, W = tensor.shape
            if C == 3:
                tensor = tensor.unsqueeze(0)  # (1,3,H,W)
            elif tensor.shape[2] == 3:
                tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
            else:
                raise ValueError(f"Expected (3,H,W) or (H,W,3), got {tensor.shape}")
        elif tensor.ndim == 4:
            N, A, B, C = tensor.shape
            if A == 3:
                pass  # already (N,3,H,W)
            elif C == 3:
                tensor = tensor.permute(0, 3, 1, 2)  # (N,3,H,W)
            else:
                raise ValueError(f"Expected (N,3,H,W) or (N,H,W,3), got {tensor.shape}")
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got ndim={tensor.ndim}")
        return tensor.to(device)

    ref = _prep(gt_tensor)
    est = _prep(img_tensor)

    # Normalize to [-1, 1]
    ref = ref * 2.0 - 1.0
    est = est * 2.0 - 1.0

    # Compute LPIPS
    loss_fn = lpips.LPIPS(net='alex', version='0.1', spatial=False).to(device)
    dist = loss_fn(ref, est)
    return dist.item()


def gaussians_in_tile_fast_gpu(tile, centers_t, radii_t, tile_size=13):

    x0, x1 = tile.x*tile_size, (tile.x+1)*tile_size
    y0, y1 = tile.y*tile_size, (tile.y+1)*tile_size
    cx, cy, r = centers_t[:,0], centers_t[:,1], radii_t
    MARGIN = 1.0  # or 2.0
    mask = ~(
        (cx + r*MARGIN < x0) |
        (cx - r*MARGIN >= x1) |
        (cy + r*MARGIN < y0) |
        (cy - r*MARGIN >= y1)
    )

    out_of_tile_mask = (
        (cx + r < x0) |
        (cx - r >= x1) |
        (cy + r < y0) |
        (cy - r >= y1)
    )
    
    # 타일 밖에 있는 가우시안의 개수를 출력
    num_out_of_tile_gaussians = out_of_tile_mask.sum().item()  # True인 값의 개수

    return torch.nonzero(mask, as_tuple=False).view(-1)  # [g_id, ...]


def compute_intrinsics(camera):

    width, height = camera.image_width, camera.image_height 
    fx = fov2focal(camera.FoVx, width)
    fy = fov2focal(camera.FoVy, height)
    cx = width / 2
    cy = height / 2
  
    intrinsics = {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "image_size": (width, height)
        }
        
    #for k, v in intrinsics.items():
    #  print(f"{k}: {v}")
            
    return intrinsics, width, height


def analytic_2x2_eigendecomp(M: torch.Tensor, eps: float = 1e-6):

    single = (M.dim() == 2)
    if single:
        M = M.unsqueeze(0)  # (1,2,2)

    a = M[:, 0, 0]
    b = M[:, 0, 1]
    c = M[:, 1, 1]

    t = (a + c) * 0.5
    d = (a - c) * 0.5
    disc = (d**2 + b**2).sqrt()

    lam_min = t - disc
    lam_max = t + disc

    vx = b
    vy = lam_min - a
    norm = (vx**2 + vy**2).sqrt().clamp(min=eps)
    e_min = torch.stack([vx / norm, vy / norm], dim=1)  # (N,2)

    if single:
        lam_min = lam_min.squeeze(0)
        lam_max = lam_max.squeeze(0)
        e_min    = e_min.squeeze(0)

    return lam_min, lam_max, e_min



def shape_analysis(cov2d_t):
    C_x = cov2d_t[:, 0, 0]
    C_y = cov2d_t[:, 0, 1]
    C_z = cov2d_t[:, 1, 1]
    
    #compute eigenvalue
    trace = (C_x + C_z)
    det = (C_x * C_z - C_y * C_y)
    det_inv = torch.zeros_like(det)
    nonzero = det != 0
    det_inv[nonzero] = 1.0 / det[nonzero]

    conic = torch.stack([
        C_z * det_inv,    # A = C_z / det
        -C_y * det_inv,   # B = -C_y / det
        C_x * det_inv     # C = C_x / det
    ], dim=1)           # shape (M, 3)
    
    mid  = 0.5 * (C_x + C_z)                   # shape (M,)
    disc = mid * mid - det                     # shape (M,)
    disc_clamped = torch.clamp(disc, min=0.1)  # shape (M,)
    sqrt_disc    = torch.sqrt(disc_clamped)    # shape (M,)
    
    lam1 = mid + sqrt_disc                     # largest eigenvalue (σ²)
    lam2 = mid - sqrt_disc                     # smallest eigenvalue (σ²)
    #lam2 = torch.clamp(lam2, min=0)
    lambda_max = torch.max(lam1, lam2)
    my_radius = torch.ceil(3.0 * torch.sqrt(lambda_max))

    R1 = torch.rsqrt(lam1)
    R2 = torch.rsqrt(lam2)
    ratio = R2 / R1
    mode_spiky = (R2/R1) > 2.0
    
    num_spiky = mode_spiky.sum().item()
    #N_total = cov2d_t.shape[0]
    #print(f"Spiky인 Gaussian 개수: {num_spiky}, 전체 gaussian 개수: {N_total}")
    
    return my_radius, R1, R2, mode_spiky
  



if __name__ == "__main__":
    # 0) Paths & device (argument setting)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    MAC_PER_BLENDING = 4
    FLOPS_PER_BLENDING = 0
    TEST_GAUSSIAN_LIST =  [27, 456, 632]

    with torch.no_grad():
      parser = ArgumentParser(description="Testing script parameters")
      model = ModelParams(parser, sentinel=False)
      parser.add_argument("--iteration", default=-1, type=int)  #pretrained model iters. °áÁ¤(¸ðµç dataset¿¡ ´ëÇØ ÅëÀÏ) 
      parser.add_argument("--skip_train", action="store_true")  #skip_train ¿©ºÎ °áÁ¤(ÀÔ·Â: skip train, ÀÔ·Â x: skip test)
      parser.add_argument("--format_type", default="", type=str)  #format type °áÁ¤ (colmap or blender)
      parser.add_argument("--view_index", "-v", type=int, default=0, help="Index of the view to render.")
      parser.add_argument("--device", default=None, type=str, help="Device to run on (e.g. 'cuda:0' or 'cpu')")
      args = get_combined_args(parser)
      dataset = model.extract(args)   
      
      
      print("=== Dataset Arguments ===")
      for k, v in vars(dataset).items():
          print(f"{k}: {v}")
      print("=========================")
      
      chosen = args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu")
      device = torch.device(chosen)
      iteration = args.iteration
      format_type = args.format_type
      source_path = args.source_path
      scene_name = os.path.basename(os.path.normpath(source_path))
      
      print("Using device:", device)
      print("Pretrained model iters. :", iteration)
      print("format type: ", format_type)
      print("Workload: ", scene_name)
      print("최종본 실행 ... ing")

      # 1) 3D Gaussian load & train/test dataset
      gaussians = GaussianModel(sh_degree=dataset.sh_degree, optimizer_type="default")
      scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
      skip_train = args.skip_train
      view_index = args.view_index
       
      #source path의 cameara extrinsics file(sparse/0/images.bin)에서 하나씩 읽어서 views에 할당
      if not skip_train:
        views = scene.getTrainCameras()
      else: 
        views = scene.getTestCameras()
      
      num_views = len(views)
      if args.view_index < 0 or args.view_index >= num_views:
          print(f"Error: view_index {args.view_index} out of bounds (0 to {num_views-1})")
          sys.exit(1)
      
      view = views[view_index]
      W, H = view.image_width, view.image_height
      
      ### === debug: resolution 줄이기 === ###
      # 해상도 축소 비율
      '''
      resize_ratio = 0.5
      W = int(W * resize_ratio)
      H = int(H * resize_ratio)
      
      
      # 카메라 및 이미지 해상도 축소 적용
      view.image_width = W
      view.image_height = H
      view.original_image = torch.nn.functional.interpolate(
          view.original_image.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False
      ).squeeze(0)
      '''
      
      # 2) Prepare caches & rendering
      cache = GaussianCache( num_sets=1024, ways=8 )
      metrics = defaultdict(int)

      # 3) Render loop 
      t_all = time.time()
      # single view rendering    
      print(f"[View idx={view_index}] image name: {view.image_name}")
      
      # 3-1) rendering image 설정  
      img = torch.zeros((3, H, W), dtype=torch.float32, device=device) 
      gt = view.original_image[0:3, :, :]
      
      #view 마다 intrinsics 계산
      intr, W, H = compute_intrinsics(view) 

      # 3-2) gaussian preprocessing
      start_time = time.time()            
      gaussians_for_view = preprocess_from_model(gaussians, intr, view, device)
      elapsed = time.time() - start_time            
      print(f"Finished preprocessing gaussians (elapsed: {elapsed:.3f} sec)")
      
      
      # 3-3) shape analysis 실시 
      cov2d_t  = gaussians_for_view["cov2d"].to(device) 
      #jitter = 1e-6 * torch.eye(2, device=device).unsqueeze(0)
      #cov2d_stabilized = cov2d_t + jitter
      
      # 3. 이 안정화된 행렬의 역행렬을 구합니다.
      cov_inv_t = torch.linalg.inv(cov2d_t)
      if torch.isnan(cov_inv_t).any() or torch.isinf(cov_inv_t).any():
              print("[DEBUG] NaN or Inf detected in cov_inv_t!")
      
      my_radius, R1, R2, mode_spiky = shape_analysis(cov2d_t)
      N_total = cov2d_t.shape[0]

      gaussians_for_view["cov_inv"] = cov_inv_t
      print(f"Finished shape analysis of {view_index}th frame")
      
      centers_t   = gaussians_for_view["center2d"].to(device)  # (N_remaining, 2)
      radii_t     = my_radius.to(device)                      # (N_remaining,)
      intensity_t = gaussians_for_view["intensity"].to(device) 
      
      # 3-4) traverse tiles
      print("Start traverse tiles") 
      tiles = list(traverse_tiles(gaussians_for_view, format_type, H, W, tile_size=13, device=device)) 
      print("총 처리할 타일 수:", len(tiles))

      '''   
      #[METRICS] intersected tile 개수
      #[METRICS] tiles per gaussian 
      intersected_tile_list= []

      # 여러 개의 테스트할 Gaussian ID
      TEST_GAUSSIANS = gaussians_for_view["id"][TEST_GAUSSIAN_LIST]  # 예: [123, 456, 789]

      # [METRICS] intersected tiles
      for gs_metrics in TEST_GAUSSIANS:
          tile_count = 0

          for tile in tqdm(tiles, desc=f"Tiles for Gaussian {gs_metrics.item()}", unit="tile"):
              # gaussian-tile intersection test
              idxs = gaussians_in_tile_fast_gpu(tile, centers_t, radii_t)

              # 4-1) SESC: skip할 gaussians 걸러내기  
              skip_mask = should_skip_batch(
                  idxs, centers_t, cov_inv_t, radii_t, intensity_t,
                  tile.x, tile.y, R1, R2, device,
                  tile_size=13, alpha_thresh=1e-8
              )
              idxs = idxs[~skip_mask]
  
              # 해당 Gaussian이 이 타일에 포함되어 있으면 카운트
              if gs_metrics in idxs:
                  tile_count += 1
          
          print(f" intersected tiles: {tile_count} at {gs_metrics.item()}")
          intersected_tile_list.append((gs_metrics.item(), tile_count))
      '''
      # 타이밍 변수 초기화
      total_time_skip    = 0.0
      total_time_cache   = 0.0
      total_time_split   = 0.0
      total_time_compute = 0.0

      prev_idxs = None    
      add_target = None
      start_time1 = time.time()
      

      # 4) alpha-blending per tile
      gaussians_per_tile_list = []
      gaussians_per_tile_list_metrics = []
      tile_idx_list = []
      tile_coords_list = []
      skipped_gaussians_per_tile_list = []
      tile_idx = 0
      total_skipped = 0
      total_test_gaussians = 0

      for tile in tqdm(tiles, desc="Tiles", unit="tile"):
          if tile.x0 == 208 and tile.y0 == 364:
              print(f"tile_idx: {tile_idx}")
              add_target = tile_idx
          
          #gaussian-tile intersection test
          idxs_all = gaussians_in_tile_fast_gpu(tile, centers_t, radii_t) 
          N = centers_t.shape[0]
          #idxs_all = torch.arange(N, device=device)
          t0 = time.perf_counter()   

          # 4-1) SESC: skip할 gaussians 걸러내기                   
          skip_mask = should_skip_batch(
              idxs_all, centers_t, cov_inv_t, radii_t, intensity_t,
              tile.x, tile.y, R1, R2, device, 
              tile_size=13, alpha_thresh=1e-8
          )

          idxs_kept = idxs_all[~skip_mask]
          idxs_skipped = idxs_all[skip_mask]
          skipped_gaussians = idxs_skipped.numel()
          total_gaussians = idxs_all.numel()
          total_skipped    += skipped_gaussians 
          total_test_gaussians += idxs_all.numel()


          idxs = idxs_kept #편의를 위해 ... 

          t1 = time.perf_counter()
          total_time_skip += (t1 - t0)
          
          # 4-2) cache hit rate 계산 
          '''
          hits = cache.access_batch(idxs)  # (M,) bool
          metrics["cache_hits"]   += hits.sum().item()
          misses = (~hits).sum().item()
          metrics["cache_misses"] += misses
          metrics["mem_access"]   += misses
          t2 = time.perf_counter()
          total_time_cache += (t2 - t1)
          
          cache_hits = metrics.get("cache_hits", 0)
          cache_misses = metrics.get("cache_misses", 0)
          #print(f"\n cache_hits: {cache_hits}, cache_misses: {cache_misses}")
          '''
          # 4-3) depth에 따른 gaussian sorting
          depths_tile = gaussians_for_view["depth"][idxs]                       # (M,)
          sorted_vals, sorted_idx = torch.sort(depths_tile)  # 깊이 작은 순으로 정렬
          idxs = idxs[sorted_idx] 
          mask_sp_sorted = mode_spiky[idxs]                # (M,) bool; 각 index의 가우시안이 spiky인지 아닌지 알려줌(boolean)
          #sp_local = idxs  # 모두 뾰족한 것으로 처리
          #sm_local = torch.tensor([], dtype=torch.long, device=device)
          sp_local = idxs[mask_sp_sorted]    #spiky gaussians id 저장
          sm_local = idxs[~mask_sp_sorted]   #smooth gaussians id 저장
          
          t3 = time.perf_counter()
          
          # 4-4) tile별alpha-blending (배치) 

          pixel = compute_tile_contrib_batch(
                centers   = gaussians_for_view["center2d"],    # (K_sp,2)
                cov2d     = gaussians_for_view["cov2d"],       # (K_sp,2,2)
                intensity = gaussians_for_view["intensity"],   # (K_sp,)
                color     = gaussians_for_view["color"],       # (K_sp,3,1,1)
                cov_inv   = gaussians_for_view["cov_inv"],
                N_total   = N_total,    #전체 gaussian 개수 
                N         = idxs.numel(),
                idxs      = idxs,       #tile intersected gaussians id
                sp        = sp_local,   #spiky gaussians id
                sm        = sm_local,   #smooth gaussians id
                tile      = tile,
                N_tiles   = len(tiles),
                device    = device
            ) 
   
          # 4-5) pixel 병합 
          pixel = pixel.to(device)
          img[:, tile.y0:tile.y1, tile.x0:tile.x1] = pixel
          
          
          # 4-6) metrics 계산 
          if sp_local.numel() > 0:
            metrics["total_alpha_computation"] += 169 * sp_local.numel()  
            metrics["alpha_computation_RE"] += 169 * sp_local.numel()           
            #print(f"RE: {metrics['alpha_computation_RE']}")
          if sm_local.numel() > 0:
            metrics["total_alpha_computation"] += (49  + 120 * 0.1) * sm_local.numel()        
            metrics["alpha_computation_IE"] += 120 * sm_local.numel()
            metrics["alpha_computation_RE"] += 49 * sm_local.numel()  
            #print(f"IE: {metrics['alpha_computation_IE']}") 
          metrics["tiles_total"] += 1
          
          # 가우시안 개수만 계산
          if tile_idx == 7 or tile_idx == 50 or tile_idx == add_target:
            tile_coords = (tile.x0, tile.y0, tile.x1, tile.y1)
            print(f" idx value: {idxs.numel()}, idx shape: {idxs.shape}")
            gaussians_per_tile_list_metrics.append(idxs_kept.numel()) 
            tile_idx_list.append(tile_idx)
            tile_coords_list.append(tile_coords)
            skipped_gaussians_per_tile_list.append(idxs_skipped.numel())
            #metrics["gaussians_per_tile"] = idxs.numel()
            #metrics["tile_idx"] = tile_idx
            #metrics["tile_coords"] = (tile.x0, tile.y0, tile.x1, tile.y1)
          
          tile_idx += 1
          gaussians_per_tile_list.append(idxs.numel())
      
    
      # Metrics 계산
      metrics["total_gaussians_per_tile"] = gaussians_per_tile_list_metrics
      metrics["mac_per_tile"] = MAC_PER_BLENDING * metrics["total_gaussians_per_tile"] 
      metrics["flops_per_tile"] = FLOPS_PER_BLENDING * metrics["total_gaussians_per_tile"] 
      #metrics["intersected_tiles"] = intersected_tile_list
      metrics["skipped_gaussians_per_tile"] = skipped_gaussians_per_tile_list
      metrics["skipped_rate_percent"] = 0.0 if total_test_gaussians == 0 else round(100.0 * total_skipped / total_test_gaussians, 2)
      metrics["avg_gaussians_per_tile"] = 0.0 if not gaussians_per_tile_list else sum(gaussians_per_tile_list) / len(gaussians_per_tile_list)
      metrics["max_gaussians_per_tile"] = max(gaussians_per_tile_list) if gaussians_per_tile_list else 0
      metrics["tile_idx"] = tile_idx_list
      metrics["tile_coords"] = tile_coords_list




      # (선택) 디버깅 출력
      print(f" \n전체 {len(gaussians_per_tile_list)}개 타일 처리")
      print(f"   평균 가우시안/타일: {metrics['avg_gaussians_per_tile']}")
      print(f"   최대 가우시안/타일: {metrics['max_gaussians_per_tile']}")
      print(f"   특정 타일에서 가우시안: {metrics['total_gaussians_per_tile'][0]} at {metrics['tile_coords'][0]}")
      print(f"   alpha 연산량: {metrics['total_alpha_computation']}, RE: {metrics['alpha_computation_RE']}, IE: {metrics['alpha_computation_IE']}")
      print(f"   skipping rate: {metrics['skipped_rate_percent']}(%) ")

      t4 = time.perf_counter()
      total_time_compute += (t4 - t3)
      
      print(f"\n[Timing Summary]")
      print(f"  Skip batch time   : {total_time_skip:.3f}s")
      print(f"  Cache access time : {total_time_cache:.3f}s")
      print(f"  Split mode time   : {total_time_split:.3f}s")
      print(f"  Compute batch time: {total_time_compute:.3f}s")
      
      elapsed1 = time.time() - start_time 
      print(f"Finished rendering (elapsed: {elapsed1:.3f} sec)")   

      # 5) rendered image 저장 
      save_dir = "/home/hyoh/shape-aware-GS-processor/results/images/"
      rendered_path = os.path.join(save_dir, f"{scene_name}_rendered_{view_index:03d}.png")
      gt_path       = os.path.join(save_dir, f"{scene_name}_gt_{view_index:03d}.png")
      vutils.save_image(img, rendered_path)
      vutils.save_image(gt,  gt_path)


      # 6) Compute PSNR and FPS
      img_np = img.detach().cpu().numpy()
      gt_np = gt.detach().cpu().numpy()
      psnr_val = compute_psnr(gt_np, img_np)
      lpips_val = compute_lpips(gt, img, device)
      #lpips_val = compute_lpips(gt_np, img_np, device)

      total_time = elapsed1
  
      # Report metrics
      print(f"PSNR: {psnr_val:.4f} dB")
      print(f"LPIPS: {lpips_val:.4f} ")
      #print(f"LPIPS: {lpips_val:.4f} ")
      print(f"Total time: {total_time:.3f}s")
      
      # 7) cache rates & tile skipped rates - 수정 필요
      '''
      cache_hits = metrics.get("cache_hits", 0)
      cache_misses = metrics.get("cache_misses", 0)
      tiles_total = metrics.get("tiles_total", 0)
      tiles_skipped = metrics.get("tiles_skipped", 0)
  
      cache_hit_rate = cache_hits / (cache_hits + cache_misses) if (cache_hits + cache_misses) > 0 else 0.0
      tile_skipped_rate = tiles_skipped / (tiles_total + tiles_skipped) if tiles_total > 0 else 0.0
  
      metrics["cache_hit_rate"] = cache_hit_rate
      metrics["tile_skipped_rate"] = tile_skipped_rate
  
      print(f"Cache hit rate: {cache_hit_rate:.2%}")
      print(f"Tile skipped rate: {tile_skipped_rate:.2%}")
      ''' 

      # Dump to JSON
      metrics.update({"psnr": psnr_val, "lpips": lpips_val})
      with open(f"results/metrics/metrics_{scene_name}_{view_index}.json", 'w') as f:
          json.dump(metrics, f, indent=4)
      
      print(f"gt shape: {gt.shape}, img shape: {img.shape}")
      
      
      