import time
import numpy as np
from collections import defaultdict
from SESC import analayze_gaussian_shape, should_skip_tile
from gaussian_reuse_cache import Gaussiancache, traverse_tiles
from hybrid_array import compute_tile_contrib
from preprocessing import preprocess_from_model
from gaussian-splatting.scene import Scene, GaussianModel
from gaussian-splatting.scene.dataset_readers import sceneLoadTypeCallbacks, SceneInfo

# ———— 기존 구현 모듈 불러오기 ————
# compute_tile_contrib, analyze_gaussian_shape, should_skip_tile, GaussianCache
# load_synthetic_dataset, load_edina_dataset, render_frame_shape_aware, compute_psnr


def compute_psnr(reference, estimate):
    """
    Compute the Peak Signal-to-Noise Ratio (PSNR) between two images.

    Parameters:
        reference (np.ndarray): The reference image.
        estimate (np.ndarray): The estimated/reconstructed image.

    Returns:
        float: The PSNR value in decibels (dB).
    """
    mse = np.mean((reference - estimate) ** 2)
    if mse == 0:
        return float('inf')  # Perfect match
    max_pixel = 1.0  # Assuming normalized images in range [0, 1]
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def get_reference_images(dataset_name):
    """
    Retrieve reference images for PSNR computation.

    Parameters:
        dataset_name (str): The name of the dataset (e.g., "synthetic", "EDINA").

    Returns:
        list of np.ndarray: A list of reference images.
    """
    if dataset_name == "synthetic":
        # Load ground-truth images for synthetic dataset
        return load_gt_images()
    elif dataset_name == "EDINA":
        # Load reference RGB images for EDINA dataset
        return load_edina_gt_images()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def load_dataset(
    path: str,
    format_type: str,
    images: str = "images",
    depths: str = "",
    eval: bool = False,
    train_test_exp: bool = False,
    llffhold: int = 8,
    white_background: bool = True,
    extension: str = ".png"
) -> SceneInfo:
    """
    Load a dataset in either COLMAP or Blender (NeRF synthetic) format.

    Args:
        path:               Root directory of the dataset.
        format_type:        One of "colmap" or "blender" (case-insensitive).
        images:             Name of the images folder (for COLMAP).
        depths:             Name of the depths folder (optional).
        eval:               If True, use evaluation/test split.
        train_test_exp:     If True, include all cameras in training for COLMAP.
        llffhold:           LLFF hold-out interval for COLMAP.
        white_background:   Composite alpha over white background (for Blender).
        extension:          Image file extension (for Blender).

    Returns:
        SceneInfo:          Loaded scene information.
    """
    key = format_type.strip().lower()
    if key == "colmap":
        loader = sceneLoadTypeCallbacks.get("Colmap")
        if loader is None:
            raise ValueError("Colmap loader not available.")
        scene_info = loader(
            path=path,
            images=images,
            #depths=depths, depth 정보 있는 데이터셋 사용할 경우에 추가
            eval=eval,
            train_test_exp=train_test_exp,
            llffhold=llffhold
        )
    elif key in ("blender", "nerf", "synthetic"):
        loader = sceneLoadTypeCallbacks.get("Blender")
        if loader is None:
            raise ValueError("Blender loader not available.")
        scene_info = loader(
            path=path,
            white_background=white_background,
            #depths=depths, depth 정보 있는 데이터셋 사용할 경우에 추가
            eval=eval,
            extension=extension
        )
    else:
        raise ValueError(f"Unknown format_type '{format_type}': choose 'colmap' or 'blender'.")

    return scene_info


if __name__ == "__main__":
    # 0) 경로 설정
    synthetic_dataset_path = "path/to/synthetic_dataset"  # NeRF synthetic scenes
    edina_dataset_path = "path/to/EDINA_dataset"  # EDINA egocentric scenes

    # 1) 데이터셋 로딩
    datasets = {
        "synthetic": load_dataset(path=synthetic_dataset_path, format_type="Blender", images="images", eval=True, train_test_exp=False),  # 논문에서 사용된 NeRF synthetic scenes
        "EDINA":    load_dataset(path=edina_dataset_path, format_type="Blender", images="images", eval=True, train_test_exp=False),  # EDINA egocentric scenes
    }

    # 2) 결과 저장용
    results = {}

    # 3) 각 데이터셋별로 렌더링 및 메트릭 수집
    for name, frames in datasets.items():
        # Gaussian 준비 (첫 프레임 기준), 그리고 SESC를 위한 shape 분석
        sh_degree = 3   #학습은 하지 않기 때문에 sh_degree, optimizer_type은 임의로 지정(사용안함)
        optimizer_type = "default"
        gaussians_3d = GaussianModel(sh_degree, optimizer_type)    #3d Gaussian 모델 인스턴스화
        gaussians = preprocess_from_model(gaussians_3d, intrinsics)  # 3D Gaussian 모델을 2D로 전처리/ intrinsics는 카메라 내부 파라미터(dataset에 따라 다름)

        for g in gaussians:
            analyze_gaussian_shape(g)
        
        cache = GaussianCache(num_sets=1024, ways=8)
        metrics = {
            "psnr":      None,
            "total_time":0.0,
            "frames":    len(frames),
            "ops":       0,
            "mem_access":0,
            "cache_hits":0,
            "cache_misses":0,
            "tiles_skipped":0,
            "tiles_total":0
        }
        rendered_images = []
        
        # 4) 프레임 단위 렌더링
        start_all = time.time()
        for frame_idx, frame_data in enumerate(frames):
            # (옵션) 카메라/gaussian 업데이트
            update_gaussian_positions(gaussians, frame_data)    # gaussian 위치 업데이트
            
            H, W = frame_data["height"], frame_data["width"]
            image = np.zeros((H, W), dtype=np.float32)
            
            # 타일 순회 (Z-order traversal 권장)
            for tile in traverse_tiles(frame_data, order="z"):
                metrics["tiles_total"] += 1
                
                for g_id, g in enumerate(gaussians_in_tile(tile, gaussians)):
                    # SESC skip 여부 판단
                    if should_skip_tile(g, tile.x, tile.y):
                        metrics["tiles_skipped"] += 1
                        continue
                    
                    # Cache access
                    hit = cache.access(g_id)
                    if hit:
                        metrics["cache_hits"] += 1
                    else:
                        metrics["cache_misses"] += 1
                        metrics["mem_access"] += 1
                    
                    # mode 결정 (smooth vs spiky)
                    mode = "interp" if g["min_radius"] >= shape_threshold else "full"
                    
                    # 타일 기여 계산
                    contrib = compute_tile_contrib(g, tile.x, tile.y, mode=mode)
                    image[tile.y*13:(tile.y+1)*13, tile.x*13:(tile.x+1)*13] += contrib
                    
                    # 연산량 집계 (RE/IE 개수)
                    if mode == "full":
                        metrics["ops"] += 169
                    else:
                        shape_metrics['ops'] += 49  # RE ops
                        shape_metrics['ops'] += 120 # IE ops (counted as simpler int ops)
            # store rendered image
            rendered_images.append(image)
        metrics["total_time"] = time.time() - start_all
        
        # 5) 품질 평가 (PSNR)
        ref_images = get_reference_images(name)  
        psnrs = [ compute_psnr(ref, est) 
                for ref, est in zip(ref_images, rendered_images) ]
        metrics["psnr"] = float(np.mean(psnrs))
        
        # 6) 프레임당 Throughput 계산
        metrics["fps"] = metrics["frames"] / metrics["total_time"]
        
        results[name] = metrics

    # 7) 결과 출력
    print(f"{'Dataset':<10} | {'PSNR':>6} | {'FPS':>6} | {'Ops/frame':>10} | {'MemAcc':>7} | "
        f"{'CacheHit%':>9} | {'Skip%':>6}")
    print("-"*75)
    for name, m in results.items():
        hit_rate = 100*m["cache_hits"]/(m["cache_hits"]+m["cache_misses"])
        skip_rate = 100*m["tiles_skipped"]/m["tiles_total"]
        print(f"{name:<10} | {m['psnr']:6.2f} | {m['fps']:6.1f} | {m['ops']/m['frames']:10.1f} | "
            f"{m['mem_access']/m['frames']:7.2f} | {hit_rate:9.2f}% | {skip_rate:6.2f}%")
