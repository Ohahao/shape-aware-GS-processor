"""Module providing a function printing """

import time
import os
import re
import numpy as np
import cv2  # OpenCV for image loading
from collections import defaultdict
from SESC import analyze_gaussian_shape, should_skip_tile
from gaussian_reuse_cache import GaussianCache, traverse_tiles
from hybrid_array import compute_tile_contrib
from preprocessing import preprocess_from_model
from gaussian_splatting.scene import Scene, GaussianModel
from gaussian_splatting.scene.dataset_readers import sceneLoadTypeCallbacks, SceneInfo

# ———— 기존 구현 모듈 불러오기 ————
# compute_tile_contrib, analyze_gaussian_shape, should_skip_tile, GaussianCache
# load_synthetic_dataset, load_ego_dataset, render_frame_shape_aware, compute_psnr


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


def load_dataset(
    path: str,
    format_type: str,
    images: str = "images",
    depths: str = "",
    eval: bool = False,
    train_test_exp: bool = False,
    dataset_type: str = "",
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
            depths=depths, 
            eval=eval,
            train_test_exp=train_test_exp,
            llffhold=llffhold
        )


    elif key in ("blender"):
        loader = sceneLoadTypeCallbacks.get("Blender")
        if loader is None:
            raise ValueError("Blender loader not available.")
        scene_info = loader(
            path=path,
            white_background=white_background,
            depths=depths, 
            eval=eval,
            dataset_type=dataset_type,
            extension=extension
        )
    else:
        raise ValueError(f"Unknown format_type '{format_type}': choose 'colmap' or 'blender'.")

    return scene_info


def load_gt_images(dataset_path, extension=".png"):
    """
    Load ground-truth images for the synthetic dataset.

    Parameters:
        dataset_path (str): Path to the directory containing ground-truth images.
        extension (str): File extension of the images (default: ".png").

    Returns:
        list of np.ndarray: A list of ground-truth images as NumPy arrays.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    
    #input image만 읽기(depth map 제외)
    match = re.search(r'(\d+)', filename)
    extract_number =  int(match.group(1)) if match else float('inf')
    filenames = sorted(
        [f for f in os.listdir(dataset_path) if f.endswith(extension)],
        key=extract_number
    )

    gt_images = []
    for filename in filenames:
        if filename.endswith(extension):
            image_path = os.path.join(dataset_path, filename)
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # Load image
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            # Normalize image to range [0, 1] if needed(psnr 계산을 위해)
            image = image.astype(np.float32) / 255.0
            gt_images.append(image)

    if not gt_images:
        raise ValueError(f"No images found in {dataset_path} with extension {extension}")

    return gt_images

def load_ego_gt_images(dataset_path, extension=".png"):
    gt_images = []
    txt_file_path = os.path.join(dataset_path, "test.txt")
    image_dir = os.path.join(dataset_path, "images")

    with open(txt_file_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            filename = line.strip()
            image_path = os.path.join(image_dir, filename)
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # Load image
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            # Normalize image to range [0, 1] if needed(psnr 계산을 위해)
            image = image.astype(np.float32) / 255.0
            gt_images.append(image)

    if not gt_images:
        raise ValueError(f"No images found in {dataset_path} with extension {extension}")

    return gt_images


def get_reference_images(dataset_name):
    """
    Retrieve reference images for PSNR computation.

    Parameters:
        dataset_name (str): The name of the dataset (e.g., "synthetic", "ego").

    Returns:
        list of np.ndarray: A list of reference images.
    """
    if dataset_name == "synthetic":
        # Load ground-truth images for synthetic dataset
        return load_gt_images()
    elif dataset_name == "ego":
        # Load reference RGB images for ego dataset
        return load_ego_gt_images()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def gaussians_in_tile(tile, gaussians, tile_size=13):
    """
    현재 타일(tile)이 영향을 받을 가능성이 있는 Gaussians만 필터링하여
    (gaussian_index, gaussian_dict) 튜플의 리스트로 반환합니다.

    Args:
        tile: namedtuple("Tile", ["x", "y"]) 형태로, 타일의 가로(x)·세로(y) 인덱스입니다.
        gaussians: 각 원소가 {
            "center": (cx, cy),        # 2D 투영된 Gaussian 센터
            "max_radius": float,       # shape 분석으로 얻은 최대 반지름 (σ_max)
            …                          # 기타 필드 (Cov, intensity 등)
        } 형태인 리스트입니다.
        tile_size: 하나의 타일이 차지하는 픽셀 너비 (기본 13)

    Returns:
        List[(int, dict)]: 타일에 기여할 수 있는 Gaussian의 (원본 인덱스, 딕셔너리) 리스트
    """
    x0 = tile.x * tile_size
    x1 = x0 + tile_size
    y0 = tile.y * tile_size
    y1 = y0 + tile_size

    visible = []
    for  g in gaussians:
        cx, cy = g["center"]
        # 3σ 영역을 영향 범위로 가정
        influence_radius = 3.0 * g["max_radius"]

        # 타일 범위 [x0, x1)×[y0, y1)와 원형(중심(cx,cy), 반지름) 간의 겹침 검사
        if (cx + influence_radius < x0) or (cx - influence_radius >= x1) or \
           (cy + influence_radius < y0) or (cy - influence_radius >= y1):
            # 완전히 떨어져 있으면 스킵
            continue

        # 겹침이 있다면 이 Gaussian은 이 타일에 기여할 수 있음
        visible.append((g["id"], g))

    return visible

def compute_intrinsics(dataset_name):
    """
    Compute camera intrinsics based on the dataset.

    Parameters:
        dataset_name (str): The name of the dataset (e.g., "synthetic", "ego").
        H (int): The height of the image.
        W (int): The width of the image.

    Returns:
        dict: A dictionary containing intrinsics parameters (e.g., focal length, principal point).
    """
    if dataset_name == "synthetic":
        # Example intrinsics for synthetic dataset
        intrinsics = {
            "fx": 800.0,
            "fy": 800.0,
            "cx": 800/2,
            "cy": 800/2,
            "image_size": (800, 800)
        }
    elif dataset_name == "ego":
        # Example intrinsics for ego dataset
        intrinsics = {
            "fx": 2000.0,
            "fy": 1000.0,
            "cx": 2000/2,
            "cy": 1000/2,
            "image_size": (2000, 1000)
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return intrinsics


if __name__ == "__main__":
    # 0) 경로 설정
    synthetic_dataset_path = "nerf_synthetic/chair"  # NeRF synthetic scenes
    ego_dataset_path = "OmniBlender/archiviz-flat"  # ego egocentric scenes

    # 1) 데이터셋 로딩
    datasets = {
        "synthetic": load_dataset(path=synthetic_dataset_path, format_type="Blender", images="images", eval=True, train_test_exp=False, dataset_type="nerf_synthetic") # 논문에서 사용된 NeRF synthetic scenes
        #"ego":    load_dataset(path=ego_dataset_path, format_type="Blender", images="images", eval=True, train_test_exp=False, dataset_type="omniblender"),  # ego
    }

    # 2) 결과 저장용
    results = {}

    # 3) 각 데이터셋별로 렌더링 및 메트릭 수집
    for name, scene_info in datasets.items():
        pc = scene_info.point_cloud
        frames = scene_info.test_frames
        shape_metrics = defaultdict(int)
        
        intrinsics = compute_intrinsics(name)  # Compute intrinsics based on dataset name
        sh_degree = 3   #학습은 하지 않기 때문에 sh_degree, optimizer_type은 임의로 지정(사용안함)
        optimizer_type = "default"

        # 3D Gaussian 모델 생성 및 전처리
        gaussians_3d = GaussianModel(sh_degree, optimizer_type)    #3d Gaussian 모델 인스턴스화
        gaussians = preprocess_from_model(gaussians_3d, intrinsics)  # 3D Gaussian 모델을 2D로 전처리/ intrinsics는 카메라 내부 파라미터(dataset에 따라 다름)

        #H, W 정보 로드(dataset에 따라 다름)
        if name == "synthetic":
            H, W = 800, 800
        elif name == "ego":
            H, W = 1920, 1080

        # shape analysis 판단 기준
        Rmin_avgs = []
        for g in gaussians_3d:
            Cov = gaussians_3d.get_covariance().detach().cpu().numpy()  # (N,3,3)   # 3x3 covariance matrix
            C_x = Cov[:, 0, 0]   # shape (N,)
            C_y = Cov[:, 1, 1]   # shape (N,)
            C_z = Cov[:, 2, 2]   # shape (N,)

            Rmin_min = (C_x + C_z) ** -0.5
            Rmin_max = (C_x * C_z - C_y ** 2) ** -0.25

            Rmin_avg = (Rmin_max + Rmin_min) / 2
            Rmin_avgs.append(Rmin_avg)

        shape_threshold = float(np.mean(Rmin_avgs))               # 평균값
        print(f"[{name}] Rmin_avg (shape_threshold) = {shape_threshold:.4f}")

        # Cache 초기화
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
            
            image = np.zeros((H, W), dtype=np.float32)
            
            # 타일 순회 (Z-order traversal 권장)
            for tile in traverse_tiles(frame_data):
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
                    
                    # shape analysis algo. (smooth vs spiky)
                    mode = analyze_gaussian_shape(g, shape_threshold)
                    
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
        ref_images = get_reference_images(name, dataset_path=synthetic_dataset_path if name == "synthetic" else ego_dataset_path)  
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
