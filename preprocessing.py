import numpy as np
from gaussian_splatting.utils.sh_utils import eval_sh
from gaussian_splatting.scene.gaussian_model import GaussianModel


def project_point(xyz, intrinsics):
    """
    3D 점을 2D 픽셀 좌표로 투영합니다.
    xyz: (3,) 배열, 카메라 좌표계 내 3D 좌표 (X, Y, Z)
    intrinsics: dict with keys fx, fy, cx, cy
    returns: (u, v) 픽셀 좌표
    """
    X, Y, Z = xyz
    u = intrinsics['fx'] * (X / Z) + intrinsics['cx']
    v = intrinsics['fy'] * (Y / Z) + intrinsics['cy']
    return np.array([u, v], dtype=np.float32)


def project_covariance(cov3d, xyz, intrinsics):
    """
    3×3 공분산을 2×2 이미지 평면 공분산으로 변환합니다.
    cov3d: (3,3) 공분산 행렬 Σ_3D
    xyz:   (3,) 대응 3D 중심 좌표
    intrinsics: 카메라 내부 파라미터
    returns: (2,2) 투영된 공분산 Σ_2D
    """
    X, Y, Z = xyz
    fx, fy = intrinsics['fx'], intrinsics['fy']
    # 투영 함수의 Jacobian J (2×3)
    J = np.array([
        [ fx / Z,          0.0, -fx * X / (Z * Z)],
        [   0.0,       fy / Z, -fy * Y / (Z * Z)]
    ], dtype=np.float32)
    # Σ_2D = J · Σ_3D · Jᵀ
    cov2d = J @ cov3d @ J.T    #이미 camera 좌표계에서의 공분산이므로, J.T로 transpose(World→Camera 변환 필요 없음)
    return cov2d


def preprocess_from_model(model: GaussianModel,
                          intrinsics: dict,
                          cam_pose: np.ndarray = None):
    """
    GaussianModel 인스턴스에서 직접 가져온 파라미터로 전처리.
    Returns: list of dict with keys:
      - id       : int (global ID)
      - center   : (2,) pixel coords
      - cov2d    : (2,2) projected covariance
      - Cov      : same as cov2d
      - Cov_inv  : inverse of cov2d
      - intensity: float (opacity)
      - color    : (3,) RGB
      - min_radius, max_radius: floats for SESC & mode decision
    """
    # 1) 텐서→NumPy
    xyz3D      = model.get_xyz.detach().cpu().numpy()           # (N,3)
    cov3D      = model.get_covariance().detach().cpu().numpy()  # (N,3,3)
    sh_all     = model.get_features.detach().cpu().numpy()      # (N,B,3)
    opacity    = model.get_opacity.detach().cpu().numpy().squeeze()  # (N,)

    out = []
    N, B, _ = sh_all.shape
    for i in range(N):
        # 2) world→camera (필요 시)
        xyz = xyz3D[i]
        if cam_pose is not None:
            xyz_h   = np.concatenate([xyz, [1.0]])
            xyz_cam = (cam_pose @ xyz_h)[:3]
        else:
            xyz_cam = xyz

        # 3) 3D→2D projection
        center2d = project_point(xyz_cam, intrinsics)
        # 4) covariance projection
        cov2d    = project_covariance(cov3D[i], xyz_cam, intrinsics)
        # 5) SH→RGB via GraphDECO's eval_sh
        view_dir = xyz_cam / np.linalg.norm(xyz_cam)
        sh_coeffs = sh_all[i]  # shape (B,3)
        color = eval_sh(sh_coeffs, view_dir)  # returns (3,) float32

        out.append({
            "id":           i,      # global ID
            "center":     center2d,
            "cov2d":      cov2d,
            "Cov":        cov3d,
            "cov2d_inv":    np.linalg.inv(cov2d),
            "intensity":  float(opacity[i]),
            "color":      color.astype(np.float32),
            "min_radius": 0,
            "max_radius": 0
        })
    return out