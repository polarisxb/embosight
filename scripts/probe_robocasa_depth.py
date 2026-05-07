"""探测 RoboCasa 是否暴露 depth image 和 camera intrinsics.

2D bbox → 3D 世界坐标需要:
    1. depth image (每个像素的深度)
    2. camera intrinsic matrix K (fx, fy, cx, cy)
    3. camera extrinsic: 相机在世界坐标系中的位置 + 朝向

RoboCasa 基于 robosuite, robosuite 应该通过 camera_depths=True 暴露 depth.

运行: MUJOCO_GL=egl python scripts/probe_robocasa_depth.py
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

# 直接用 robosuite.make 以便加 camera_depths=True
import robocasa  # noqa: F401 — 注册 RoboCasa 环境
import robosuite as suite


def main():
    env = suite.make(
        env_name="PickPlaceCounterToCabinet",
        robots="PandaOmron",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["robot0_agentview_center"],
        camera_heights=256,
        camera_widths=256,
        camera_depths=True,           # ← 关键: 请求 depth
        camera_segmentations=None,
        control_freq=20,
    )
    obs = env.reset()

    print("=" * 60)
    print("ROBOCASA DEPTH + INTRINSICS PROBE")
    print("=" * 60)

    # 1) 所有 obs keys
    print(f"\n--- obs keys ---")
    for k in sorted(obs):
        v = obs[k]
        shape = getattr(v, "shape", "N/A")
        dtype = getattr(v, "dtype", type(v).__name__)
        print(f"  {k}: shape={shape}, dtype={dtype}")

    # 2) depth image
    depth_key = "robot0_agentview_center_depth"
    print(f"\n--- depth {depth_key} ---")
    if depth_key in obs:
        depth = obs[depth_key]
        print(f"  shape: {depth.shape}")
        print(f"  range: [{depth.min():.4f}, {depth.max():.4f}]")
        print(f"  dtype: {depth.dtype}")
        # 中心像素
        if depth.ndim >= 2:
            h, w = depth.shape[:2]
            center_val = float(depth[h // 2, w // 2].squeeze())
            print(f"  center pixel ({w//2},{h//2}) value: {center_val:.4f}")
    else:
        print(f"  !!! depth key '{depth_key}' NOT FOUND !!!")
        print(f"  alternative depth keys in obs:")
        for k in obs:
            if "depth" in k.lower():
                print(f"    {k}")

    # 3) camera intrinsics from mujoco model
    sim = env.sim
    print(f"\n--- camera intrinsics (from sim.model.cam_fovy) ---")
    try:
        cam_name = "robot0_agentview_center"
        cam_id = sim.model.camera_name2id(cam_name)
        fovy_deg = sim.model.cam_fovy[cam_id]
        height = 256
        width = 256
        fy = 0.5 * height / np.tan(0.5 * np.radians(fovy_deg))
        fx = fy  # 假设正方形像素
        cx = width / 2
        cy = height / 2
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        print(f"  cam_id: {cam_id}")
        print(f"  fovy(deg): {fovy_deg}")
        print(f"  fx={fx:.2f}, fy={fy:.2f}, cx={cx:.1f}, cy={cy:.1f}")
        print(f"  K matrix:\n{K}")
    except Exception as e:
        print(f"  ERROR: {e}")
        K = None

    # 4) camera extrinsic (position + orientation in world)
    print(f"\n--- camera extrinsic ---")
    try:
        cam_pos = sim.data.cam_xpos[cam_id].copy()
        cam_mat = sim.data.cam_xmat[cam_id].reshape(3, 3).copy()
        print(f"  position (world): {cam_pos}")
        print(f"  rotation matrix (world):\n{cam_mat}")
    except Exception as e:
        print(f"  ERROR: {e}")
        cam_pos, cam_mat = None, None

    # 5) 试做一次 2D→3D 反投影验证
    print(f"\n--- 2D→3D backprojection test ---")
    if depth_key in obs and K is not None and cam_pos is not None:
        depth = obs[depth_key]
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        # 取图像中心一点
        u, v = 128, 128
        try:
            z_buffer = float(depth[v, u])
        except Exception:
            z_buffer = float(depth[v, u].squeeze())

        # MuJoCo depth buffer [0,1] 规范化 → 真实距离 (需要 near/far)
        try:
            extent = float(sim.model.stat.extent)
            znear_ratio = float(sim.model.vis.map.znear)
            zfar_ratio = float(sim.model.vis.map.zfar)
            near = znear_ratio * extent
            far = zfar_ratio * extent
            # Inverted perspective projection
            if z_buffer >= 1.0:
                real_z = far
            else:
                real_z = near / (1.0 - z_buffer * (1.0 - near / far))
            print(f"  pixel ({u},{v}) depth_buffer = {z_buffer:.4f}")
            print(f"  extent={extent:.3f} near={near:.4f} far={far:.3f}")
            print(f"  real_z(camera forward distance) = {real_z:.3f}m")
        except Exception as e:
            print(f"  depth normalization failed: {e}")
            real_z = None

        if real_z is not None and real_z < far:
            # 像素 → 相机系 (pinhole 模型; MuJoCo camera looks along -z)
            # image coord: u right, v down; cam coord: x right, y up, z back
            fx_val, fy_val = float(K[0, 0]), float(K[1, 1])
            x_cam = (u - cx) * real_z / fx_val
            y_cam = -(v - cy) * real_z / fy_val
            z_cam = -real_z
            pt_cam = np.array([x_cam, y_cam, z_cam])
            pt_world = cam_mat @ pt_cam + cam_pos
            print(f"  camera-frame coord: ({x_cam:.3f}, {y_cam:.3f}, {z_cam:.3f})")
            print(f"  world coord: {pt_world}")
            print(f"  (sanity check: z_world should be ~0.9-1.1m for table height)")
    else:
        print("  SKIP (missing depth/K/extrinsic)")

    # 6) 检查是否有其他有用 API
    print(f"\n--- sim.data / sim.model useful attrs ---")
    for attr in ["cam_fovy", "cam_resolution", "cam_intrinsic", "cam_mode"]:
        if hasattr(sim.model, attr):
            v = getattr(sim.model, attr)
            print(f"  sim.model.{attr} = {v if not hasattr(v, 'shape') else v.shape}")

    env.close()


if __name__ == "__main__":
    main()
