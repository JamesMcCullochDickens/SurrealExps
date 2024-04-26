#  See:
#  https://github.com/gulvarol/surreal/blob/master/datageneration/misc/3Dto2D/getExtrinsicBlender.m

import numpy as np
import open3d as o3d

res_x_px = 320.0
res_y_px = 240.0
f_mm = 60.0
sensor_w_mm = 32.0
sensor_h_mm = sensor_w_mm * res_y_px / res_x_px

scale = 1.0
skew = 0.0
pixel_aspect_ratio = 1.0

fx_px = f_mm * res_x_px * scale / sensor_w_mm
fy_px = f_mm * res_y_px * scale * pixel_aspect_ratio / sensor_h_mm

cx = res_x_px * scale / 2.0
cy = res_y_px * scale / 2.0


bad_depth_val = 10000000000.0


def depth_map_to_pc(depth_map: np.ndarray) -> np.ndarray:
      global cx, cy, fx_px, fy_px
      h, w = depth_map.shape

      sorted_unique_values = np.sort(np.unique(depth_map).reshape(-1))[::-1]
      max_valid_d = sorted_unique_values[1]
      depth_map = np.clip(depth_map, a_min=0.0, a_max=max_valid_d)
      y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
      x_val = ((x - cx) / fx_px) * depth_map
      y_val = ((y - cy) / fy_px) * depth_map

      pc = np.empty((h, w, 3), dtype=float)
      pc[..., 0] = x_val
      pc[..., 1] = y_val
      pc[..., 2] = depth_map

      pc[:, :, 1] *= -1
      return pc


def display_pc(pc, depth_filter_val=None):
      pc_ = pc.reshape(-1, 3)
      if depth_filter_val is not None:
            mask = pc_[:, 2] != depth_filter_val
            pc_ = pc_[mask]
      pcd = o3d.geometry.PointCloud()
      pcd.points = o3d.utility.Vector3dVector(pc_)
      coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
      o3d.visualization.draw_geometries([pcd, coordinate_frame])


"""
m1 = m_proc.get_depth_ims(m_proc.load_mat(m_proc.sample_depth_mat_fp))[0]
unique_vals = np.unique(m1)
pc = depth_map_to_pc(m1)
display_pc(pc, np.max(pc[:, :, -1]))
"""