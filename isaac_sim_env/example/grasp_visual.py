import numpy as np
import open3d as o3d

point_cloud = np.load("./point_cloud.npy")
normals = np.load("./normals.npy")

grasp_point_indices = np.load("./grasp_point_indices.npy")

grasp_point_indices = list(set(grasp_point_indices))

sample_point_cloud = point_cloud[grasp_point_indices]
sample_normals = normals[grasp_point_indices]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_cloud.astype(np.float64))

pcd.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
pcd.paint_uniform_color([0.1, 0.6, 0.9])
o3d.visualization.draw_geometries([pcd], point_show_normal=True)
