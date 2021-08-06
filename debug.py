import numpy as np
import open3d as o3d


def visualize_pc_o3d(pts, clrs, normals=None):
    pts = pts.copy().astype(np.float64)
    clrs = clrs.copy().astype(np.float64)
    o3d_pc = [o3d.geometry.PointCloud()]
    o3d_pc[0].points = o3d.utility.Vector3dVector(pts)
    o3d_pc[0].colors = o3d.utility.Vector3dVector(clrs)
    if normals is not None:
        normals = normals.copy().astype(np.float64)
        o3d_pc[0].normals = o3d.utility.Vector3dVector(normals)
    o3d_pc.append(
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    )
    o3d.visualization.draw_geometries(o3d_pc)
