import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def visualize_pc_o3d(pts, clrs, normals=None, downsample=None):
    pts = pts.copy().astype(np.float64)
    clrs = clrs.copy().astype(np.float64)
    o3d_pc = [o3d.geometry.PointCloud()]
    o3d_pc[0].points = o3d.utility.Vector3dVector(pts)
    o3d_pc[0].colors = o3d.utility.Vector3dVector(clrs)
    if normals is not None:
        normals = normals.copy().astype(np.float64)
        o3d_pc[0].normals = o3d.utility.Vector3dVector(normals)
    if downsample is not None:
        o3d_pc[0] = o3d_pc[0].voxel_down_sample(voxel_size=downsample)
    o3d_pc[0].transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    o3d_pc.append(
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    )
    o3d.visualization.draw_geometries(o3d_pc)


def visualize_normal_map(nmap: np.ndarray, world2cam: np.ndarray):
    h, w = nmap.shape[:2]
    nmap_flat = nmap.reshape(-1, 3)
    nmap_cam_flat = (world2cam[:3, :3] @ nmap_flat.T).T
    nmap_cam_flat += world2cam[:3, 3]
    nmap_cam = nmap_cam_flat.reshape(h, w, 3)
    img = nmap_cam * 0.5 + 0.5  # Shifts from [-1, 1] to [0, 1].
    img = (255.0 * img).astype(np.uint8)
    plt.imshow(img)
    plt.axis("off")
    plt.show()
