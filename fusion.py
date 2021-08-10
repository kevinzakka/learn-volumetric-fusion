"""Volumetric TSDF integration.

[1]: Curless and Levoy, 1996: A Volumetric Method for Building Complex Models from Range Images.
[2]: Newcombe et al., 2011: KinectFusion: Real-Time Dense Surface Mapping and Tracking.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import numpy as np
from numpy.linalg.linalg import norm
from skimage.measure import marching_cubes as skimage_marching_cubes

# Global constants.
BYTES_TO_GIGABYTES = 1.0 / (1 << 30)
SHORTMAX = np.iinfo(np.int16).max
DIVSHORTMAX = 1.0 / SHORTMAX
MAX_WEIGHT = 128

# Global types.
Int3 = Tuple[int, int, int]
Mesh = Tuple[np.ndarray, ...]
PyramidArray = Dict[int, np.ndarray]

from ipdb import set_trace

from debug import visualize_normal_map, visualize_pc_o3d


@dataclass(frozen=True)
class GlobalConfig:
    """The global configuration."""

    volume_size: Int3 = (512, 512, 512)
    """The overall x, y, z dimensions of the volume."""

    voxel_scale: float = 0.02
    """The volume discretization (in m), i.e. how much space each voxel occupies."""

    truncation_distance: float = 0.1
    """The truncation distance (in m) for updating the TSDF volume."""

    depth_cutoff_distance: float = 4.0
    """The distance (in m) after which the depth is set to 0 in the depth frames."""

    bilateral_kernel_size: int = 5
    bilateral_color_sigma: float = 1.0
    bilateral_spatial_sigma: float = 0.3
    """Bilateral filter parameters."""

    num_levels: int = 3
    """Pyramid levels."""

    icp_distance_threshold: float = 0.01
    icp_angle_threshold: float = 20.0
    icp_iterations: Tuple[int, int, int] = (10, 5, 4)
    """ICP parameters."""

    def __post_init__(self):
        assert len(self.volume_size) == 3
        assert len(self.icp_iterations) == self.num_levels


@dataclass(frozen=True)
class Intrinsic:
    """Pinhole camera parameters."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @property
    def resolution(self) -> Tuple[int, int]:
        return (self.width, self.height)

    @property
    def matrix(self) -> np.ndarray:
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    def level(self, level: int) -> Intrinsic:
        """Returns intrinsic parameters at a specified pyramid level."""

        scale_factor = math.pow(0.5, level)
        return Intrinsic(
            self.width >> level,
            self.height >> level,
            self.fx * scale_factor,
            self.fy * scale_factor,
            (self.cx + 0.5) * scale_factor - 0.5,
            (self.cy + 0.5) * scale_factor - 0.5,
        )


@dataclass
class FrameObservation:
    """Raw and pre-processed frame data obtained from an RGB-D sensor."""

    color_pyramid: PyramidArray
    depth_pyramid: PyramidArray
    smoothed_depth_pyramid: PyramidArray
    vertex_pyramid: PyramidArray
    normal_pyramid: PyramidArray

    @staticmethod
    def initialize(camera_params: Intrinsic, config: GlobalConfig) -> FrameObservation:
        color_pyramid = dict()
        depth_pyramid, smoothed_depth_pyramid = dict(), dict()
        vertex_pyramid, normal_pyramid = dict(), dict()

        for level in range(config.num_levels):
            width, height = camera_params.level(level).resolution
            color_pyramid[level] = np.empty((height, width, 3), dtype=np.uint8)
            depth_pyramid[level] = np.empty((height, width), dtype=np.float32)
            smoothed_depth_pyramid[level] = np.empty((height, width), dtype=np.float32)
            vertex_pyramid[level] = np.empty((height, width, 3), dtype=np.float32)
            normal_pyramid[level] = np.empty((height, width, 3), dtype=np.float32)

        return FrameObservation(
            color_pyramid,
            depth_pyramid,
            smoothed_depth_pyramid,
            vertex_pyramid,
            normal_pyramid,
        )


@dataclass
class FrameRender:
    """Synthetic frame data rendered via raycasting the internal TSDF volume."""

    color_pyramid: PyramidArray
    vertex_pyramid: PyramidArray
    normal_pyramid: PyramidArray

    @staticmethod
    def initialize(camera_params: Intrinsic, config: GlobalConfig) -> FrameRender:
        color_pyramid, vertex_pyramid, normal_pyramid = dict(), dict(), dict()

        for level in range(config.num_levels):
            width, height = camera_params.level(level).resolution
            color_pyramid[level] = np.empty((height, width, 3), dtype=np.uint8)
            vertex_pyramid[level] = np.empty((height, width, 3), dtype=np.float32)
            normal_pyramid[level] = np.empty((height, width, 3), dtype=np.float32)

        return FrameRender(color_pyramid, vertex_pyramid, normal_pyramid)


@dataclass(frozen=False)
class TSDFVolume:
    """A TSDF volume with a uniform voxel grid."""

    camera_params: Intrinsic
    """The camera parameters."""

    config: GlobalConfig
    """The config values."""

    tsdf_volume: np.ndarray
    """The global volume in which depth frames are fused."""

    weight_volume: np.ndarray
    """Holds the weights of the moving average."""

    color_volume: np.ndarray
    """The global color volume in which color frames are fused."""

    voxel_coords: np.ndarray
    """Voxel grid coordinates."""

    world_pts: np.ndarray
    """Voxel coordinates in base camera frame."""

    frame_render: FrameRender
    """View of the implicit surface as raycast from the TSDF volume."""

    current_pose: np.ndarray
    """Current estimated pose of the camera with respect to the world frame."""

    @staticmethod
    def initialize(camera_params: Intrinsic, config: GlobalConfig) -> TSDFVolume:
        # Allocate volumes.
        tsdf_volume = np.ones(config.volume_size, dtype=np.int16)
        weight_volume = np.zeros(config.volume_size, dtype=np.int16)
        color_volume = np.zeros(config.volume_size + (3,), dtype=np.uint8)

        # Create voxel grid indices.
        voxel_coords = np.indices(config.volume_size).reshape(3, -1).T

        # Convert voxel grid indices to voxel coordinates in world frame.
        world_pts = (voxel_coords.astype(np.float32) + 0.5) * config.voxel_scale

        # Initialize first camera pose estimate in the middle of the volume.
        current_pose = np.eye(4)
        for i in range(3):
            current_pose[i, 3] = 0.5 * config.volume_size[i] * config.voxel_scale

        frame_render = FrameRender.initialize(camera_params, config)

        return TSDFVolume(
            camera_params,
            config,
            tsdf_volume,
            weight_volume,
            color_volume,
            voxel_coords,
            world_pts,
            frame_render,
            current_pose,
        )

    def integrate(
        self,
        color_im: np.ndarray,
        depth_im: np.ndarray,
        frame_id: int,
        pose,
    ):
        """Integrate an RGB-D frame into the TSDF volume."""

        # Sanity check shapes.
        assert color_im.shape[:2] == depth_im.shape[:2]
        if (
            depth_im.shape[0] != self.camera_params.height
            or depth_im.shape[1] != self.camera_params.width
        ):
            raise ValueError("Depth image size does not match camera parameters.")

        # 1. Surface measurement.
        tic = time.time()
        frame = FrameObservation.initialize(self.camera_params, self.config)
        surface_measurement(
            color_im,
            depth_im,
            frame,
            self.camera_params,
            self.config.num_levels,
            self.config.bilateral_kernel_size,
            self.config.bilateral_color_sigma,
            self.config.bilateral_spatial_sigma,
            self.config.depth_cutoff_distance,
            pose,
        )
        print(f"Surface measurement completed in {time.time() - tic}s")

        # # 2. Pose estimation.
        # tic = time.time()
        # if frame_id > 0:
        #     pose_estimation(
        #         self.current_pose,
        #         frame,
        #         self.camera_params,
        #         self.config.num_levels,
        #         self.config.icp_distance_threshold,
        #         self.config.icp_angle_threshold,
        #         self.config.icp_iterations,
        #     )
        # print(f"ICP completed in {time.time() - tic}s")

        # 3. Surface reconstruction.
        tic = time.time()
        surface_reconstruction(
            frame.color_pyramid[0],
            frame.depth_pyramid[0],
            # Note: Converts camera2world into world2camera pose.
            se3_inverse(self.current_pose),
            self.camera_params,
            self.config.truncation_distance,
            self.voxel_coords,
            self.world_pts,
            self.tsdf_volume,
            self.weight_volume,
            self.color_volume,
        )
        print(f"Surface reconstruction completed in {time.time() - tic}s")

        # # 4. Surface prediction.
        # tic = time.time()
        # for level in range(self.config.num_levels):
        #     surface_prediction(
        #         self.tsdf_volume,
        #         self.config.volume_size,
        #         self.config.voxel_scale,
        #         self.camera_params.level(level),
        #         self.config.truncation_distance,
        #         self.current_pose,
        #         self.frame_render.color_pyramid[level],
        #         self.frame_render.vertex_pyramid[level],
        #         self.frame_render.normal_pyramid[level],
        #     )
        # print(f"Surface prediction completed in {time.time() - tic}s")

    def extract_mesh(self) -> Mesh:
        return marching_cubes(
            self.tsdf_volume,
            self.color_volume,
            self.config.voxel_scale,
        )


# ======================================================= #
# Kinect fusion methods.
# ======================================================= #


def surface_measurement(
    color_im: np.ndarray,
    depth_im: np.ndarray,
    frame: FrameObservation,
    intr: Intrinsic,
    num_levels: int,
    bilateral_kernel_size: int,
    bilateral_color_sigma: float,
    bilateral_spatial_sigma: float,
    depth_cutoff_distance: float,
    pose_gt: np.ndarray,
):
    """Generate dense vertex and normal map pyramids from raw RGB-D frames."""
    # Build pyramids.
    frame.color_pyramid[0] = color_im
    frame.depth_pyramid[0] = depth_im
    for level in range(1, num_levels):
        # In the paper, depth values are used in the Gaussian pyramid average only if
        # they are within 3σr of the central pixel to ensure smoothing does not occur
        # over depth boundaries. We don't have control over this when we use opencv's
        # pyrDown method.
        cv2.pyrDown(frame.depth_pyramid[level - 1], dst=frame.depth_pyramid[level])
        # Creating a color pyramid for levels above 0 isn't really necessary.
        cv2.pyrDown(frame.color_pyramid[level - 1], dst=frame.color_pyramid[level])

    # Bilaterally filter depth pyramids.
    for level in range(num_levels):
        cv2.bilateralFilter(
            src=frame.depth_pyramid[level],
            d=bilateral_kernel_size,
            sigmaColor=bilateral_color_sigma,
            sigmaSpace=bilateral_spatial_sigma,
            borderType=cv2.BORDER_DEFAULT,
            dst=frame.smoothed_depth_pyramid[level],
        )

    # Compute vertex and normal maps.
    for level in range(num_levels):
        tic = time.time()
        frame.vertex_pyramid[level] = compute_vertex_map(
            frame.smoothed_depth_pyramid[level],
            depth_cutoff_distance,
            intr.level(level),
        )
        print(f"Vertex map computation took: {time.time() - tic}s")
        tic = time.time()
        frame.normal_pyramid[level] = compute_normals_map(frame.vertex_pyramid[level])
        print(f"Normal map computation took: {time.time() - tic}s")

        # # Visualize pointcloud and normals.
        # if level == 0:
        #     visualize_normal_map(frame.normal_pyramid[level], se3_inverse(pose_gt))
        #     visualize_pc_o3d(
        #         frame.vertex_pyramid[level].reshape(-1, 3),
        #         frame.color_pyramid[level].reshape(-1, 3).copy() / 255.0,
        #         frame.normal_pyramid[level].reshape(-1, 3),
        #         downsample=0.03,
        #     )


def pose_estimation(
    current_pose: np.ndarray,
    frame: FrameObservation,
    intr: Intrinsic,
    num_levels: int,
    icp_distance_threshold: float,
    icp_angle_threshold: float,
    icp_iterations: Tuple[int, int, int],
):
    """Camera pose estimation.

    Uses multi-scale ICP alignment between the predicted surface and current sensor
    measurement to estimate the sensor pose.

    Assumptions:
        High frame-rate -> small motion from one frame to the next. Can use fast
        projective data association algorithm to obtain correspondence and the
        point-plane metric for pose optimization.

    References:
        Blais and Levine, 1993: Registering Multiview Range Data to Create 3D
            Computer Objects. (Projection-based matching).
        Chen and Medioni, 1991: Object Modeling by Registration of Multiple Range
            Images. (Point-to-plane error metric)
    """
    pass


def surface_reconstruction(
    color_im: np.ndarray,
    depth_im: np.ndarray,
    pose: np.ndarray,
    intr: Intrinsic,
    truncation_distance: float,
    voxel_coords: np.ndarray,
    world_pts: np.ndarray,
    tsdf_volume: np.ndarray,
    weight_volume: np.ndarray,
    color_volume: np.ndarray,
):
    """Fuse the surface measurement into the global TSDF volume."""
    # Convert voxel grid coordinates in world frame to coordinates in camera frame.
    cam_pts = apply_se3(world_pts, pose)

    # Convert camera coordinates to camera pixels.
    pix_x, pix_y, pix_z = cam_pts.T
    with np.errstate(divide="ignore"):
        pix_x /= pix_z
        pix_x *= intr.fx
        pix_x += intr.cx
        pix_x = np.round(pix_x).astype(np.int32)
        pix_y /= pix_z
        pix_y *= intr.fy
        pix_y += intr.cy
        pix_y = np.round(pix_y).astype(np.int32)
    pix_x = np.nan_to_num(pix_x, copy=False, nan=0)
    pix_y = np.nan_to_num(pix_y, copy=False, nan=0)

    # Eliminate pixels outside view frustum.
    mask = pix_z > 0
    mask &= pix_x >= 0
    mask &= pix_x < intr.width
    mask &= pix_y >= 0
    mask &= pix_y < intr.height
    depth_val = np.zeros_like(pix_x, dtype=np.float32)
    depth_val[mask] = depth_im[pix_y[mask], pix_x[mask]]

    # Compute sdf and truncate -> tsdf.
    tsdf = depth_val - pix_z
    valid_pts = depth_val > 0
    valid_pts &= tsdf >= -truncation_distance
    tsdf[valid_pts] /= truncation_distance
    tsdf[valid_pts] = np.minimum(1.0, tsdf[valid_pts])

    obs_weight = 1
    vx, vy, vz = voxel_coords[valid_pts].T

    # Integrate tsdf.
    tsdf_new = tsdf[valid_pts]
    tsdf_old = tsdf_volume[vx, vy, vz].astype(np.float32) * DIVSHORTMAX
    w_old = weight_volume[vx, vy, vz]
    w_new = w_old + obs_weight
    tsdf_vol_new = (w_old * tsdf_old + obs_weight * tsdf_new) / w_new
    tsdf_volume[vx, vy, vz] = np.clip(
        (tsdf_vol_new * SHORTMAX).astype(np.int16),
        a_min=-SHORTMAX,
        a_max=SHORTMAX,
    )
    weight_volume[vx, vy, vz] = np.minimum(w_new, MAX_WEIGHT)

    # Integrate color.
    for i in range(3):
        color_old = color_volume[vx, vy, vz, i]
        color_new = color_im[pix_y[valid_pts], pix_x[valid_pts], i]
        color_volume[vx, vy, vz, i] = (
            (w_old * color_old + obs_weight * color_new) / w_new
        ).astype(np.uint8)


def surface_prediction(
    tsdf_volume: np.ndarray,
    volume_size: Int3,
    voxel_scale: float,
    intr: Intrinsic,
    truncation_distance: float,
    current_pose: np.ndarray,
    color_pyramid: np.ndarray,
    vertex_pyramid: np.ndarray,
    normal_pyramid: np.ndarray,
):
    """Raycasts the TSDF volume from the current estimated pose.

    This generates a view of the implicit surface at the current pose in the form of
    vertex and normal maps which are used in the subsequence ICP iteration.

    Ideally, this is implemented on a GPU where all pixel rays are run in parallel.

    References:
        Parker et al, 1998: Interactive Ray Tracing for Isosurface Rendering.
    """

    def trilerp():
        """Trilinearly interpolate SDF value."""
        pass

    # Compute a direction vector from the camera center through each pixel in world
    # coordinates.
    xy = np.indices(intr.resolution, dtype=np.int16).reshape(2, -1)
    xyz = np.vstack([xy, np.ones((1, xy.shape[-1]))])
    rays = (xyz.T @ np.linalg.inv(intr.matrix) @ current_pose[:3, :3]).T
    rays /= np.linalg.norm(rays, axis=0, keepdims=True)

    # Calculate ray length.
    volume_range = np.asarray(volume_size) * voxel_scale


# ======================================================= #
# Helper methods.
# ======================================================= #


def se3_inverse(se3: np.ndarray) -> np.ndarray:
    """Compute the inverse of an SE(3) transform."""
    inv_se3 = np.empty_like(se3)
    tr = se3[:3, :3].T
    inv_se3[:3, :3] = tr
    inv_se3[:3, 3] = -tr @ se3[:3, 3]
    inv_se3[3, :] = [0, 0, 0, 1.0]
    return inv_se3


def apply_se3(pts: np.ndarray, se3: np.ndarray) -> np.ndarray:
    """Apply an SE(3) transform to a pointcloud."""
    # Turns out doing (se3[:3, :3] @ pts.T).T + se3[:3, 3] is slower because it makes
    # a copy. This inplace version is roughly 2x faster!
    o = (se3[:3, :3] @ pts.T).T
    o += se3[:3, 3]
    return o


def marching_cubes(
    tsdf_volume: np.ndarray,
    color_volume: np.ndarray,
    voxel_scale: float,
) -> Mesh:
    """Extract a surface mesh from a TSDF volume using Marching Cubes."""
    tsdf_volume = tsdf_volume.astype(np.float32) * DIVSHORTMAX
    mask = (tsdf_volume > -0.5) & (tsdf_volume < 0.5)
    verts, faces, norms, _ = skimage_marching_cubes(tsdf_volume, mask=mask, level=0)
    vix, viy, viz = np.round(verts).astype(np.int16).T
    verts = verts * voxel_scale
    colors = color_volume[vix, viy, viz]
    return verts, faces, norms, colors


def compute_vertex_map(
    depth_im: np.ndarray,
    depth_cutoff_distance: float,
    intr: Intrinsic,
) -> np.ndarray:
    """Back-project a depth image into a 3D vertex map."""
    cc, rr = np.meshgrid(np.arange(intr.width), np.arange(intr.height), sparse=True)
    valid = (depth_im > 0) & (depth_im < depth_cutoff_distance)
    z = np.where(valid, depth_im, 0.0)
    x = np.where(valid, z * (cc - intr.cx).astype(np.float32) / intr.fx, 0.0)
    y = np.where(valid, z * (rr - intr.cy).astype(np.float32) / intr.fy, 0.0)
    return np.stack([x, y, z], axis=-1)  # (H, W, 3)


# TODO(kevin): Consider casting to float64 to increase accuracy.
def compute_normals_map(vmap: np.ndarray) -> np.ndarray:
    """Compute normal vectors from a vertex map.

    Assumes neighbouring pixels in the vertex map correspond to neighboring vertices
    in the scene. Concretely, given a vertex map V of shape (H, W, 3), this function
    returns a dense normal map N of same shape (H, W, 3) such that:

        N[y, x] = (V[y - 1, x] - V[y + 1, x]) x (V[y, x - 1] - V[y, x + 1])
        N[y, x] /= L2_norm(N[y, x])
    """
    assert vmap.ndim == 3 and vmap.shape[-1] == 3
    height, width = vmap.shape[:2]
    upper = vmap[0 : height - 2, 1 : width - 1]
    lower = vmap[2:height, 1 : width - 1]
    vertical = upper - lower  # This vector points up.
    left = vmap[1 : height - 1, 0 : width - 2]
    right = vmap[1 : height - 1, 2:width]
    horizontal = left - right  # This vector points left.
    cross = np.cross(vertical, horizontal)  # Right-hand rule.
    nmap = np.zeros_like(vmap)
    nmap[1 : height - 1, 1 : width - 1] = cross
    nmap[1 : height - 1, 1 : width - 1] /= (
        np.linalg.norm(nmap[1 : height - 1, 1 : width - 1], axis=-1, keepdims=True)
        + 1e-10
    )
    # NaN-ify normals where the original vertex depth value was invalid and thus set to
    # 0 in `compute_vertex_map`.
    nmap[vmap[..., -1] == 0.0] = np.nan
    return nmap
