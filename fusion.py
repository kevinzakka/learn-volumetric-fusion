from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Union

import numpy as np
from ipdb import set_trace
from skimage.measure import marching_cubes

# Global constants.
BYTES_TO_GIGABYTES = 1.0 / (1 << 30)
SHORTMAX = np.iinfo(np.int16).max
DIVSHORTMAX = 0.0000305185
MAX_WEIGHT = 128

# Global types.
Int3 = Tuple[int, int, int]


@dataclass
class GlobalConfig:
    """The global configuration."""

    volume_size: Int3 = (512, 512, 512)
    """The overall size of the volume in mm."""

    voxel_scale: float = 20.0
    """The volume discretization in mm."""

    truncation_distance: float = 100.0
    """The truncation distance (in mm) for updating the TSDF volume."""

    depth_cutoff_distance: float = 4000.0
    """The distance (in mm) after which the depth is set to 0 in the depth frames."""


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
    def matrix(self) -> np.ndarray:
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    @classmethod
    def from_file(
        cls, filename: Union[str, Path], width: int, height: int
    ) -> Intrinsic:
        mat = np.loadtxt(filename, delimiter=" ", dtype=np.float32)
        return cls(width, height, mat[0, 0], mat[1, 1], mat[0, 2], mat[1, 2])


class UniformTSDFVolume:
    """A TSDF volume with a uniform voxel grid [1].

    References:
        [1]: Curless and Levoy, 1996.
    """

    def __init__(self, volume_size: Int3, voxel_scale: float):
        """Constructor.

        Args:
            volume_size: The dimensions of the 3D volume in mm. Will be allocated on
                the compute device so should be decreased for low RAM.
            voxel_scale: Controls the resolution of the volume. Lowering this value
                makes a high-resolution TSDF volume, but will make the integration
                susceptible to depth noise.
        """
        self.volume_size = volume_size
        self.voxel_scale = voxel_scale

        self._tsdf_volume = None
        self._weight_volume = None
        self._color_volume = None

    def reset(self):
        self._tsdf_volume = np.zeros(self.volume_size, dtype=np.int16)
        self._weight_volume = np.zeros(self.volume_size, dtype=np.int16)
        self._color_volume = np.zeros(self.volume_size + (3,), dtype=np.uint8)

        size = 0
        for volume in [self._tsdf_volume, self._weight_volume, self._color_volume]:
            size += volume.nbytes
        size *= BYTES_TO_GIGABYTES
        print(f"TSDF volume: {size:.4} GB")

    @property
    def tsdf_volume(self) -> np.ndarray:
        return self._tsdf_volume

    @property
    def weight_volume(self) -> np.ndarray:
        return self._weight_volume

    @property
    def color_volume(self) -> np.ndarray:
        return self._color_volume

    @property
    def center_point(self) -> np.ndarray:
        return np.asarray(self.volume_size, dtype=np.float32) * 0.5 * self.voxel_scale


class TSDFFusion:
    """Volumetric TSDF Fusion of RGB-D Images."""

    def __init__(self, camera_params: Intrinsic, config: GlobalConfig):
        """Set up the internal volume and camera parameters."""

        self._camera_params = camera_params
        self._config = config
        self._volume = UniformTSDFVolume(config.volume_size, config.voxel_scale)

        self.reset()

    def reset(self):
        self._frame_id = 0
        self._volume.reset()

    def integrate(
        self,
        color_im,
        depth_im,
        pose,
    ):
        """Integrate an RGB-D frame into the TSDF volume."""

        # Sanity check shapes.
        assert pose.shape == (4, 4)
        assert color_im.shape[:2] == depth_im.shape[:2]
        if (
            depth_im.shape[0] != self._camera_params.height
            or depth_im.shape[1] != self._camera_params.width
        ):
            raise ValueError(f"Depth image size does not match camera parameters.")

        # Truncate depth values >= than the cutoff.
        depth_im[depth_im >= self._config.depth_cutoff_distance] = 0.0

        # Fuse the RGB-D frame into the volume.
        surface_reconstruction(
            depth_im,
            color_im,
            self._volume,
            self._camera_params,
            self._config.truncation_distance,
            se3_inverse(pose),
        )

        self._frame_id += 1

    def extract_pointcloud(self):
        return extract_points(self._volume)

    def extract_mesh(self):
        return extract_mesh(self._volume)

    @property
    def frame_id(self) -> int:
        return self._frame_id


def se3_inverse(pose: np.ndarray) -> np.ndarray:
    """Compute the inverse of an SE(3) transform."""
    inv_pose = np.empty_like(pose)
    tr = pose[:3, :3].T
    inv_pose[:3, :3] = tr
    inv_pose[:3, 3] = -tr @ pose[:3, 3]
    inv_pose[3, :] = [0, 0, 0, 1.0]
    return inv_pose


def surface_reconstruction(
    depth_im: np.ndarray,
    color_im: np.ndarray,
    volume: UniformTSDFVolume,
    intr: Intrinsic,
    truncation_distance: float,
    pose: np.ndarray,
):
    """Integration of surface measurements into a global volume."""

    # Create voxel grid coordinates.
    # tic = time.time()
    vox_coords = np.indices(volume.volume_size).reshape(3, -1).T
    # print(f"voxel creation: {time.time() - tic}s")

    # Convert voxel center from grid coordinates to base frame camera coordinates.
    # tic = time.time()
    world_pts = ((vox_coords.astype(np.float32) + 0.5) * volume.voxel_scale) - volume.center_point
    # print(f"world coords: {time.time() - tic}s")

    # Convert from base frame camera coordinates to current frame camera coordinates
    # tic = time.time()
    cam_pts = rigid_transform(world_pts, pose)
    # print(f"rigid transform: {time.time() - tic}s")

    # Remove points with negative z.
    # tic = time.time()
    mask = cam_pts[:, 2] > 0
    # print(f"invalid z mask: {time.time() - tic}s")

    # Convert to camera pixels.
    # tic = time.time()
    pix_x = np.full_like(cam_pts[:, 0], -1, dtype=np.int32)
    pix_y = np.full_like(cam_pts[:, 0], -1, dtype=np.int32)
    pix_x[mask] = np.around(
        cam_pts[mask, 0] / cam_pts[mask, 2] * intr.fx + intr.cx
    ).astype(np.int32)
    pix_y[mask] = np.around(
        cam_pts[mask, 1] / cam_pts[mask, 2] * intr.fy + intr.cy
    ).astype(np.int32)
    # print(f"projection: {time.time() - tic}s")

    # Eliminate pixels outside view frustum.
    # tic = time.time()
    mask &= (pix_x >= 0) & (pix_x < intr.width) & (pix_y >= 0) & (pix_y < intr.height)
    # print(f"invalid pix mask: {time.time() - tic}s")

    # tic = time.time()
    depth_val = np.zeros_like(pix_x, dtype=np.float32)
    depth_val[mask] = depth_im[pix_y[mask], pix_x[mask]]
    # print(f"index depth values: {time.time() - tic}s")

    # tic = time.time()
    mask &= depth_val > 0
    # print(f"positive only depth mask: {time.time() - tic}s")

    # tic = time.time()
    depth_diff = depth_val - cam_pts[:, 2]
    valid_pts = (depth_val > 0) & (depth_diff >= -truncation_distance)
    dist = np.minimum(1.0, depth_diff / truncation_distance)
    valid_vox_x = vox_coords[valid_pts, 0]
    valid_vox_y = vox_coords[valid_pts, 1]
    valid_vox_z = vox_coords[valid_pts, 2]
    valid_dist = dist[valid_pts]

    tsdf_vals = volume.tsdf_volume[valid_vox_x, valid_vox_y, valid_vox_z]
    tsdf_vals = tsdf_vals.astype(np.float32) * DIVSHORTMAX

    w_old = volume.weight_volume[valid_vox_x, valid_vox_y, valid_vox_z]
    obs_weight = 1

    tsdf_vol_new = (w_old * tsdf_vals + obs_weight * valid_dist) / (w_old + obs_weight)
    tsdf_vol_new = np.clip(
        (tsdf_vol_new * SHORTMAX).astype(np.int16),
        a_min=-SHORTMAX,
        a_max=SHORTMAX,
    )
    volume.tsdf_volume[valid_vox_x, valid_vox_y, valid_vox_z] = tsdf_vol_new

    # From paper: Moreover, by truncating the updated weight over some value MAX_WEIGHT,
    # a moving average surface reconstruction can be obtained enabling reconstruction in
    # scenes with dynamic object motion.
    w_new = np.minimum(w_old + obs_weight, MAX_WEIGHT)
    volume.weight_volume[valid_vox_x, valid_vox_y, valid_vox_z] = w_new
    # print(f"tsdf integration: {time.time() - tic}s")

    # tic = time.time()
    for i in range(3):
        volume.color_volume[valid_vox_x, valid_vox_y, valid_vox_z, i] = (
            (w_old * volume.color_volume[valid_vox_x, valid_vox_y, valid_vox_z, i] \
                + obs_weight * color_im[pix_y[valid_pts], pix_x[valid_pts], i]) \
                    / (w_old + obs_weight)).astype(np.uint8)
    # print(f"color integration: {time.time() - tic}s")


    # color_const = 256 * 256
    # color_im = color_im.astype(np.float32)
    # color_im = np.floor(color_im[..., 2] * color_const + color_im[..., 1] * 256 + color_im[..., 0])
    # old_color = volume.color_volume[valid_vox_x, valid_vox_y, valid_vox_z]
    # old_b = np.floor(old_color / color_const)
    # old_g = np.floor((old_color - old_b * color_const) / 256)
    # old_r = old_color - old_b * color_const - old_g * 256
    # new_color = color_im[pix_y[valid_pts], pix_x[valid_pts]]
    # new_b = np.floor(new_color / color_const)
    # new_g = np.floor((new_color - new_b * color_const) / 256)
    # new_r = new_color - new_b * color_const - new_g * 256
    # new_b = np.minimum(255.0, np.round((w_old * old_b + obs_weight * new_b) / w_new))
    # new_g = np.minimum(255.0, np.round((w_old * old_g + obs_weight * new_g) / w_new))
    # new_r = np.minimum(255.0, np.round((w_old * old_r + obs_weight * new_r) / w_new))
    # volume.color_volume[valid_vox_x, valid_vox_y, valid_vox_z] = (
    #     new_b * color_const + new_g * 256 + new_r
    # )
    # print(f"color integration: {time.time() - tic}s")

    # print(f"total: {time.time() - start}s")


def extract_points(volume: UniformTSDFVolume):
    """Extract a pointcloud from a TSDF volume."""
    raise NotImplementedError


def extract_mesh(volume: UniformTSDFVolume):
    """Extract a surface mesh from a TSDF volume using the Marching Cubes algorithm."""
    tsdf_volume = volume.tsdf_volume.astype(np.float32) * DIVSHORTMAX
    mask = (tsdf_volume > -0.5) & (tsdf_volume < 0.5)
    verts, faces, norms, _ = marching_cubes(tsdf_volume, mask=mask, level=0)
    verts_ind = np.round(verts).astype(int)
    verts = verts * volume.voxel_scale - volume.center_point
    colors = volume.color_volume[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
    return verts, faces, norms, colors


def rigid_transform(xyz, transform):
    return (transform[:3, :3] @ xyz.T).T + transform[:3, 3]
