from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Tuple, Union

import numpy as np
from skimage.measure import marching_cubes

# Global constants.
BYTES_TO_GIGABYTES = 1.0 / (1 << 30)
SHORTMAX = np.iinfo(np.int16).max
DIVSHORTMAX = 0.0000305185
MAX_WEIGHT = 128

# Global types.
Int3 = Tuple[int, int, int]
Mesh = Tuple[np.ndarray, ...]


@dataclass(frozen=True)
class GlobalConfig:
    """The global configuration."""

    volume_size: Int3 = (512, 512, 512)
    """The overall size of the volume in meters (m)."""

    voxel_scale: float = 0.02
    """The volume discretization (in m)."""

    truncation_distance: float = 20.0
    """The truncation distance (in m) for updating the TSDF volume."""

    depth_cutoff_distance: float = 4.0
    """The distance (in m) after which the depth is set to 0 in the depth frames."""


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


@dataclass(frozen=True)
class UniformTSDFVolume:
    """A TSDF volume with a uniform voxel grid [1].

    References:
        [1]: Curless and Levoy, 1996.
    """

    volume_size: Int3
    """The dimensions of the 3D volume."""

    voxel_scale: float
    """Controls the resolution of the volume."""

    tsdf_volume: np.ndarray
    weight_volume: np.ndarray
    color_volume: np.ndarray

    @staticmethod
    def initialize(config: GlobalConfig) -> UniformTSDFVolume:
        tsdf_volume = np.ones(config.volume_size, dtype=np.int16)
        weight_volume = np.zeros(config.volume_size, dtype=np.int16)
        color_volume = np.zeros(config.volume_size + (3,), dtype=np.uint8)

        return UniformTSDFVolume(
            config.volume_size,
            config.voxel_scale,
            tsdf_volume,
            weight_volume,
            color_volume,
        )

    @property
    def center_point(self) -> np.ndarray:
        return -0.5 * self.voxel_scale * np.asarray(self.volume_size, dtype=np.float32)

    def integrate(self, color_im, depth_im, pose, intr, truncation_distance) -> UniformTSDFVolume:
        """Integration of surface measurements into a global volume."""

        tsdf_volume_new = np.copy(self.tsdf_volume)
        weight_volume_new = np.copy(self.weight_volume)
        color_volume_new = np.copy(self.color_volume)

        # Create voxel grid coordinates.
        vox_coords = np.indices(self.volume_size).reshape(3, -1).T

        # Convert voxel center from grid coordinates to base frame camera coordinates.
        world_pts = (
            (vox_coords.astype(np.float32) + 0.5) * self.voxel_scale
        ) + self.center_point

        # Convert from base frame camera coordinates to current frame camera coordinates.
        cam_pts = se3_transform(world_pts, pose)

        # Remove points with negative z.
        mask = cam_pts[:, 2] > 0

        # Convert to camera pixels.
        pix_x = np.full_like(cam_pts[:, 0], -1, dtype=np.int32)
        pix_y = np.full_like(cam_pts[:, 0], -1, dtype=np.int32)
        pix_x[mask] = np.around(
            cam_pts[mask, 0] / cam_pts[mask, 2] * intr.fx + intr.cx
        ).astype(np.int32)
        pix_y[mask] = np.around(
            cam_pts[mask, 1] / cam_pts[mask, 2] * intr.fy + intr.cy
        ).astype(np.int32)

        # Eliminate pixels outside view frustum.
        mask &= (pix_x >= 0) & (pix_x < intr.width) & (pix_y >= 0) & (pix_y < intr.height)
        depth_val = np.zeros_like(pix_x, dtype=np.float32)
        depth_val[mask] = depth_im[pix_y[mask], pix_x[mask]]

        sdf = depth_val - cam_pts[:, 2]
        valid_pts = (depth_val > 0) & (sdf >= -truncation_distance)
        tsdf = np.minimum(1.0, sdf / truncation_distance)
        valid_vox_x = vox_coords[valid_pts, 0]
        valid_vox_y = vox_coords[valid_pts, 1]
        valid_vox_z = vox_coords[valid_pts, 2]
        tsdf_new = tsdf[valid_pts]
        tsdf_vals = self.tsdf_volume[valid_vox_x, valid_vox_y, valid_vox_z]
        tsdf_vals = tsdf_vals.astype(np.float32) * DIVSHORTMAX
        w_old = self.weight_volume[valid_vox_x, valid_vox_y, valid_vox_z]
        obs_weight = 1
        tsdf_vol_new = (w_old * tsdf_vals + obs_weight * tsdf_new) / (w_old + obs_weight)
        tsdf_vol_new = np.clip(
            (tsdf_vol_new * SHORTMAX).astype(np.int16),
            a_min=-SHORTMAX,
            a_max=SHORTMAX,
        )

        tsdf_volume_new[valid_vox_x, valid_vox_y, valid_vox_z] = tsdf_vol_new
        w_new = np.minimum(w_old + obs_weight, MAX_WEIGHT)
        weight_volume_new[valid_vox_x, valid_vox_y, valid_vox_z] = w_new

        for i in range(3):
            color_volume_new[valid_vox_x, valid_vox_y, valid_vox_z, i] = (
                (
                    w_old * self.color_volume[valid_vox_x, valid_vox_y, valid_vox_z, i]
                    + obs_weight * color_im[pix_y[valid_pts], pix_x[valid_pts], i]
                )
                / (w_old + obs_weight)
            ).astype(np.uint8)

        return replace(
            self,
            tsdf_volume=tsdf_volume_new,
            weight_volume=weight_volume_new,
            color_volume=color_volume_new,
        )


class TSDFFusion:
    """Volumetric TSDF Fusion of RGB-D Images."""

    def __init__(self, camera_params: Intrinsic, config: GlobalConfig):
        """Set up the internal volume and camera parameters."""

        self._camera_params = camera_params
        self._config = config

        self._volume = None
        self._frame_id = None
        self.reset()

    def reset(self):
        self._volume = UniformTSDFVolume.initialize(self._config)
        self._frame_id = 0

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
        self._volume = self._volume.integrate(
            color_im,
            depth_im,
            se3_inverse(pose),
            self._camera_params,
            self._config.truncation_distance,
        )

        self._frame_id += 1

    def extract_pointcloud(self):
        return extract_points(self._volume)

    def extract_mesh(self) -> Mesh:
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


def se3_transform(xyz, transform):
    return (transform[:3, :3] @ xyz.T).T + transform[:3, 3]


def extract_points(volume: UniformTSDFVolume):
    """Extract a pointcloud from a TSDF volume."""
    raise NotImplementedError


def extract_mesh(volume: UniformTSDFVolume) -> Mesh:
    """Extract a surface mesh from a TSDF volume using the Marching Cubes algorithm."""
    tsdf_volume = volume.tsdf_volume.astype(np.float32) * DIVSHORTMAX
    mask = (tsdf_volume > -0.5) & (tsdf_volume < 0.5)
    verts, faces, norms, _ = marching_cubes(tsdf_volume, mask=mask, level=0)
    verts_ind = np.round(verts).astype(int)
    verts = verts * volume.voxel_scale + volume.center_point
    colors = volume.color_volume[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
    return verts, faces, norms, colors
