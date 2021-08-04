from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Tuple, Dict, List

import numpy as np
from skimage.measure import marching_cubes

# Global constants.
BYTES_TO_GIGABYTES = 1.0 / (1 << 30)
SHORTMAX = np.iinfo(np.int16).max
DIVSHORTMAX = 1.0 / SHORTMAX
MAX_WEIGHT = 128

# Global types.
Int3 = Tuple[int, int, int]
Mesh = Tuple[np.ndarray, ...]


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


# ======================================================= #
# Volume implementations.
# ======================================================= #


@dataclass(frozen=True)
class TSDFVolume(abc.ABC):
    """Base TSDF volume class."""

    camera_params: Intrinsic
    """The camera parameters."""

    config: GlobalConfig
    """The config values."""

    @abc.abstractstaticmethod
    def initialize(camera_params: Intrinsic, config: GlobalConfig) -> TSDFVolume:
        """Construct and return an instance of this class."""

    @abc.abstractmethod
    def _integrate(
        self,
        color_im: np.ndarray,
        depth_im: np.ndarray,
        pose: np.ndarray,
        intr: Intrinsic,
        truncation_distance: float,
        voxel_coords: np.ndarray,
        world_pts: np.ndarray,
    ):
        """Fuse RGB-D frame into volume."""

    def integrate(
        self,
        color_im: np.ndarray,
        depth_im: np.ndarray,
        pose: np.ndarray,
    ):
        """Integrate an RGB-D frame into the TSDF volume."""

        # Sanity check shapes.
        assert pose.shape == (4, 4)
        assert color_im.shape[:2] == depth_im.shape[:2]
        if (
            depth_im.shape[0] != self.camera_params.height
            or depth_im.shape[1] != self.camera_params.width
        ):
            raise ValueError("Depth image size does not match camera parameters.")

        # Truncate depth values >= than the cutoff.
        depth_im[depth_im >= self.config.depth_cutoff_distance] = 0.0

        self._integrate(
            color_im,
            depth_im,
            pose,
            self.camera_params,
            self.config.truncation_distance,
            self.voxel_coords,
            self.world_pts,
        )

    def extract_mesh(self) -> Mesh:
        return marching_cubes(
            self.tsdf_volume,
            self.color_volume,
            self.config.voxel_scale,
        )


@dataclass(frozen=True)
class UniformTSDFVolume(TSDFVolume):
    """A TSDF volume with a uniform voxel grid.

    References:
        Curless and Levoy, 1996: A Volumetric Method for Building Complex Models from
            Range Images.
    """

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

    @staticmethod
    def initialize(camera_params: Intrinsic, config: GlobalConfig) -> UniformTSDFVolume:
        # Allocate volumes.
        tsdf_volume = np.ones(config.volume_size, dtype=np.int16)
        weight_volume = np.zeros(config.volume_size, dtype=np.int16)
        color_volume = np.zeros(config.volume_size + (3,), dtype=np.uint8)

        # Create voxel grid coordinates.
        voxel_coords = np.indices(config.volume_size).reshape(3, -1).T

        # Convert voxel grid coordiantes to base frame camera coordinates.
        world_pts = (voxel_coords.astype(np.float32) + 0.5) * config.voxel_scale

        return UniformTSDFVolume(
            camera_params,
            config,
            tsdf_volume,
            weight_volume,
            color_volume,
            voxel_coords,
            world_pts,
        )

    def _integrate(
        self,
        color_im: np.ndarray,
        depth_im: np.ndarray,
        pose: np.ndarray,
        intr: Intrinsic,
        truncation_distance: float,
        voxel_coords: np.ndarray,
        world_pts: np.ndarray,
    ):
        # Shift the voxel coordinate frame origin from the bottom left corner of the
        # cube to the cube centroid.
        pose[:3, 3] += np.array(self.config.volume_size) * self.config.voxel_scale * 0.5

        # Convert from base frame camera coordinates to current frame camera coordinates.
        cam_pts = apply_se3(world_pts, se3_inverse(pose))

        # Convert to camera pixels.
        with np.errstate(divide="ignore"):
            pix_x = cam_pts[:, 0] / cam_pts[:, 2]
            pix_x *= intr.fx
            pix_x += intr.cx
            pix_x = np.round(pix_x).astype(np.int32)
            pix_y = cam_pts[:, 1] / cam_pts[:, 2]
            pix_y *= intr.fy
            pix_y += intr.cy
            pix_y = np.round(pix_y).astype(np.int32)
        pix_x = np.nan_to_num(pix_x, copy=False, nan=0)
        pix_y = np.nan_to_num(pix_y, copy=False, nan=0)

        # Eliminate pixels outside view frustum.
        mask = cam_pts[:, 2] > 0
        mask &= pix_x >= 0
        mask &= pix_x < intr.width
        mask &= pix_y >= 0
        mask &= pix_y < intr.height
        depth_val = np.zeros_like(pix_x, dtype=np.float32)
        depth_val[mask] = depth_im[pix_y[mask], pix_x[mask]]

        tsdf = depth_val - cam_pts[:, 2]
        valid_pts = depth_val > 0
        valid_pts &= tsdf >= -truncation_distance
        tsdf[valid_pts] /= truncation_distance
        tsdf[valid_pts] = np.minimum(1.0, tsdf[valid_pts])

        obs_weight = 1
        vx, vy, vz = np.moveaxis(voxel_coords[valid_pts], 1, 0)

        # Integrate TSDF.
        tsdf_new = tsdf[valid_pts]
        tsdf_old = self.tsdf_volume[vx, vy, vz].astype(np.float32) * DIVSHORTMAX
        w_old = self.weight_volume[vx, vy, vz]
        w_new = w_old + obs_weight
        tsdf_vol_new = (w_old * tsdf_old + obs_weight * tsdf_new) / w_new
        self.tsdf_volume[vx, vy, vz] = np.clip(
            (tsdf_vol_new * SHORTMAX).astype(np.int16),
            a_min=-SHORTMAX,
            a_max=SHORTMAX,
        )
        self.weight_volume[vx, vy, vz] = np.minimum(w_new, MAX_WEIGHT)

        # Integrate color.
        for i in range(3):
            color_old = self.color_volume[vx, vy, vz, i]
            color_new = color_im[pix_y[valid_pts], pix_x[valid_pts], i]
            self.color_volume[vx, vy, vz, i] = (
                (w_old * color_old + obs_weight * color_new) / w_new
            ).astype(np.uint8)


@dataclass
class Voxel:
    """A voxel unit."""
    tsdf: float
    weight: int
    color: Int3

@dataclass
class VoxelBlock:
    """A block of voxels."""
    voxels: List[Voxel]


@dataclass(frozen=True)
class HashTSDFVolume(TSDFVolume):
    """A TSDF volume that uses a hashing scheme to store voxel data.

    References:
        Nießner et al, 2013: Real-time 3D Reconstruction at Scale using Voxel Hashing.
    """

    voxel_dict: Dict[np.ndarray, VoxelBlock] = {}

    @staticmethod
    def initialize(camera_params: Intrinsic, config: GlobalConfig) -> UniformTSDFVolume:
        return HashTSDFVolume(
            camera_params,
            config,
        )

    def _integrate(
        self,
        color_im: np.ndarray,
        depth_im: np.ndarray,
        pose: np.ndarray,
        intr: Intrinsic,
        truncation_distance: float,
        voxel_coords: np.ndarray,
        world_pts: np.ndarray,
    ):
        pass
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
    verts, faces, norms, _ = marching_cubes(tsdf_volume, mask=mask, level=0)
    vix, viy, viz = np.round(verts).astype(np.int16).T
    verts = verts * voxel_scale
    colors = color_volume[vix, viy, viz]
    return verts, faces, norms, colors
