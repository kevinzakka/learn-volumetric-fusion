from __future__ import annotations

import jax_dataclasses
from pathlib import Path
from typing import Tuple, Union

import numpy as onp
from skimage.measure import marching_cubes

import jax.numpy as np
from jax import jit, device_put, partial

# Global constants.
BYTES_TO_GIGABYTES = 1.0 / (1 << 30)
SHORTMAX = np.iinfo(np.int16).max
DIVSHORTMAX = 0.0000305185
MAX_WEIGHT = 128

# Global types.
Int3 = Tuple[int, int, int]
Mesh = Tuple[np.ndarray, ...]


@jax_dataclasses.pytree_dataclass
class GlobalConfig:
    """The global configuration."""

    volume_size: Int3 = jax_dataclasses.static_field(default=(512, 512, 512))
    """The overall size of the volume in meters (m)."""

    voxel_scale: float = jax_dataclasses.static_field(default=0.02)
    """The volume discretization (in m)."""

    truncation_distance: float = jax_dataclasses.static_field(default=20.0)
    """The truncation distance (in m) for updating the TSDF volume."""

    depth_cutoff_distance: float = jax_dataclasses.static_field(default=4.0)
    """The distance (in m) after which the depth is set to 0 in the depth frames."""


@jax_dataclasses.pytree_dataclass
class Intrinsic:
    """Pinhole camera parameters."""

    width: int = jax_dataclasses.static_field()
    height: int = jax_dataclasses.static_field()
    fx: float = jax_dataclasses.static_field()
    fy: float = jax_dataclasses.static_field()
    cx: float = jax_dataclasses.static_field()
    cy: float = jax_dataclasses.static_field()

    # @property
    # def matrix(self) -> np.ndarray:
    #     return np.array(
    #         [
    #             [self.fx, 0.0, self.cx],
    #             [0.0, self.fy, self.cy],
    #             [0.0, 0.0, 1.0],
    #         ],
    #         dtype=np.float32,
    #     )

    @staticmethod
    def from_file(filename: Union[str, Path], width: int, height: int) -> Intrinsic:
        mat = onp.loadtxt(filename, delimiter=" ", dtype=onp.float32)
        return Intrinsic(width, height, mat[0, 0], mat[1, 1], mat[0, 2], mat[1, 2])


@jax_dataclasses.pytree_dataclass
class TSDFVolume:
    """A TSDF volume with a uniform voxel grid [1].

    References:
        [1]: Curless and Levoy, 1996.
    """

    camera_params: Intrinsic = jax_dataclasses.static_field()
    """The camera parameters."""

    config: GlobalConfig = jax_dataclasses.static_field()
    """The config values."""

    tsdf_volume: np.ndarray
    """The global volume in which depth frames are fused."""

    weight_volume: np.ndarray
    """Holds the weights of the moving average."""

    color_volume: np.ndarray
    """The global color volume in which color frames are fused."""

    @staticmethod
    def initialize(camera_params: Intrinsic, config: GlobalConfig) -> TSDFVolume:
        tsdf_volume = np.ones(config.volume_size, dtype=np.int16)
        weight_volume = np.zeros(config.volume_size, dtype=np.int16)
        color_volume = np.zeros(config.volume_size + (3,), dtype=np.uint8)

        return TSDFVolume(
            camera_params,
            config,
            tsdf_volume,
            weight_volume,
            color_volume,
        )

    @property
    def center_point(self) -> np.ndarray:
        arr = np.array(self.config.volume_size, dtype=np.float32)
        return -0.5 * self.config.voxel_scale * arr

    def integrate(
        self,
        color_im: np.ndarray,
        depth_im: np.ndarray,
        pose: np.ndarray,
    ) -> TSDFVolume:
        """Integrate an RGB-D frame into the TSDF volume."""

        # Sanity check shapes.
        assert pose.shape == (4, 4)
        assert color_im.shape[:2] == depth_im.shape[:2]
        if (
            depth_im.shape[0] != self.camera_params.height
            or depth_im.shape[1] != self.camera_params.width
        ):
            raise ValueError(f"Depth image size does not match camera parameters.")

        # Truncate depth values >= than the cutoff.
        depth_im[depth_im >= self.config.depth_cutoff_distance] = 0.0

        # Send frames to device.
        color_im = device_put(color_im)
        depth_im = device_put(depth_im)
        pose = device_put(pose)

        return self._integrate(
            color_im,
            depth_im,
            se3_inverse(pose),
            self.camera_params,
            self.config.truncation_distance,
        )

    def extract_mesh(self) -> Mesh:
        tsdf_arr = onp.array(self.tsdf_volume.astype(np.float32) * DIVSHORTMAX)
        color_arr = onp.array(self.color_volume)
        cp = onp.array(self.center_point)

        return extract_mesh(
            tsdf_arr,
            color_arr,
            self.config.voxel_scale,
            cp,
        )

    # @partial(jit, static_argnums=[4, 5])
    def _integrate(
        self,
        color_im: np.ndarray,
        depth_im: np.ndarray,
        pose: np.ndarray,
        intr: Intrinsic,
        truncation_distance: float,
    ) -> TSDFVolume:
        tsdf_volume_new = np.array(self.tsdf_volume)
        weight_volume_new = np.array(self.weight_volume)
        color_volume_new = np.array(self.color_volume)

        # Create voxel grid coordinates.
        vox_coords = np.indices(self.config.volume_size).reshape(3, -1).T

        # Convert voxel center from grid coordinates to base frame camera coordinates.
        world_pts = (
            (vox_coords.astype(np.float32) + 0.5) * self.config.voxel_scale
        ) + self.center_point

        # Convert from base frame camera coordinates to current frame camera coordinates.
        cam_pts = se3_transform(world_pts, pose)

        # Remove points with negative z.
        mask = cam_pts[:, 2] > 0

        from ipdb import set_trace
        set_trace()

        mask = cam_pts[:, 2] > 0
        pix_x = np.around(cam_pts[:, 0] / cam_pts[:, 2] * intr.fx + intr.cx).astype(np.int32)

        pix_y = np.around(cam_pts[:, 1] / cam_pts[:, 2] * intr.fy + intr.cy).astype(np.int32)

        # pix_x = np.full_like(cam_pts[:, 0], -1, dtype=np.int32)
        # pix_y = np.full_like(cam_pts[:, 0], -1, dtype=np.int32)
        # pix_x = pix_x.at[mask].set(
        #     np.around(
        #         cam_pts[mask, 0] / cam_pts[mask, 2] * intr.fx + intr.cx
        #     ).astype(np.int32)
        # )
        # pix_y = pix_y.at[mask].set(
        #     np.around(
        #         cam_pts[mask, 1] / cam_pts[mask, 2] * intr.fy + intr.cy
        #     ).astype(np.int32)
        # )

        # # Eliminate pixels outside view frustum.
        # mask &= (pix_x >= 0) & (pix_x < intr.width) & (pix_y >= 0) & (pix_y < intr.height)
        # depth_val = np.zeros_like(pix_x, dtype=np.float32)
        # depth_val = depth_val.at[mask].set(depth_im[pix_y[mask], pix_x[mask]])

        # sdf = depth_val - cam_pts[:, 2]
        # valid_pts = (depth_val > 0) & (sdf >= -truncation_distance)
        # tsdf = np.minimum(1.0, sdf / truncation_distance)
        # valid_vox_x = vox_coords[valid_pts, 0]
        # valid_vox_y = vox_coords[valid_pts, 1]
        # valid_vox_z = vox_coords[valid_pts, 2]
        # tsdf_new = tsdf[valid_pts]
        # tsdf_vals = self.tsdf_volume[valid_vox_x, valid_vox_y, valid_vox_z]
        # tsdf_vals = tsdf_vals.astype(np.float32) * DIVSHORTMAX
        # w_old = self.weight_volume[valid_vox_x, valid_vox_y, valid_vox_z]
        # obs_weight = 1
        # tsdf_vol_new = (w_old * tsdf_vals + obs_weight * tsdf_new) / (w_old + obs_weight)
        # tsdf_vol_new = np.clip(
        #     (tsdf_vol_new * SHORTMAX).astype(np.int16),
        #     a_min=-SHORTMAX,
        #     a_max=SHORTMAX,
        # )

        # tsdf_volume_new = tsdf_volume_new.at[valid_vox_x, valid_vox_y, valid_vox_z].set(tsdf_vol_new)
        # w_new = np.minimum(w_old + obs_weight, MAX_WEIGHT)
        # weight_volume_new = weight_volume_new.at[valid_vox_x, valid_vox_y, valid_vox_z].set(w_new)

        # for i in range(3):
        #     color_volume_new = color_volume_new.at[valid_vox_x, valid_vox_y, valid_vox_z, i].set(
        #         (
        #             w_old * self.color_volume[valid_vox_x, valid_vox_y, valid_vox_z, i]
        #             + obs_weight * color_im[pix_y[valid_pts], pix_x[valid_pts], i]
        #         )
        #         / (w_old + obs_weight)
        #     ).astype(np.uint8)

        with jax_dataclasses.copy_and_mutate(self) as new_state:
            new_state.tsdf_volume = tsdf_volume_new
            new_state.weight_volume = weight_volume_new
            new_state.color_volume = color_volume_new
        return new_state


# ======================================================= #
# Helper methods.
# ======================================================= #

def se3_inverse(se3: np.ndarray) -> np.ndarray:
    """Compute the inverse of an SE(3) transform."""
    inv_se3 = np.empty_like(se3)
    tr = se3[:3, :3].T
    inv_se3 = inv_se3.at[:3, :3].set(tr)
    inv_se3 = inv_se3.at[:3, 3].set(-tr @ se3[:3, 3])
    inv_se3 = inv_se3.at[3, :].set([0, 0, 0, 1.0])
    return inv_se3


def se3_transform(pts: np.ndarray, se3: np.ndarray) -> np.ndarray:
    """Apply an SE(3) transform to a pointcloud."""
    return (se3[:3, :3] @ pts.T).T + se3[:3, 3]


def extract_mesh(
    tsdf_volume: onp.ndarray,
    color_volume: onp.ndarray,
    voxel_scale: float,
    origin: onp.ndarray,
) -> Mesh:
    """Extract a surface mesh from a TSDF volume using Marching Cubes."""
    mask = (tsdf_volume > -0.5) & (tsdf_volume < 0.5)
    verts, faces, norms, _ = marching_cubes(tsdf_volume, mask=mask, level=0)
    vix, viy, viz = onp.round(verts).astype(onp.int16).T
    verts = verts * voxel_scale + origin
    colors = color_volume[vix, viy, viz]
    return verts, faces, norms, colors
