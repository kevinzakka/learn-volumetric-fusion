import dataclasses

import jax.numpy as jnp
import numpy as np


@dataclasses.dataclass(frozen=True)
class Intrinsic:
    """Pinhole camera parameters."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @property
    def matrix(self) -> jnp.ndarray:
        return jnp.array(
            [
                [self.fx, 0, self.cx],
                [0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=jnp.float32,
        )


@dataclasses.dataclass(frozen=True)
class Frame:
    color: jnp.ndarray
    """An RGB image, of shape (H, W, 3)."""

    depth: jnp.ndarray
    """A depth image, of shape (H, W)."""

    intrinsic: Intrinsic
    """Pinhole camera intrinsic parameters, of shape (3, 3)."""

    extrinsic: jnp.ndarray
    """Camera pose, of shape (4, 4)."""


def get_view_frustum(
    depth_im: np.ndarray,
    intr: Intrinsic,
    extr: np.ndarray,
) -> np.ndarray:
    """Get the corners of the camera frustum from the depth image."""
    pass


def rigid_transform(xyz, transform):
    """Applies an SE(3) transform on a pointcloud."""
    # Homogenize coordinates.
    # Matrix multiplication.
    pass



class TSDFVolume:
    """Volumetric TSDF Fusion of RGB-D Images."""

    def __init__(
        self,
        volume_bounds,
        voxel_size: float,
        sdf_truncation: float,
    ) -> None:
        """Constructor.

        Args:
            volume_bounds:
            voxel_size:
            sdf_truncation:
        """
        self.reset()

    def reset(self) -> None:
        """Reset the volume."""

    def integrate(self, frame: Frame, weight: float) -> None:
        """Integrate an RGB-D frame into the TSDF volume.

        Args:
            color_im:
            depth_im:
            cam_intr:
            cam_pose:
            weight:
        """
