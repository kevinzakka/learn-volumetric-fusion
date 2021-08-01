from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Union

import numpy as np

# Global constants.
BYTES_TO_MEGABYTES = 1.0 / 1024.0
BYTES_TO_GIGABYTES = 1.0 / 1024.0 / 1024.0

# Global types.
Int3 = Tuple[int, int, int]


@dataclass
class GlobalConfig:
    """The global configuration."""

    volume_size: Int3 = (512, 512, 512)
    """The overall size of the volume in mm."""

    voxel_scale: float = 2.0
    """The volume discretization in mm."""

    truncation_distance: float = 25.0
    """The truncation distance for updating the TSDF volume."""

    depth_cutoff_distance: float = 3000.0
    """The distance in mm after which the depth is set to 0 in the depth frames."""

    triangles_buffer_size: int = 3 * 2000000
    pointcloud_buffer_size: int = 3 * 2000000
    """The maximum buffer size for exporting triangles and pointclouds."""


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
        cls,
        filename: Union[str, Path],
        width: int,
        height: int,
    ) -> "Intrinsic":
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
        self._color_volume = None

    def reset(self):
        self._tsdf_volume = np.zeros(self.volume_size + (2,), dtype=np.float32)
        self._color_volume = np.zeros(self.volume_size, dtype=np.uint8)

        print(f"TSDF volume: {self._tsdf_volume.nbytes * BYTES_TO_GIGABYTES} GB")

    @property
    def tsdf_volume(self) -> np.ndarray:
        return self._tsdf_volume

    @property
    def color_volume(self) -> np.ndarray:
        return self._color_volume


class TSDFFusion:
    """Volumetric TSDF Fusion of RGB-D Images."""

    def __init__(self, camera_params: Intrinsic, config: GlobalConfig):
        """Set up the internal volume and camera parameters."""

        self._camera_params = camera_params
        self._config = config
        self._volume = UniformTSDFVolume(config.volume_size, config.voxel_scale)

        self.reset()

    def reset(self):
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

        # Fuse the RGB-D frame into the volume.
        surface_reconstruction(
            depth_im,
            color_im,
            self._volume,
            self._camera_params,
            self._config.truncation_distance,
            se3_inverse(pose),
        )

    def extract_pointcloud(self):
        return extract_points(
            self._volume,
            self._config.pointcloud_buffer_size,
        )

    def extract_mesh(self):
        return marching_cubes(
            self._volume,
            self._config.triangles_buffer_size,
        )


def se3_inverse(pose: np.ndarray) -> np.ndarray:
    """Compute the inverse of an SE(3) transform."""
    inv_pose = np.empty_like(pose)
    tr = pose[:3, :3].T
    inv_pose[:3, :3] = tr
    inv_pose[:3, 3] = -tr @ pose[:3, 3]
    inv_pose[3, :] = [0, 0, 0, 1.0]
    return inv_pose


def surface_reconstruction(
    depth_image: np.ndarray,
    color_image: np.ndarray,
    volume: UniformTSDFVolume,
    cam_params: Intrinsic,
    truncation_distance: float,
    pose: np.ndarray,
):
    """Integration of surface measurements into a global volume."""

    # Compute xyz coordinates using voxel_scale.
    # Convert to camera coordinates.
    # If z is <= 0 remove.
    # Convert to pixel coordinates using intrinsic.
    # Eliminate pixels outside view frustum.
    # Get depth[u, v]. If <=0, remove.
    # Reproject and unit normalize.
    # Compute sdf.


def extract_points(volume: UniformTSDFVolume, buffer_size: int):
    """Extract a pointcloud from a TSDF volume."""


def marching_cubes(volume: UniformTSDFVolume, buffer_size: int):
    """Extract a surface mesh from a TSDF volume using the Marching Cubes algorithm."""
