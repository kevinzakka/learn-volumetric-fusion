import dataclasses
import time
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from absl import app, flags
from PIL import Image
from tqdm import tqdm

import fusion

FLAGS = flags.FLAGS

flags.DEFINE_string("path", "data/", "Path to dataset.")
flags.DEFINE_float(
    "volume_bounds_frac", 0.05, "Fraction of the data to estimate volume bounds on."
)

# flags.mark_flag_as_required("experiment_name")

# Path templates.
DEPTH_PATH = "frame-{:06}.depth.png"
COLOR_PATH = "frame-{:06}.color.jpg"
POSE_PATH = "frame-{:06}.pose.txt"

# @dataclasses.dataclass(frozen=True)
# class Voxel:
#     voxel_size: float
#     """The volume discretization in meters."""

#     volume_bounds: jnp.ndarray
#     """The Cartesian bounds (min/max) in meters."""


def load_intrinsics(filename: Union[str, Path]) -> fusion.Intrinsic:
    mat = np.loadtxt(filename, delimiter=" ")
    return fusion.Intrinsic(
        width=640,
        height=480,
        fx=mat[0, 0],
        fy=mat[1, 1],
        cx=mat[0, 2],
        cy=mat[1, 2],
    )


def load_depth(filename: Union[str, Path]) -> np.ndarray:
    img = np.array(Image.open(filename), dtype=np.float32)
    img /= 1000.0
    img[img == 65.535] = 0
    return img


def load_color(filename: Union[str, Path]) -> np.ndarray:
    return np.array(Image.open(filename).convert("RGB"), dtype=np.uint8)


def load_pose(filename: Union[str, Path]) -> np.ndarray:
    return np.loadtxt(filename).astype(np.float32)


def get_convex_hull(
    path: Path, intr: fusion.Intrinsic, max_frames: int = -1
) -> np.ndarray:
    """Compute extrema of convex hull of camera view frustums."""
    depth_filenames = list(path.glob("*depth.png"))
    pose_filenames = list(path.glob("*pose.txt"))
    if max_frames != -1:
        np.random.shuffle(depth_filenames)
        np.random.shuffle(pose_filenames)
        depth_filenames = depth_filenames[:max_frames]
        pose_filenames = pose_filenames[:max_frames]
    n_files = len(pose_filenames)
    vol_bounds = np.zeros((3, 2))
    for depth_f, pose_f in tqdm(zip(depth_filenames, pose_filenames), total=n_files):
        depth_im = load_depth(depth_f)
        cam_pose = load_pose(pose_f)
        view_frustum = fusion.get_view_frustum(depth_im, intr, cam_pose)
        vol_bounds[:, 0] = np.minimum(vol_bounds[:, 0], np.amin(view_frustum, axis=1))
        vol_bounds[:, 1] = np.maximum(vol_bounds[:, 1], np.amax(view_frustum, axis=1))
    return vol_bounds


def main(_):
    path = Path(FLAGS.path)
    assert path.exists(), "f{path} does not exist."

    # Figure out the total number of frames.
    n_total = len(list(path.glob("*jpg")))

    # Load camera intrinsic parameters.
    intr = load_intrinsics(path / "camera-intrinsics.txt")

    # Compute volume bounds in world coordindates of the convex hull of camera view
    # frustums in the dataset.
    volume_bounds = get_convex_hull(
        path,
        intr,
        int(FLAGS.volume_bounds_frac * n_total),
    )



if __name__ == "__main__":
    app.run(main)
