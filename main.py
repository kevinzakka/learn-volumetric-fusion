import dataclasses
import time
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import tqdm
from absl import app, flags
from PIL import Image

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


def get_convex_hull(intr: fusion.Intrinsic, max_frames: int = -1) -> np.ndarray:
    """Compute the bounds of the convex hull of camera view frustums."""
    print(max_frames)
    volume_bounds = np.zeros((3, 2))
    # Load depth image.
    # Load camera pose.
    # Compute camera view frustum.
    # Extend convex hull.
    pass


def main(_):
    path = Path(FLAGS.path)
    assert path.exists(), "f{path} does not exist."

    # Figure out the total number of frames.
    n_total = len(list(path.glob("*jpg")))

    # Load camera intrinsic parameters.
    intr = load_intrinsics(path / "camera-intrinsics.txt")

    # Compute volume bounds on 10% of the dataset.
    volume_bounds = get_convex_hull(intr, int(FLAGS.volume_bounds_frac * n_total))

    # pose = load_pose(path / POSE_PATH.format(500))
    # color_im = load_color(path / COLOR_PATH.format(500))
    # depth_im = load_depth(path / DEPTH_PATH.format(500))
    # frame = fusion.Frame(
    #     color=color_im,
    #     depth=depth_im,
    #     intrinsic=intr,
    #     extrinsic=pose,
    # )
    # fig, axes = plt.subplots(1, 2)
    # for ax, im in zip(axes, [color_im, depth_im]):
    #     ax.imshow(im)
    # plt.show()


if __name__ == "__main__":
    app.run(main)
