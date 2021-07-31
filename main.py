from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from absl import app, flags
from PIL import Image
from tqdm import tqdm

import fusion

FLAGS = flags.FLAGS

flags.DEFINE_string("path", "data", "Path to dataset.")

# Path templates.
DEPTH_PATH = "frame-{:06}.depth.png"
COLOR_PATH = "frame-{:06}.color.jpg"
POSE_PATH = "frame-{:06}.pose.txt"


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
    img[img == 65535] = 0.0
    return np.array(img)  # Values are in millimeters.


def load_color(filename: Union[str, Path]) -> np.ndarray:
    return np.array(Image.open(filename).convert("RGB"), dtype=np.uint8)


def load_pose(filename: Union[str, Path]) -> np.ndarray:
    return np.array(np.loadtxt(filename), dtype=np.float64)


def get_convex_hull(path: Path, intr: fusion.Intrinsic) -> np.ndarray:
    """Compute extrema of convex hull of camera view frustums."""
    depth_filenames = list(path.glob("*depth.png"))
    pose_filenames = list(path.glob("*pose.txt"))
    n_files = len(pose_filenames)
    vol_bounds = np.zeros((3, 2))
    # for depth_f, pose_f in tqdm(zip(depth_filenames, pose_filenames), total=n_files):
    # depth_im = load_depth(depth_f)
    # cam_pose = load_pose(pose_f)
    # view_frustum = fusion.get_view_frustum(depth_im, intr, cam_pose)
    # vol_bounds[:, 0] = np.minimum(vol_bounds[:, 0], np.amin(view_frustum, axis=1))
    # vol_bounds[:, 1] = np.maximum(vol_bounds[:, 1], np.amax(view_frustum, axis=1))
    return vol_bounds


def main(_):
    path = Path(FLAGS.path)
    assert path.exists(), "f{path} does not exist."

    # Load camera intrinsic parameters.
    intr = load_intrinsics(path / "camera-intrinsics.txt")

    # Instantiate global config.
    config = fusion.GlobalConfig()
    config.volume_scale = 0.04

    # Instantiate fusion pipeline with camera parameters and global config.
    pipeline = fusion.TSDFFusion(intr, config)

    # Loop through the RGB-D frames and fuse.
    n_frames = 3
    for i in tqdm(range(n_frames)):
        pipeline.integrate(
            load_color(path / COLOR_PATH.format(i)),
            load_depth(path / DEPTH_PATH.format(i)),
            load_pose(path / POSE_PATH.format(i)),
        )


if __name__ == "__main__":
    app.run(main)
