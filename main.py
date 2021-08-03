"""A demo for fusing RGB-D data from the 7-scenes dataset."""

import time
from pathlib import Path

from absl import app, flags
from tqdm import tqdm

import fusion
import utils

FLAGS = flags.FLAGS

flags.DEFINE_string("path", "data", "Path to dataset.")
flags.DEFINE_integer("save_freq", 25, "Frequency at which to dump mesh to disk.")


def main(_):
    path = Path(FLAGS.path)
    assert path.exists(), "f{path} does not exist."

    # Determine number of images in dataset.
    n_frames = len(list(path.glob("*.png")))

    # Camera parameters for the 7-scenes dataset.
    intr = fusion.Intrinsic(width=640, height=480, fx=585, fy=585, cx=320, cy=240)

    # Instantiate global config.
    # These parameters should be tuned for each scene.
    config = fusion.GlobalConfig(
        volume_size=(471, 300, 400),
        voxel_scale=0.02,
        truncation_distance=5 * 0.02,
        depth_cutoff_distance=4.0,
    )

    # Initialize TSDF volume with camera parameters and global config.
    volume = fusion.TSDFVolume.initialize(intr, config)

    # Loop through the RGB-D frames and integrate.
    start = time.time()
    for i in tqdm(range(n_frames)):
        try:
            color_im = utils.load_color(path / f"frame-{i:06}.color.png")
            depth_im = utils.load_depth(path / f"frame-{i:06}.depth.png")
            pose = utils.load_pose(path / f"frame-{i:06}.pose.txt")
        except:
            continue

        volume.integrate(color_im, depth_im, pose)

        if not i % FLAGS.save_freq:
            mesh_args = volume.extract_mesh()
            utils.meshwrite("./mesh.ply", *mesh_args)

    print(f"Average fps: {n_frames / (time.time() - start):.2}")

    mesh_args = volume.extract_mesh()
    utils.meshwrite("./mesh.ply", *mesh_args)


if __name__ == "__main__":
    app.run(main)
