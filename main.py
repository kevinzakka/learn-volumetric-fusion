from pathlib import Path

from absl import app, flags
from tqdm import tqdm

import fusion
import utils

FLAGS = flags.FLAGS

flags.DEFINE_string("path", "data", "Path to dataset.")

# Path templates.
DEPTH_PATH = "frame-{:06}.depth.png"
COLOR_PATH = "frame-{:06}.color.jpg"
POSE_PATH = "frame-{:06}.pose.txt"


def main(_):
    path = Path(FLAGS.path)
    assert path.exists(), "f{path} does not exist."

    # Load camera intrinsic parameters.
    intr_filename = path / "camera-intrinsics.txt"
    intr = fusion.Intrinsic.from_file(intr_filename, width=640, height=480)

    # Instantiate global config.
    config = fusion.GlobalConfig()
    config.volume_scale = 0.04

    # Instantiate fusion pipeline with camera parameters and global config.
    pipeline = fusion.TSDFFusion(intr, config)

    # Loop through the RGB-D frames and fuse.
    n_frames = 3
    for i in tqdm(range(n_frames)):
        pipeline.integrate(
            utils.load_color(path / COLOR_PATH.format(i)),
            utils.load_depth(path / DEPTH_PATH.format(i)),
            utils.load_pose(path / POSE_PATH.format(i)),
        )


if __name__ == "__main__":
    app.run(main)
