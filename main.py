from pathlib import Path

from absl import app, flags
from tqdm import tqdm

import fusion
import utils

FLAGS = flags.FLAGS

flags.DEFINE_string("path", "data", "Path to dataset.")


def main(_):
    path = Path(FLAGS.path)
    assert path.exists(), "f{path} does not exist."

    # Load camera intrinsic parameters.
    intr_filename = path / "camera-intrinsics.txt"
    intr = fusion.Intrinsic.from_file(intr_filename, width=640, height=480)

    # Instantiate global config.
    config = fusion.GlobalConfig(
        volume_size=(512, 512, 512),
        voxel_scale=0.02,
        truncation_distance=5 * 0.02,
        depth_cutoff_distance=4.0,
    )
    print(config)

    # Initialize TSDF volume with camera parameters and global config.
    volume = fusion.TSDFVolume.initialize(intr, config)

    # Loop through the RGB-D frames and integrate.
    n_frames = 100
    for i in tqdm(range(0, n_frames, 5)):
        volume = volume.integrate(
            utils.load_color(path / f"frame-{i:06}.color.jpg"),
            utils.load_depth(path / f"frame-{i:06}.depth.png"),
            utils.load_pose(path / f"frame-{i:06}.pose.txt"),
        )
        if not i % 25:
            mesh_args = volume.extract_mesh()
            utils.meshwrite("./mesh.ply", *mesh_args)

    mesh_args = volume.extract_mesh()
    utils.meshwrite("./mesh.ply", *mesh_args)


if __name__ == "__main__":
    app.run(main)
