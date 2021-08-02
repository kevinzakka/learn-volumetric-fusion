from pathlib import Path
from typing import Union

import cv2
import numpy as np


def load_depth(filename):
    # depth in millimeters, 16-bit, PNG, invalid depth is set to 65535.
    img = cv2.imread(str(filename), cv2.IMREAD_ANYDEPTH).astype(np.float32)
    # Set invalid depth values to 0. This is specific to the 7-scenes dataset, see:
    # https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/
    img[img == 65535] = 0
    return img


def load_color(filename: Union[str, Path]) -> np.ndarray:
    # RGB, 24-bit, PNG.
    return cv2.cvtColor(cv2.imread(str(filename)), cv2.COLOR_BGR2RGB)


def load_pose(filename: Union[str, Path]) -> np.ndarray:
    # camera-to-world, 4×4 matrix in homogeneous coordinates.
    return np.loadtxt(filename, dtype=np.float32)


def meshwrite(filename, verts, faces, norms, colors):
    """Save a 3D mesh to a polygon .ply file."""
    # Write header
    ply_file = open(filename, "w")
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex %d\n" % (verts.shape[0]))
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    ply_file.write("property float nx\n")
    ply_file.write("property float ny\n")
    ply_file.write("property float nz\n")
    ply_file.write("property uchar red\n")
    ply_file.write("property uchar green\n")
    ply_file.write("property uchar blue\n")
    ply_file.write("element face %d\n" % (faces.shape[0]))
    ply_file.write("property list uchar int vertex_index\n")
    ply_file.write("end_header\n")

    # Write vertex list
    for i in range(verts.shape[0]):
        ply_file.write(
            "%f %f %f %f %f %f %d %d %d\n"
            % (
                verts[i, 0],
                verts[i, 1],
                verts[i, 2],
                norms[i, 0],
                norms[i, 1],
                norms[i, 2],
                colors[i, 0],
                colors[i, 1],
                colors[i, 2],
            )
        )

    # Write face list
    for i in range(faces.shape[0]):
        ply_file.write("3 %d %d %d\n" % (faces[i, 0], faces[i, 1], faces[i, 2]))

    ply_file.close()
