from pathlib import Path
from typing import Union

import cv2
import numpy as np


def load_depth(filename):
    img = cv2.imread(str(filename), cv2.IMREAD_ANYDEPTH).astype(np.float32)
    # Set invalid depth values to 0. This is specific to the 7-scenes dataset, see:
    # https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/
    img[img == 65535.0] = 0.0
    return img


def load_color(filename: Union[str, Path]) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(str(filename)), cv2.COLOR_BGR2RGB)


def load_pose(filename: Union[str, Path]) -> np.ndarray:
    return np.loadtxt(filename, dtype=np.float32)
