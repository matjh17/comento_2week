import numpy as np
import pytest
import cv2
from depth_utils import generate_depth_map, generate_point_cloud

def test_generate_depth_map():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    depth = generate_depth_map(img)
    assert depth.shape == img.shape
    assert isinstance(depth, np.ndarray)
    assert depth.dtype == np.uint8

def test_generate_point_cloud():
    gray = np.ones((50, 50), dtype=np.uint8) * 128
    cloud = generate_point_cloud(gray)
    assert cloud.shape == (50, 50, 3)
    assert cloud.dtype == np.float32
