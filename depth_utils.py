import numpy as np
import cv2

def generate_depth_map(image):
    if image is None:
        raise ValueError("이미지가 없습니다.")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.applyColorMap(gray, cv2.COLORMAP_JET)

def generate_point_cloud(gray_image):
    h, w = gray_image.shape
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    Z = gray_image.astype(np.float32)
    return np.dstack((X, Y, Z))

def test_generate_depth_map():
    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
    depth_map = generate_depth_map(dummy_image)
    assert depth_map.shape == dummy_image.shape
    assert isinstance(depth_map, np.ndarray)

def test_generate_point_cloud():
    gray = np.ones((50, 50), dtype=np.uint8) * 128
    cloud = generate_point_cloud(gray)
    assert cloud.shape == (50, 50, 3)
    assert cloud.dtype == np.float32
