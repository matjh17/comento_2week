import cv2
import numpy as np
import pytest

def generate_depth_map(image):
    if image is None:
        raise ValueError("입력된 이미지가 없습니다.")
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    depth_map = cv2.applyColorMap(grayscale, cv2.COLORMAP_JET)
    return depth_map

def test_generate_depth_map():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    depth_map = generate_depth_map(image)
    assert depth_map.shape == image.shape
    assert isinstance(depth_map, np.ndarray)

def test_generate_depth_map_with_none():
    with pytest.raises(ValueError):
        generate_depth_map(None)

if __name__ == "__main__":
    pytest.main()
