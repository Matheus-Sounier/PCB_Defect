import numpy as np
import cv2
import pytest

@pytest.fixture
def sample_image(tmp_path):
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    path = tmp_path / "img.png"
    cv2.imwrite(str(path), img)
    return str(path), img

@pytest.fixture
def detections():
    return [(10, 10, 30, 30, 0.9), (40, 10, 60, 30, 0.8)]

@pytest.fixture
def labels():
    return ["damaged_trace", "other"]
