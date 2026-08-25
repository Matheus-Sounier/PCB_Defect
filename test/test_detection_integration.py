import os
import numpy as np
import pytest

from src.detection.yolo_pipeline import (
    generate_heatmap,
    heatmap_to_bgr,
    generate_mask,
    generate_segmentation,
    build_panel,
)
from src.reporting.reporting import build_detection_report, save_detection_report

@pytest.mark.integration
def test_pipeline_integration_creates_report_and_panel(sample_image, detections, labels, tmp_path):
    img_path, img = sample_image

    heatmap = generate_heatmap(img.shape, detections)
    assert heatmap.shape == img.shape[:2]

    bgr = heatmap_to_bgr(heatmap)
    assert bgr.shape == img.shape

    mask = generate_mask(heatmap, threshold=0.1)
    assert mask.dtype == np.uint8

    seg = generate_segmentation(img, detections, labels=labels, use_rectangle=True)
    assert seg.shape == img.shape

    panel = build_panel([img, bgr, mask, seg], ["Image", "Heat", "Mask", "Segmentation"])
    assert panel.shape[2] == 3

    report = build_detection_report(img_path, detections, labels)
    assert report["detections"] == len(detections)

    out = save_detection_report(report, output_dir=str(tmp_path))
    assert os.path.exists(out)