import cv2
import numpy as np

def generate_heatmap(image_shape, detections, sigma_scale=0.35):
    h, w = image_shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)

    for (x1, y1, x2, y2, conf) in detections:
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        box_w, box_h = (x2 - x1), (y2 - y1)
        sigma_x = max(box_w * sigma_scale, 5)
        sigma_y = max(box_h * sigma_scale, 5)

        y_grid, x_grid = np.ogrid[:h, :w]
        gaussian = np.exp(
            -(((x_grid - cx) ** 2) / (2 * sigma_x ** 2)
              + ((y_grid - cy) ** 2) / (2 * sigma_y ** 2))
        )
        heatmap += gaussian * conf

    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    return heatmap

def heatmap_to_bgr(heatmap):
    heatmap_uint8 = np.uint8(255 * heatmap)
    return cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

def generate_mask(heatmap, threshold=0.4):
    return np.uint8(heatmap >= threshold) * 255

def get_yolo_labels(results):
    labels = []
    names = getattr(results, "names", {}) or {}
    for box in results.boxes:
        if box.cls is None or len(box.cls) == 0:
            labels.append("other")
            continue
        class_id = int(box.cls[0])
        labels.append(names.get(class_id, "other"))
    return labels