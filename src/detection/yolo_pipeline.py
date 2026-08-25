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

def generate_segmentation(image, detections, labels=None, use_rectangle=True):
    seg = image.copy()
    labels = labels or [None] * len(detections)

    def draw_label(i, base_x, base_y):
        if not labels[i]:
            return
        text = str(labels[i]).replace("_", " ")
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.7
        thickness = 2
        (text_width, text_height), _ = cv2.getTextSize(text, font, scale, thickness)
        x = max(10, min(seg.shape[1] - text_width - 12, int(base_x)))
        y = max(text_height + 12, int(base_y) - 10)

        cv2.rectangle(
            seg,
            (x - 8, y - text_height - 8),
            (x + text_width + 8, y + 8),
            (0, 255, 0),
            -1,
        )
        cv2.putText(
            seg,
            text,
            (x, y),
            font,
            scale,
            (0, 0, 0),
            cv2.LINE_AA,
        )

    if use_rectangle:
        for i, (x1, y1, x2, y2, conf) in enumerate(detections):
            cv2.rectangle(seg, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cx = int((x1 + x2) / 2)
            draw_label(i, int(cx) - 60, int(y1) - 10)
    else:
        for i, (x1, y1, x2, y2, conf) in enumerate(detections):
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            radius = max(int(max(x2 - x1, y2 - y1) / 2), 8)
            cv2.circle(seg, (int(cx), int(cy)), radius + 5, (0, 0, 255), 2)
            draw_label(i, int(cx) - 60, int(cy) - radius - 10)
    return seg

def build_panel(imgs, titles):
    h, w = imgs[0].shape[:2]
    title_bar = 101
    total_width = w * len(imgs)

    panel = np.ones((h + title_bar, total_width, 3), dtype=np.uint8) * 255

    for i, (img, title) in enumerate(zip(imgs, titles)):
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        panel[title_bar:, i * w:(i + 1) * w] = img
        cv2.putText(panel, title, (i * w + 10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    return panel