import cv2
import numpy as np
from ultralytics import YOLO

from src.detection import (
    build_panel,
    generate_heatmap,
    generate_mask,
    generate_segmentation,
    get_yolo_labels,
    heatmap_to_bgr,
)
from src.reporting import build_detection_report, save_detection_report
from src.utils.cli import build_parser

def main():
    args = build_parser().parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Could not open the image: {args.image}")

    model = YOLO(args.model)
    results = model.predict(source=args.image, conf=args.conf, verbose=False)[0]

    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        detections.append((x1, y1, x2, y2, conf))

    print(f"{len(detections)} detection(s) found.")

    heatmap = generate_heatmap(image.shape, detections)
    heatmap_bgr = heatmap_to_bgr(heatmap)
    heatmap_overlay = cv2.addWeighted(image, 0.5, heatmap_bgr, 0.5, 0)
    mask = generate_mask(heatmap, threshold=args.mask_threshold)

    labels = get_yolo_labels(results) if detections else ["other"]

    if labels:
        print(f"Main defect type (YOLO): {labels[0]}")
        for i, label in enumerate(labels, start=1):
            print(f"  Detection {i}: {label}")

    report = build_detection_report(args.image, detections, labels)
    report_path = save_detection_report(report)
    print(f"JSON report saved to: {report_path}")

    segmentation = generate_segmentation(image, detections, labels=labels, use_rectangle=(args.shape == "rectangle"))
    panel = build_panel(
        [image, heatmap_overlay, mask, segmentation],
        ["Image", "Heat Map", "Mask", "Segmentation"],
    )

    cv2.imwrite(args.output, panel)
    print(f"Result saved to: {args.output}")

if __name__ == "__main__":
    main()