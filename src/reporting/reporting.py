import json
import os
from collections import Counter


def build_detection_report(image_path, detections, labels):
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    labels = labels or ["other"] * len(detections)

    if not detections:
        report = {
            "image_name": image_name,
            "detections": 0,
            "most_detection_type": "none",
            "specific_detections": [],
        }
        return report

    counts = Counter(labels)
    most_detection_type = max(counts.items(), key=lambda item: (item[1], item[0]))[0]
    specific_detections = [
        {f"detection_{index}": label}
        for index, label in enumerate(labels, start=1)
    ]

    return {
        "image_name": image_name,
        "detections": len(detections),
        "most_detection_type": most_detection_type,
        "specific_detections": specific_detections,
    }


def save_detection_report(report, output_dir="output-json"):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{report['image_name']}.json")
    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(report, json_file, ensure_ascii=False, indent=2)
    return output_path
