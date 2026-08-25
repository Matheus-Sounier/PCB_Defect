from src.reporting.reporting import build_detection_report

def test_build_detection_report_uses_english_keys_and_labels():
    report = build_detection_report(
        "test5.png",
        [
            (0, 0, 10, 10, 0.99),
            (20, 0, 30, 10, 0.95),
            (40, 0, 50, 10, 0.93),
            (60, 0, 70, 10, 0.91),
        ],
        ["damaged trace"] * 4,
    )

    assert report["image_name"] == "test5"
    assert report["detections"] == 4
    assert report["most_detection_type"] == "damaged trace"
    assert report["specific_detections"][0]["detection_1"] == "damaged trace"
    assert report["specific_detections"][1]["detection_2"] == "damaged trace"

def test_build_detection_report_no_detections():
    report = build_detection_report("img.png", [], [])

    assert report["image_name"] == "img"
    assert report["detections"] == 0
    assert report["most_detection_type"] == "none"
    assert report["specific_detections"] == []

def test_build_detection_report_labels_none_defaults_to_other():
    detections = [(0, 0, 1, 1, 0.5)]
    report = build_detection_report("some.jpg", detections, None)

    assert report["detections"] == 1
    assert report["most_detection_type"] == "other"
    assert report["specific_detections"][0]["detection_1"] == "other"