import argparse

def normalize_shape(value):
    aliases = {"circle": "circle", "rectangle": "rectangle"}
    normalized = aliases.get(value)
    if normalized is None:
        raise argparse.ArgumentTypeError(f"Invalid shape: {value}")
    return normalized

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to the YOLO model (.pt)")
    parser.add_argument("--image", required=True, help="Path to the image to be tested")
    parser.add_argument("--output", default="result.png", help="Output file path")
    parser.add_argument("--conf", type=float, default=0.1, help="Minimum detection confidence")
    parser.add_argument("--mask-threshold", "--limiar-mascara", dest="mask_threshold", type=float, default=0.4)
    parser.add_argument("--shape", "--forma", dest="shape", type=normalize_shape, choices=["circle", "rectangle"], default="rectangle")
    return parser