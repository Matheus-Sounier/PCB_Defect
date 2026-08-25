# PCB Defect Detection

This project detects defects in printed circuit boards (PCBs) using a YOLO model and image processing with OpenCV. It loads an input image, runs object detection, generates a heat map, creates a mask, and saves a JSON report with the detected defect information.

### Project

<img src="./output-img/result12.jpg" width="1200"/>

## Required Folders

Before running the project, make sure the following folders exist:

- `imgs/` - input images
- `models/` - YOLO model file (`.pt`)
- `output-img/` - output visualization images
- `output-json/` - generated JSON reports

If these folders are missing, create them manually before execution.

## Dependencies

### Production dependencies

Install the runtime dependencies to run the application:

```bash
pip install -r requirements-prod.txt
```

### Development dependencies

Install development dependencies if you want to run tests and work on the project:

```bash
pip install -r requirements-dev.txt
```

## Train Your Own Model

This project expects a custom YOLO model file to be placed in the `models/` folder. If no `best.pt` is available, you can train your own model first.

### 1. Download a dataset

Instead of creating the dataset structure manually, you can use a PCB defect dataset already available online and adapt it to the YOLO format.

Recommended options:

- Ultralytics dataset: https://platform.ultralytics.com/muhammadrizwanmunawar/datasets/pcb-defects-detection
- Roboflow dataset: https://universe.roboflow.com/test-jksby/pcb_defect-9cisw

After downloading, organize the project so that the dataset is stored in a folder such as:

```text
dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── data.yaml
```

### 2. Train the model

Example using Python:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.train(
    data="dataset/data.yaml",
    epochs=25,
    imgsz=640,
    batch=16,
    project="pcb_defect",
    name="training"
)
```

Example using the command line:

```bash
yolo detect train \
    model=yolov8n.pt \
    data=dataset/data.yaml \
    epochs=25 \
    imgsz=640
```

### 3. Save the trained model in the expected folder

After training, the generated file will usually be located in a folder similar to:

```text
runs/detect/training/weights/best.pt
```

Copy or move it to:

```text
models/best.pt
```

Then you can run the project normally.

### 4. Resume or continue training

Resume from the last checkpoint:

```python
from ultralytics import YOLO

model = YOLO("runs/detect/pcb_defect_training/weights/last.pt")
results = model.train(resume=True)
```

Continue training for more epochs:

```python
from ultralytics import YOLO

model = YOLO("runs/detect/pcb_defect_training/weights/best.pt")

results = model.train(
    data="dataset/data.yaml",
    epochs=100,
    project="pcb_defect",
    name="training_100epochs"
)
```

## How to Run

Run the project from the root folder:

```bash
python main.py --model models/best.pt --image imgs/your-image.jpg --output output-img/result.png --conf 0.1 --shape rectangle
```

### Arguments

- `--model`: path to the YOLO model file
- `--image`: path to the input image
- `--output`: output image path
- `--conf`: minimum confidence threshold for detections
- `--mask-threshold`: threshold used to generate the mask
- `--shape`: output shape for annotations (`rectangle` or `circle`)

Example:

```bash
python main.py --model models/best.pt --image imgs/test1.jpg --output output-img/test1.png
```

Since the `imgs/` folder is available in the repository, the user can normally run the project by pointing to an image inside that folder, for example:

## Output

The project generates:

- a processed image in `output-img/`
- a report in JSON format in `output-json/`

The JSON file contains information such as:

- image name
- total number of detections
- most frequent defect type
- list of detected defect labels

## CI/CD

This repository includes a continuous integration workflow in `.github/workflows/ci.yml`.

The workflow performs:

- Python installation
- dependency installation from `requirements-dev.txt`
- unit test execution
- wheel build
- integration tests on the `main` and `dev` branches

This ensures that changes in the project are validated automatically in GitHub Actions.