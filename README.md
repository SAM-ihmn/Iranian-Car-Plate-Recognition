# License Plate Detection and Recognition using YOLOv7

## Introduction

This project aims to detect and recognize license plates using YOLOv7, an object detection model, integrated with Python, OpenCV, and PyTorch.

## Setup

### 1. Install Requirements
First, install the required Python packages by running:

```bash
pip install -r requirements.txt
```
### 2. Clone YOLOv7 Repository

Clone the YOLOv7 repository from Augmented Startups:

```bash
git clone https://github.com/augmentedstartups/yolov7.git
```
### 3. Prepare Environment

Move "ImageTest.py" and "WebcamTest.py" to the YOLOv7 repository folder.

Create a ".env" file in the root directory (same level as ImageTest.py and WebcamTest.py).

### 4. Configure Environment Variables
In the .env file, set the following variables:
```bash
LPD_WEIGHT_PATH=/path/to/lpd_weight.pt
LPR_WEIGHT_PATH=/path/to/lpr_weight.pt
IMAGE_PATH=/path/to/your/image.jpg
```
Replace "/path/to/..." with the actual paths to your YOLOv7 weights ("LPD.pt" and "LPR.pt") and any image you want to test with (image.jpg).

## Usage

### Webcam Test

To run the webcam test:
```bash
python WebcamTest.py
```
This script will use your webcam to detect license plates in real-time.


### Image Test

To run the image test:
```bash
python ImageTest.py
```

This script will use the specified image path (IMAGE_PATH in .env) to detect and recognize license plates.

## Notes

Make sure your environment is set up correctly with Python 3.7+ and necessary libraries installed.

Adjust paths in .env file according to your setup before running the scripts.

# Acknowledgments
Special thanks to Sanaz Yari's Persian Car Licence Plate Detection and Recognition repository for invaluable assistance during the development of this project.

https://github.com/sanazy/Persian-Car-Licence-Plate-Detection-and-Recognition/
