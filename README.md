<div id="top"></div>
<div align="center">

[![python](https://img.shields.io/badge/Python-3.10-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
<img src="https://img.shields.io/badge/FastAPI-0.89.1-red" alt="FastAPI - FastAPI">
<img src="https://img.shields.io/badge/gunicorn-20.1.0-green" alt="gunicorn - gunicorn">
<img src="https://img.shields.io/badge/Ultralytics-Yolov8-blue" alt="Ultralytics - Yolov8">
[![GitHub](https://badgen.net/badge/icon/github?icon=github&label)](https://github.com)

</div>
<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/Falcons-ai/weapon_detection_fastapi_open">
    <img src="assets/falcons-logo2.png" alt="Logo" >
  </a>
</div>


# Weapon Detection Model on FastAPI

This repository contains a YOLOv8-based model that has been fine-tuned for weapon detection. The model is served using FastAPI, a modern Python web framework, making it easy to deploy and use for real-time weapon detection applications.

## Introduction

The YOLOv8 (You Only Look Once) model is a popular real-time object detection algorithm, and it has been fine-tuned to detect weapons in images and videos. This repository provides a pre-trained model that is ready for use via a FastAPI-based web service.
- This repository is based on the model trained from the code in this repository:
https://github.com/Falcons-ai/weapons_detection_trainer_yolov8_open

## Prerequisites

Before using this repository, you should have the following prerequisites:

- Python 3.6 or higher
- PyTorch
- FastAPI
- OpenCV (for image processing)
- GPU (recommended for real-time performance)

You can install the required Python packages by running:

```bash
pip install -r requirements.txt
```

## Getting Started

1. Clone this repository:
```bash
git clone https://github.com/Falcons-ai/weapon_detection_fastapi_open
cd weapon_detection_fastapi_open
```

2. Install the requirements:
```bash
pip install -r requirements.txt
```

3. Run the FastAPI server:
```bash
uvicorn main:app --reload
```

The server will start on `http://localhost:8000`.

## Usage

To use the weapon detection model, you can make POST requests to the FastAPI server. Here is an example using Python requests:

```python
import requests

# Replace with the actual image file path
image_path = "path/to/your/image.jpg"

url = "http://localhost:8000docs#/default/img_object_detection_to_img_detect_weapon_post"
files = {"image": open(image_path, "rb")}
response = requests.post(url, files=files)

if response.status_code == 200:
    result = response.json()
    print("Weapon Detected:", result["weapon_detected"])
    print("Bounding Boxes:", result["bounding_boxes"])
else:
    print("Error:", response.text)
```

## API Documentation

The FastAPI server exposes the following endpoints:

- `POST /default/img_object_detection_to_img_detect_weapon_post`: Upload an image for weapon detection. It returns the detection result, including whether a weapon is detected and bounding boxes.

You can access the interactive API documentation at `http://localhost:8000/docs`.


## License

This repository is licensed under the MIT License. You are free to use and modify the code for your weapon detection applications. However, make sure to review the YOLOv8 and FastAPI licenses as well.

Please note that weapon detection models have legal and ethical implications. Ensure you comply with all relevant laws and regulations when using this technology.

For more information and support, feel free to contact us at [https://falcons.ai].

Happy weapon detection! 🚀
