import os
from dotenv import load_dotenv
import cv2
import torch
import numpy as np
import pandas as pd
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
from utils.torch_utils import select_device, TracedModel
from utils.datasets import letterbox
from utils.plots import plot_one_box

# Load environment variables from .env file
load_dotenv()

# Initialize parameters and models
image_size = 640
device = select_device('cpu')

lpd_weight = os.getenv('LPD_WEIGHT_PATH')
lpr_weight = os.getenv('LPR_WEIGHT_PATH')
image_path = os.getenv('IMAGE_PATH')

lpd_model = attempt_load(lpd_weight, map_location=device)
stride = int(lpd_model.stride.max())
imgsz = check_img_size(image_size, s=stride)
lpd_model = TracedModel(lpd_model, device, image_size)

lpr_model = attempt_load(lpr_weight, map_location=device)
stride = int(lpr_model.stride.max())
imgsz = check_img_size(image_size, s=stride)
lpr_model = TracedModel(lpr_model, device, image_size)

# Function to detect plates in an image
def detect_plate(frame):
    img = letterbox(frame, image_size, stride=stride)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    with torch.no_grad():
        pred = lpd_model(img, augment=True)[0]
    pred = non_max_suppression(pred, 0.25, 0.45, classes=0, agnostic=True)

    plate_detections = []
    for det in pred:
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, conf, cls in reversed(det):
                coords = [int(position) for position in (torch.tensor(xyxy).view(1, 4)).tolist()[0]]
                plate_detections.append(coords)
    return plate_detections

# Function to crop the detected plate region
def crop(image, coord):
    return image[int(coord[1]):int(coord[3]), int(coord[0]):int(coord[2])]

# Function to recognize characters on the plate
def recognize_plate(plate_image):
    img = letterbox(plate_image, imgsz, stride=stride)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    with torch.no_grad():
        pred = lpr_model(img, augment=True)[0]
    pred = non_max_suppression(pred, 0.4)
    
    lines = []
    for det in pred:
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], plate_image.shape).round()
            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / torch.tensor(plate_image.shape)[[1, 0, 1, 0]]).view(-1).tolist()
                lines.append((int(cls), *xywh))
                label = f'{cls}'
                plot_one_box(xyxy, plate_image, label=label, color=[0, 255, 0], line_thickness=1)
    return lines

# Function to extract the number plate from recognized characters
def extract_number_plate(rects):
    chars = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 
             6: "6", 7: "7", 8: "8", 9: "9", 10: "B", 11: "D", 
             12: "Gh", 13: "H", 14: "J", 15: "L", 16: "M", 17: "N", 
             18: "Sad", 19: "Sin", 20: "T", 21: "V", 22: "Y"}

    df = pd.DataFrame(rects, columns=["label", "x_center", "y_center", "width", "height"])

    df = df.sort_values("x_center")

    number_plate = ""
    for k in df["label"]:
        number_plate += chars[k]
    return number_plate

# Main function to process an image and detect number plates
def detect_number_plate(image_path):
    frame = cv2.imread(image_path)

    if frame is None:
        print(f"Error: Unable to read image from {image_path}")
        return
    
    plate_detections = detect_plate(frame)
    for coords in plate_detections:
        plate_region = crop(frame, coords)
        rects = recognize_plate(plate_region)
        number_plate = extract_number_plate(rects)
        print("Detected number plate:", number_plate)
        cv2.rectangle(frame, (coords[0], coords[1]), (coords[2], coords[3]), (0, 255, 0), 2)
        cv2.putText(frame, number_plate, (coords[0], coords[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    
    cv2.imshow('Image - Plate Detection', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    detect_number_plate(image_path)

