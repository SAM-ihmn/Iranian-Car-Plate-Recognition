import os
import cv2
import torch
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device, TracedModel
from utils.datasets import letterbox

# Load environment variables from .env file
load_dotenv()

# Set the input image size for the models
image_size = 640

# Select device ('cpu' or 'cuda')
device = select_device('cpu')

# Load model weights paths from environment variables
lpd_weight = os.getenv('LPD_WEIGHT_PATH')
lpr_weight = os.getenv('LPR_WEIGHT_PATH')

# Attempt to load License Plate Detection (LPD) model
lpd_model = attempt_load(lpd_weight, map_location=device)
stride = int(lpd_model.stride.max())
imgsz = check_img_size(image_size, s=stride)
lpd_model = TracedModel(lpd_model, device, image_size)

# Attempt to load License Plate Recognition (LPR) model
lpr_model = attempt_load(lpr_weight, map_location=device)
stride = int(lpr_model.stride.max())
imgsz = check_img_size(image_size, s=stride)
lpr_model = TracedModel(lpr_model, device, image_size)

# Function to detect license plates in a frame
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

# Function to crop the detected plate region from an image
def crop(image, coord):
    return image[int(coord[1]):int(coord[3]), int(coord[0]):int(coord[2])]

# Function to recognize characters on a license plate image
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
    # Mapping of character labels to their respective characters
    chars = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 
             6: "6", 7: "7", 8: "8", 9: "9", 10: "B", 11: "D", 
             12: "Gh", 13: "H", 14: "J", 15: "L", 16: "M", 17: "N", 
             18: "Sad", 19: "Sin", 20: "T", 21: "V", 22: "Y"}

    # Convert rectangles data to a DataFrame
    df = pd.DataFrame(rects, columns=["label", "x_center", "y_center", "width", "height"])

    # Sort rectangles by x_center for sequential character order
    df = df.sort_values("x_center")

    # Initialize an empty string to store the extracted number plate
    number_plate = ""
    for k in df["label"]:
        number_plate += chars[k]
    return number_plate

# Open the webcam (adjust the index if you have multiple cameras)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Webcam did not open.")
    exit()

# Main loop to capture frames from the webcam and perform license plate detection
while cap.isOpened():
    # Read a frame from the webcam
    ret, frame = cap.read()
    
    # Check if frame is successfully read
    if not ret:
        break

    # Detect license plates in the current frame
    plate_detections = detect_plate(frame)
    
    # Iterate through each detected plate
    for coords in plate_detections:
        # Crop the detected plate region from the frame
        plate_region = crop(frame, coords)
        
        # Recognize characters on the cropped plate region
        rects = recognize_plate(plate_region)
        
        # Extract the number plate from recognized characters
        number_plate = extract_number_plate(rects)
        print("Detected number plate:", number_plate)
        
        # Draw bounding box and number plate text on the original frame
        cv2.rectangle(frame, (coords[0], coords[1]), (coords[2], coords[3]), (0, 255, 0), 2)
        cv2.putText(frame, number_plate, (coords[0], coords[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    
    # Display the frame with overlaid detections and recognized number plate
    cv2.imshow('Webcam - Plate Detection', frame)

    # Check for 'q' key press to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
