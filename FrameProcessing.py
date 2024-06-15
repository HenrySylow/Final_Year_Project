import cv2
import csv
import torch
from ultralytics import YOLO
import numpy as np
import os



# Function to apply noise reduction techniques
def apply_noise_reduction(image, method='none'):
    if method == 'g':
        return cv2.GaussianBlur(image, (5, 5), 0)
    elif method == 'm':
        return cv2.medianBlur(image, 5)
    elif method == 'b':
        return cv2.bilateralFilter(image, 9, 75, 75)
    elif method == 'n':
        return image  # No noise reduction
    else:
        raise ValueError(f"Unknown noise reduction method: {method}")
    
# Function to generate a unique filename based on given parameters
def generate_filename(image_size, noise_reduction_type, model_name, video_name):
    model_name = model_name.replace('.pt', '')  # Remove the .pt extension
    filename = f"{image_size}_{noise_reduction_type}_{model_name}_{video_name}.csv"
    return filename


# Parameters
image_size = 640
noise_reduction_type = 'b'  # Options: 'none', 'gaussian', 'median', 'bilateral'
model_path = r"C:\Users\henry\Documents\3_Coding\Test\Models\aug_data_yolov8s_100_epch.pt"
video_path = r'C:\Users\henry\Documents\3_Coding\Belt ML Captures\CV004\capture 5.h264'
video_name = 'CV004_capture5'
max_frame_parameter = 500 #Cap 2 = 100 Cap 3 = 1250 Cap 5 = 400 CV108 = 400
Image_saving_number = 500

# Initialize the YOLOv8 model and ensure it runs on the GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO(model_path).to(device)

# Define the class names mapping
class_names = model.names  # This should be available in the model object
print("Class names:", class_names)  # Debug print to ensure class names are loaded

# Generate the CSV filename
csv_filename = generate_filename(image_size, noise_reduction_type, os.path.basename(model_path), video_name)
print(f"Results will be saved to: {csv_filename}")

# Generate the CSV filename if max_frame_parameter is 3000 or more
if max_frame_parameter > Image_saving_number:
    csv_filename = generate_filename(image_size, noise_reduction_type, os.path.basename(model_path), video_name)
    csv_dir = '1_Data_CSV'
    os.makedirs(csv_dir, exist_ok=True)
    csv_filepath = os.path.join(csv_dir, csv_filename)
    print(f"Results will be saved to: {csv_filepath}")
else:
    csv_filepath = None

# Create a directory to save output images if max_frame_parameter is less than 3000
if max_frame_parameter <= Image_saving_number:
    output_dir = os.path.join('output_images', f"{image_size}_{noise_reduction_type}_{video_name}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output images will be saved to: {output_dir}")

# Open the .h264 video file
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error opening video stream or file")

# Open the CSV file to log the results if max_frame_parameter is 3000 or more
if csv_filepath:
    file = open(csv_filepath, mode='w', newline='')
    writer = csv.writer(file)
    writer.writerow(['Frame', 'Confidence Level', 'Classes'])

frame_number = 0
max_frames = max_frame_parameter  # Set the maximum number of frames to process

# Read until video is completed or the maximum frame count is reached
while cap.isOpened() and frame_number < max_frames:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame
    frame_resized = cv2.resize(frame, (image_size, image_size))

    # Apply noise reduction
    frame_processed = apply_noise_reduction(frame_resized, method=noise_reduction_type)

    if max_frame_parameter <= Image_saving_number:
        # Save the processed frame
        processed_frame_path = os.path.join(output_dir, f"processed_frame_{frame_number:04d}.jpg")
        cv2.imwrite(processed_frame_path, frame_processed)

        # Perform detection on the saved frame
        results = model.predict(source=processed_frame_path, save=True, imgsz=image_size, device=device)
    else:
        # Perform detection directly on the processed frame
        results = model.predict(source=frame_processed, save=False, imgsz=image_size, device=device)

        # Initialize variables for logging
        lowest_confidence = 1.0  # Start with the highest possible confidence
        detected_classes = []

        # Process and log results
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2, confidence, class_id = box.data.cpu().numpy().flatten()
                class_name = class_names[int(class_id)] if int(class_id) in class_names else f"Unknown({int(class_id)})"
                confidence = float(confidence)
                detected_classes.append(class_name)
                if confidence < lowest_confidence:
                    lowest_confidence = confidence

        # If no detections, set confidence to 0.0 and classes to an empty list
        if not detected_classes:
            lowest_confidence = 0.0

        # Write results to CSV
        writer.writerow([frame_number, lowest_confidence, str(detected_classes)])

    frame_number += 1

# Release the video capture object
cap.release()

if csv_filepath:
    file.close()

print(f"Processing completed.")
if csv_filepath:
    print(f"Results saved to {csv_filepath}.")