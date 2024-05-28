from clearml import Task
from ultralytics import YOLO
import os

# Step 1: Creating a ClearML Task
task = Task.init(project_name="FYP", task_name="yolov8_FYP", repo="https://github.com/HenrySylow/Final_Year_Project", script="path/to/your_script.py")

# Step 2: Use the existing dataset in Colab
data_config_path = "/content/Conveyor-Belt-11/data.yaml"  # Path to your dataset's data.yaml

# Verify the dataset path
if not os.path.exists(data_config_path):
    raise FileNotFoundError(f"Dataset configuration file not found at {data_config_path}")

# Verify the path
print("Data config path:", data_config_path)
print("File exists in dataset path:", os.path.exists(data_config_path))

# Step 3: Selecting the YOLOv8 Model
model_variant = "yolov8m"
task.set_parameter("model_variant", model_variant)

# Step 4: Loading the YOLOv8 Model
model = YOLO(f"{model_variant}.pt")

# Step 5: Setting Up Training Arguments
args = dict(data=data_config_path, epochs=100)
task.connect(args)

# Step 6: Initiating Model Training
results = model.train(**args)


# Optionally, evaluate the model
results = model.val(data=data_config_path)
