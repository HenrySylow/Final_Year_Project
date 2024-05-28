from clearml import Task, Dataset
from ultralytics import YOLO
import os

# Step 1: Creating a ClearML Task
task = Task.init(project_name="FYP", task_name="yolov8_FYP")

# Step 2: Retrieve the finalized dataset from ClearML
dataset = Dataset.get(dataset_name='roboflow_dataset', dataset_project='FYP')
if dataset is None:
    raise ValueError("Dataset 'roboflow_dataset' not found. Ensure it is finalized and accessible.")

dataset_path = dataset.get_local_copy()

# Log the contents of the dataset directory for debugging
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        print(os.path.join(root, file))

# Step 3: Verify the dataset path
data_config_path = os.path.join(dataset_path, 'data.yaml')  # Adjust this path if necessary
if not os.path.exists(data_config_path):
    raise FileNotFoundError(f"Dataset configuration file not found at {data_config_path}")

# Verify the path
print("Dataset path:", dataset_path)
print("Data config path:", data_config_path)
print("File exists in dataset path:", os.path.exists(data_config_path))

# Step 4: Selecting the YOLOv8 Model
model_variant = "yolov8m"
task.set_parameter("model_variant", model_variant)

# Step 5: Loading the YOLOv8 Model
model = YOLO(f"{model_variant}.pt")

# Step 6: Setting Up Training Arguments
args = dict(data=data_config_path, epochs=100)
task.connect(args)

# Step 7: Initiating Model Training
results = model.train(**args)


# Optionally, evaluate the model
results = model.val(data=data_config_path)
