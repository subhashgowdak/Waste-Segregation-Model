import ultralytics
import cv2
import matplotlib.pyplot as plt
import roboflow

# Initialize Roboflow and download dataset
rf = roboflow.Roboflow(api_key="YOUR_API_KEY") # Replace with your actual API key
project = rf.workspace("YOUR_WORKSPACE").project("waste-segregation") # Replace with your actual workspace and project names
version = project.version(1) # Replace with your actual version number
dataset = version.download("yolov8") # Download in YOLOv8 format

# Load a pretrained YOLOv8 model
model = ultralytics.YOLO("yolov8n.pt")

# Train on your dataset
model.train(data="datasets/waste-detection-1/data.yaml", epochs=50, imgsz=640, batch=16, name="waste-detection-Prototype") #adjust parameters as needed