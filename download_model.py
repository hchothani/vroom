from ultralytics import YOLO
import os

# Create a models directory
os.makedirs('models', exist_ok=True)

# Download YOLOv8n model
print("Downloading YOLOv8n model...")
model = YOLO('yolov8n.pt')
model.model.save('models/yolov8n.pt')

print("YOLOv8n model downloaded successfully to 'models' directory.")

