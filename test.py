import cv2 
import torch
from torchvision import transforms

image_path = 'data/vehicle-yolo-dataset/train/images/van99.jpeg'
image = cv2.imread(image_path)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (640, 640))

image_tensor = transforms.ToTensor()(image)

image_tensor = image_tensor.unsqueeze(0).float()

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo='check')
# model = torch.load('yolov8n.pt', weights_only=True)

results = model(image_path)
print(results)

for i, item in enumerate(results):
        print(f"Index {i}: {item}, Type: {type(item)}")

