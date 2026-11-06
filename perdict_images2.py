import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = ["AI","Human"]

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# Load model
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features,2)

model.load_state_dict(torch.load("resnet_ai_art_model.pth", map_location=device))

model = model.to(device)
model.eval()

folder = "test_images2"

correct = 0
total = 0

# Read only image files
images = [img for img in os.listdir(folder) if img.endswith((".jpg",".jpeg",".png"))]
images.sort()

for img_name in images:

    img_path = os.path.join(folder,img_name)

    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        _,predicted = torch.max(output,1)

    predicted_class = class_names[predicted.item()]

    if "ai" in img_name.lower():
        actual_class = "AI"
    elif "real" in img_name.lower() or "human" in img_name.lower():
        actual_class = "Human"
    else:
        actual_class = "Unknown"

    if actual_class != "Unknown":
        total += 1
        if predicted_class == actual_class:
            correct += 1

if total > 0:
    accuracy = (correct / total) * 100
    print("Total Images Tested:", total)
    print("Correct Predictions:", correct)
    print("Accuracy:", round(accuracy,2), "%")