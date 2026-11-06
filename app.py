# python -m streamlit run "c:/Users/surya/OneDrive/Documents/AI_Art_Detector/app.py"
import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

# --------------------
# Model Loading
# --------------------
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ResNet18 model
    model = models.resnet18(weights=None)  # Avoid pretrained warning
    model.fc = nn.Linear(model.fc.in_features, 2)

    # Load trained weights
    model.load_state_dict(torch.load("resnet_ai_art_model.pth", map_location=device))
    model = model.to(device)
    model.eval()

    return model, device

model, device = load_model()

# --------------------
# Image Transform
# --------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

class_names = ["AI", "Human"]

# --------------------
# Streamlit UI
# --------------------
st.title("AI Art Detector")
st.write("Upload an image to check if it is AI-generated or Human-generated.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, width=500)  # Updated parameter instead of deprecated use_column_width

    # Preprocess image
    input_tensor = transform(img).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        predicted_class = class_names[predicted.item()]

    st.success(f"Prediction: **{predicted_class}**")