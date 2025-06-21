


import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from classifier import AgeClassifier
from config import Config
from torchvision.models import ResNet50_Weights
import numpy as np


@st.cache_resource
def load_model():
    cfg = Config()
    model = AgeClassifier(weights=ResNet50_Weights.DEFAULT).to(torch.device('cpu'))
    model.load_state_dict(torch.load(cfg.MODEL_SAVE_PATH, map_location='cpu'))
    model.eval()
    return model


def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)


def decode_age_from_outputs(outputs):
    probs = torch.sigmoid(outputs).squeeze()
    return int(torch.sum(probs > 0.5).item())

def predict_age(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        predicted_age = decode_age_from_outputs(outputs)
    return predicted_age


def main():
    st.title("Age Classification App")
    st.write("Upload an image to estimate the person's age")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_container_width=True)

        model = load_model()
        image_tensor = preprocess_image(image)
        predicted_age = predict_age(model, image_tensor)
        st.success(f"Predicted Age: {predicted_age} years")

        

if __name__ == "__main__":
    main()
