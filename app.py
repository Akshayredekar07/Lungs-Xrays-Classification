import streamlit as st
import torch
import timm
from torchvision import transforms
from PIL import Image
import logging
import os
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger("streamlit.watcher").setLevel(logging.ERROR)  # Suppress Streamlit watcher warnings

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Define transforms for inference
val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # EfficientNet-B0 expects 224x224
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), "models", "pneumonia_model.pth")
logger.info(f"Resolved model path: {model_path}")
if not os.path.exists(model_path):
    logger.error(f"Model file not found at {model_path}")
    raise FileNotFoundError(f"Model file not found at {model_path}")
try:
    model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=1, drop_rate=0.3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    logger.info(f"Model loaded successfully from {model_path}")
except Exception as e:
    logger.error(f"Failed to load model from {model_path}: {e}")
    raise RuntimeError(f"Failed to load model: {e}")

# Function to predict on an image
def predict_image(image):
    try:
        if image is None:
            logger.error("No image provided")
            return "Error: No image provided.", ""
        
        # Process image
        logger.info("Converting image to RGB and applying transforms")
        image = image.convert('RGB')
        image_tensor = val_test_transforms(image)
        
        # Log image tensor stats for debugging
        logger.info(f"Image tensor shape: {image_tensor.shape}")
        logger.info(f"Image tensor mean: {image_tensor.mean().item():.4f}, std: {image_tensor.std().item():.4f}")
        
        image_tensor = image_tensor.unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            output = model(image_tensor)
            logger.info(f"Raw model output: {output.item():.4f}")
            prob = torch.sigmoid(output).item()
            logger.info(f"Sigmoid probability: {prob:.4f}")
            prediction = 'Pneumonia' if prob > 0.5 else 'Normal'
            prob_percent = prob * 100
        
        logger.info(f"Prediction: {prediction} (Probability: {prob_percent:.2f}%)")
        return prediction, f"{prob_percent:.2f}%"
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return f"Error: {str(e)}", ""

# Streamlit interface
def main():
    st.title("Pneumonia Detection")
    st.write("Upload a chest X-ray image to predict if it shows pneumonia or is normal. Probability is shown as a percentage.")

    # File uploader for image
    uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Load and display the image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            # Make prediction
            prediction, probability = predict_image(image)
            
            # Display results
            st.subheader("Prediction")
            st.write(prediction)
            st.subheader("Probability")
            st.write(probability)
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main()