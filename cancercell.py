import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the trained model (Replace with your actual model path)
MODEL_PATH = 'cancer_model.h5'  # Ensure you have a trained model saved here
model = load_model(MODEL_PATH)

# Function to preprocess image
def preprocess_image(img):
    img = img.resize((224, 224))  # Resize to match model input size
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize pixel values
    return img

# Streamlit UI
st.title("Cancer Cell Classification")
st.write("Upload a cell image to classify it as benign or malignant.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    if st.button("Classify Cell"):
        processed_img = preprocess_image(img)
        prediction = model.predict(processed_img)
        class_label = "Malignant" if prediction[0][0] > 0.5 else "Benign"
        confidence = prediction[0][0] if class_label == "Malignant" else 1 - prediction[0][0]
        
        st.write(f"**Prediction:** {class_label}")
        st.write(f"**Confidence:** {confidence:.2f}")
