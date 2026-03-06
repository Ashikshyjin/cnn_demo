import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

# Load trained model
model = tf.keras.models.load_model("image_classifier.h5")

# Class names
class_names = ['bike', 'cars', 'cats', 'dogs', 'flowers', 'horses', 'human']

# Title
st.title("📷 Image Classification using CNN")
st.write("Upload an image and the model will predict its category.")

# Sidebar
st.sidebar.header("About Project")
st.sidebar.info(
    "This project uses a Convolutional Neural Network (CNN) "
    "trained to classify images into 7 categories."
)

st.sidebar.write("Availbale datas:")
st.sidebar.write("Bike")
st.sidebar.write("Dog")
st.sidebar.write("Cat")
st.sidebar.write("Horse")
st.sidebar.write("Human")
st.sidebar.write("Flower")
st.sidebar.write("Car")

# Upload image
uploaded_file = st.file_uploader("Upload an Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width="stretch")

    # Get model input size automatically
    img_size = model.input_shape[1]

    # Preprocess image
    img = image.convert("RGB")
    img = img.resize((img_size, img_size))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)


    # Prediction
    prediction = model.predict(img)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.success(f"Prediction: **{predicted_class}**")
    st.write(f"Confidence: **{confidence*100:.2f}%**")