import os
import base64
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
import cv2
import requests

def set_background(image_file):
    """
    This function sets the background of a Streamlit app to an image specified by the given image file.

    Parameters:
        image_file (str): The path to the image file to be used as the background.

    Returns:
        None
    """
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)

set_background('./bg.png')

plt.style.use("ggplot")

def dice_coefficients(y_true, y_pred, smooth=100):
    y_true_flatten = K.flatten(y_true)
    y_pred_flatten = K.flatten(y_pred)

    intersection = K.sum(y_true_flatten * y_pred_flatten)
    union = K.sum(y_true_flatten) + K.sum(y_pred_flatten)
    return (2 * intersection + smooth) / (union + smooth)


def dice_coefficients_loss(y_true, y_pred, smooth=100):
    return -dice_coefficients(y_true, y_pred, smooth)


def iou(y_true, y_pred, smooth=100):
    intersection = K.sum(y_true * y_pred)
    sum = K.sum(y_true + y_pred)
    iou = (intersection + smooth) / (sum - intersection + smooth)
    return iou


def jaccard_distance(y_true, y_pred):
    y_true_flatten = K.flatten(y_true)
    y_pred_flatten = K.flatten(y_pred)
    return -iou(y_true_flatten, y_pred_flatten)


# Main Streamlit app code
st.title("Brain MRI Segmentation App")


# Load the segmentation model
model = load_model("ResUNet-segmodel-brain-mri-v9.h5", custom_objects={
        'dice_coef_loss': dice_coefficients_loss, 'iou': iou, 'dice_coef': dice_coefficients})

im_height = 256
im_width = 256

# File uploader and prediction
file = st.file_uploader("Upload file", type=["csv", "png", "jpg"], accept_multiple_files=True)
if file:
    for i in file:
        st.header(f"Original Image: {i.name}")
        try:
            st.image(i)
            content = i.getvalue()
            image = np.asarray(bytearray(content), dtype="uint8")
            st.write(f"Image array shape: {image.shape}")
            
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            img2 = cv2.resize(image, (im_height, im_width))
            img3 = img2 / 255.0
            img4 = img3[np.newaxis, :, :, :]
            st.write(f"Prepared image shape: {img4.shape}")

            if st.button(f"Predict Output for {i.name}:"):
                pred_img = model.predict(img4)
                st.header("Predicted Image:")
                st.image(pred_img)
        except Exception as e:
            st.error(f"An error occurred: {e}")
