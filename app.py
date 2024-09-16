import keras
from keras.models import load_model
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image  
import io
import base64

# Function to set background
def set_background(image_path):
    with open(image_path, 'rb') as f:
        image_data = f.read()
    encoded_image = base64.b64encode(image_data).decode()
    
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded_image}");
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set the background image
set_background('Sample/background.jpg')

# Streamlit App Interface
st.markdown('<p style="font-size:70px; color:#ffffff; text-align: center; font-weight:bold;">FLORAL AI</p>', unsafe_allow_html=True)
st.markdown('<p style="font-size:20px; color:#ffffff; text-align: center; font-weight:bold">Identify the Botanical Species with FLORAL AI</p>', unsafe_allow_html=True)
flower_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

model = load_model('Flower_Recog_Model.keras')

# Function to classify images
def classify_images(image_data):
    image = Image.open(io.BytesIO(image_data))
    input_image = image.resize((180, 180))  
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array, 0)
    
    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    
    outcome = 'The Image belongs to ' + flower_names[np.argmax(result)] + ' with a score of ' + str(np.max(result) * 100) + '%'
    
    return outcome

# File uploader for image
uploaded_file = st.file_uploader('Upload an Image', type=['jpg', 'png', 'jpeg'])
if uploaded_file is not None:
    st.image(uploaded_file, width=200)  
    
    image_data = uploaded_file.getvalue()
    
    st.markdown(classify_images(image_data))
