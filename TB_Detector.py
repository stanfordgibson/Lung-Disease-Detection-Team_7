import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image # Load images
import cv2 #os

import tensorflow as tf
from tensorflow import keras

from keras.models import load_model




video_file = open('Header image.mp4', 'rb')
video_bytes = video_file.read()
st.video(video_bytes)


# Load images
@st.cache
def load_image(image_file):
    img = Image.open(image_file)
    return img
"""    
def classification_machine(image_file):
    # Load model
    new_model = load_model('tb_detection_model.h5')
    # Input shape
    shape = ((256, 256, 3))
    tf.keras.Sequential
"""

def main():
    # set the title
    st.title("Tuberculosis (TB) Detector") 
    
    menu = ('Home', 'About')
    choice = st.sidebar.selectbox('Menu', menu)
    
    if choice == 'Home':
        st.subheader('TB Detection')
        file = st.file_uploader('Upload Chest X-ray', type=['png','jpg','jpeg'])
        
        if file is not None:
            
            image_file = Image.open(file)
            
            # retrieve image details
            st.write(type(file))
            # Methods and attributes on the class
            file_details = {'file name':file.name,
            'file type': file.type, 'file size':file.size}
            # Display details on screen
            st.write(file_details)
            
            # Creating column view
            col1, col2 = st.columns(2)
            
            with col1:
                # Display the image
                st.header('Original X-ray')
                #st.image(load_image(image_file), width=255)
                
                st.image(
                    image,
                    caption=f"You amazing image has shape",
                    use_column_width=True,
                )

                img_array = np.array(image)
                img = tf.image.resize(img_array, size=(224,224))
                img = tf.expand_dims(img, axis=0)
            
            with col2:
                # Reset cursor for Upload NoneType
                image_file.seek(0)
                st.header('Detection space')
                
                # resize the image
                #resized_img = image_file.resize(256,256)
                #keras.preprocessing.image.img_to_array(resized_img)
                #resized_img = np.expand_dims(resized_img, axis=0)
                # make a prediction
                #yhat = new_model.predict(resized_img)   #(np.expand_dims(resized_img/255, 0))
                
                #if yhat > 0.5:
                  #st.write('Predicted class is TB')
                #else:
                  #st.write('Predicted class is Normal')

        
    else:
        choice == 'About'
    
if __name__ == '__main__':
    main()



















