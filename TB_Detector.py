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


def main():
    # set the title
    st.title("Tuberculosis (TB) Detector") 
    
    menu = ('Home', 'About')
    choice = st.sidebar.selectbox('Menu', menu)
    
    if choice == 'Home':
        #st.subheader('TB Detection')
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
                st.subheader('Original X-ray')
                #st.image(load_image(image_file), width=255)
                
                st.image(
                    image_file,
                    caption=file.name, #f"You amazing image has shape",
                    use_column_width=True,
                )
                

                img_array = np.array(image_file)
                st.write('1',img_array.shape)
                img = tf.image.resize(img_array, [256,256])
                st.write('2',img.shape)
                img = tf.expand_dims(img, axis=0)
                st.write('3',img.shape)
            
            with col2:
                # Reset cursor for Upload NoneType
                image_file.seek(0)
                st.subheader('Detection response')
                
                # Load model
                new_model = load_model('tb_detection_model.h5')
                # Image classification with model
                img_array = np.array(image_file)
                img = tf.image.resize(img_array, [256,256])
                #img = tf.image.resize(img_array, size=(256,256,3))
                img = tf.expand_dims(img, axis=0)
                y_hat = new_model.predict(img)
                
                if y_hat > 0.5:
                  st.write('Predicted class is TB')
                else:
                  st.write('Predicted class is Normal')    
    else:
        choice == 'About'
    
if __name__ == '__main__':
    main()



















