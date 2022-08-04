import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image # Load images
import cv2 #os

import tensorflow as tf
from tensorflow import keras
import base64
from keras.models import load_model



file_ = open('cAD_DETECTIONv2.gif', 'rb')
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")
file_.close()

st.markdown(
    f'<img src="data:image/gif;base64,{data_url}" alt="TB Detection Logo">',
    unsafe_allow_html=True,
)

#video_file = open('cAD_DETECTION.gif', 'rb')
#video_bytes = video_file.read()
#st.video(video_bytes)

#st.markdown("![Alt Text](cAD_DETECTION.gif)")

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
                #st.write('1',img_array.shape)
                img = tf.image.resize(img_array, [256,256])
                #st.write('2',img.shape)
                img = tf.expand_dims(img, axis=0)
                #st.write('3',img.shape)
            
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
        if choice == 'About':
            st.subheader("Tuberculosis (TB)")
            st.markdown("""A potentially serious infectious bacterial disease that mainly affects the lungs. \n\nThe bacteria that cause TB are spread when an infected person coughs or sneezes.            
            \n\nMost people infected with the bacteria that cause tuberculosis don't have symptoms. When symptoms do occur, they usually include: \n*a cough (sometimes blood-tinged), \n*weight loss, \n*night sweats and fever. 
            \n\nTreatment isn't always required for those without symptoms. Patients with active symptoms will require a long course of treatment involving multiple antibiotics.""")
            
            st.subheader("Why this app")
            st.markdown("""To reduce the spread of TB among mining communities, their families, and the broader population, we must continue our efforts to screen, diagnose, and effectively treat miners for TB.""")
    
if __name__ == '__main__':
    main()



















