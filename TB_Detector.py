import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image # Load images
import os#, cv2

#import tensorflow as tf
#from tensorflow import keras


#import plotly.express as px
#import time
#import copy


# Folders path
vid_dir = 'C:\\Users\\Stanford\\Lung-Disease-Detection-Team_7\\resources'
# List the folders in the path
os.listdir(vid_dir)

#vid = cv2.imread(os.path.join(vid_dir,'resources', 'Header image v2.mp4'))


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
        st.subheader('TB Detection')
        image_file = st.file_uploader('Upload Chest X-ray', type=['png','jpg','jpeg'])
        
        if image_file is not None:
            
            # retrieve image details
            st.write(type(image_file))
            # Methods and attributes on the class
            #st.write(dir(image_file))
            file_details = {'file name':image_file.name,
            'file type': image_file.type, 'file size':image_file.size}
            # Display details on screen
            st.write(file_details)
            
            # Creating column view
            col1, col2 = st.columns(2)
            
            with col1:
                # Display the image
                st.header('Original X-ray')
                st.image(load_image(image_file), width=255)
            
            with col2:
                st.header('Detection space')
                
                # resize the image
                resized_img = tf.image.resize(image_file, (256,256))
                # make a prediction
                yhat = model.predict(np.expand_dims(resized_img/255, 0))
                
                if yhat > 0.5:
                  st.write('Predicted class is TB')
                else:
                  st.write('Predicted class is Normal')

        
    else:
        choice == 'About'
    
if __name__ == '__main__':
    main()




















