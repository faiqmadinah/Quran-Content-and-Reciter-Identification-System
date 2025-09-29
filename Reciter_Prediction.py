import streamlit as st
from streamlit_option_menu import option_menu
from moviepy.video.io.VideoFileClip import VideoFileClip
import tempfile
import numpy as np
import librosa
import joblib


#load  models,tokenizer, and label econder 
model=joblib.load("model/reciter_model")



# Function to extract audio from video
def extract_audio_from_video(video_bytes):
    # Create a temporary file to store the video
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video_file:
        temp_video_file.write(video_bytes)
        video_path = temp_video_file.name

    # Load the video from the temporary file and extract audio
    audio_path = video_path.replace(".mp4", ".wav")
    video_clip = VideoFileClip(video_path)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(audio_path)  # Save audio to a temporary file
    
    return audio_path



#prediction function
def predict_reciter(audio_path):
    y, sr = librosa.load(audio_path,sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_scaled = np.mean(mfcc.T, axis=0).reshape(1, -1)
    
    prediction = model.predict(mfcc_scaled)
    pred_reciter=np.argmax(prediction)
    return pred_reciter

# Function to handle uploaded files and process them
def handle_file_upload(file):
    # Get the video file as bytes
    video_bytes = file.getbuffer()
    audio_path = extract_audio_from_video(video_bytes)

    return   audio_path



# Streamlit app layout
st.title("Reciter Identification from Video")
uploaded_file = st.file_uploader("Upload a recitation Video", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
       
    # Check if the file is a video
        if uploaded_file.type.startswith("video"):
            st.write("Video uploaded...")
            audio_path= handle_file_upload(uploaded_file)
            st.header("Reciter Name")
            reciter=predict_reciter(audio_path)
            if reciter==0:
                st.info(" Sheikh Maher Al-Muaiqly")
            elif reciter==1:
                st.info("Sheikh Saud Al-Shuraim")
            elif reciter==2:
                st.info("Sheikh Yasir Al Dossary")
        else:
            st.error("Invalid file type. Please upload a video.")
      
    
