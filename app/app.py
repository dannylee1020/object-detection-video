import streamlit as st
import os
import time

import config
from yolo import detect_objects


def run():

    st.title('Object Detection in Video')
    option = st.radio('', ['Choose a test video', 'Upload your own video (.mp4 only)'])
    st.sidebar.title('Parameters')
    confidence_slider = st.sidebar.slider('Confidence Threshold', 0.0, 1.0, config.DEFALUT_CONFIDENCE, 0.05)
    nms_slider = st.sidebar.slider('Non-Max Suppression Threshold', 0.0, 1.0, config.NMS_THRESHOLD, 0.05)

    if option == 'Choose a test video':
        test_videos = os.listdir(config.INPUT_PATH)
        test_video = st.selectbox('Please choose a test video', test_videos)

    else:
        test_video = st.file_uploader('Upload a video', type = ['mp4'])

        if test_video is not None:
            pass
        else:
            st.write('** Please upload a test video **')


    if st.button ('Detect Objects'):
        
        time.sleep(3)
        st.write(f"[INFO] Processing Video....")
        total, elap = detect_objects(config.VIDEO_PATH + test_video, confidence_slider, nms_slider)
        output_video = open(config.OUTPUT_PATH, 'rb')
        # output_video = open(config.VIDEO_PATH + test_video, 'rb')
        video_bytes = output_video.read()

        st.write(f"[INFO] The video has total of {total} frames")
        st.write(f"[INFO] Time required to process a single frame: {round(elap/60,2)} minutes")
        st.write(f"[INFO] Time required to process the entire video: {round((elap*total)/60, 2)} minutes")

        final_video = st.video(video_bytes)



if __name__ == '__main__':

    run()
    


