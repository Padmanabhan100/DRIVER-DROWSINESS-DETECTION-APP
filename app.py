from ProjectTemplate import ProjectTemplate
import winsound
import streamlit as st
import time
import cv2
import pandas as pd
import numpy as np


# create an empty area for continuous flow of images
st.title("DROWSINESS DETECTION APP 😴 ")
# select options of videos
video_option = st.selectbox("SELECT CAMERA OR TRY WITH SAMPLE VIDEOS ? (ORIENTATION AND POSITION IS IMP FOR GOOD RESULTS)",("VIDEO1","VIDEO2","VIDEO3","CAMERA"))

if video_option == "CAMERA":
    video_option = 0
elif video_option == "VIDEO1":
    video_option = "videos/video3.mp4"
elif video_option == "VIDEO2":
    video_option = "videos/video2.mp4"
elif video_option == "VIDEO3":
    video_option = "videos/video(drowsy).mp4"


main_vid = st.empty()
check_model_working = st.checkbox("DETAILS AND INFO")

col1, col2, col3 = st.columns(3)

with col1:
    r_eye_text = st.empty()
    r_eye_vid = st.empty()
    l_eye_text = st.empty()
    l_eye_vid = st.empty()

with col2:
    model_arch_text_empty = st.empty()
    model_arch_empty = st.empty()

with col3:
    eye_pred_header_empty = st.empty()
    eye_pred_empty = st.empty()
    if check_model_working:
        eye_visualize = st.checkbox("Wanna Visualize Probabilities ?(Might Slow Down The Process A Bit)")
    prob_graph = st.empty()

# start time
eye_score_list = []
start_time = time.localtime().tm_sec

if video_option is not None:
    # Initialize camera
    cap = cv2.VideoCapture(video_option)
    pt = ProjectTemplate()

    # capture Images from Camera Iteratively
    while True:
        ret,frame = cap.read()

        # Preprocess Image and Detect Eyes
        try:
            r_eye,l_eye = pt.detect_and_draw_eye_statusV2(frame)
        except:
            r_eye,l_eye = np.array([]),np.array([])

        # Predict Eyes Status as Open Or Close
        try:
            r_eye_pred,l_eye_pred = pt.predict_eye_status(r_eye,l_eye)
            # append to eye score list
            if r_eye_pred[0][0] >= 0.75:
                eye_score_list.append(1)
            if l_eye_pred[0][0] >=0.5:
                eye_score_list.append(1)
            if r_eye_pred[0][0] < 0.75:
                eye_score_list.append(0)
            if l_eye_pred[0][0] < 0.5:
                eye_score_list.append(0)


            end_time = time.localtime().tm_sec

            overall_eye_status = "ANALYSING ..."
            # in every 2 seconds
            if end_time-start_time == 2 or end_time-start_time == -57:
                start_time = time.localtime().tm_sec
                # calculate Eye score
                eye_score = sum(eye_score_list)/len(eye_score_list)
                # clear eye_score_list for revaluation
                eye_score_list.clear()
                if eye_score < 0.51:
                    overall_eye_status = "DONT FALL ASLEEP !!!"
                    winsound.Beep(2500,500)

        except:
            pass

        # Display Eyes Open Or Close On Image
        try:
            pt.display_status(frame,overall_eye_status)
        except:
            pass

        # display the main video On Streamlit
        try:
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            main_vid.image(frame)
        except Exception as e:
            print(e)



        # If Checked show the internals
        if check_model_working:
            try:
                r_eye = np.squeeze(r_eye,axis=0)
                l_eye = np.squeeze(l_eye, axis=0)

                # display eyes and text
                r_eye_vid.image(r_eye)
                r_eye_text.subheader("RIGHT EYE")
                l_eye_vid.image(l_eye)
                l_eye_text.subheader("LEFT EYE")
                # display model architecture image
                model_arch_text_empty.subheader("MODEL ARCHITECTURE")
                model_arch_empty.image("EyeClassificationArch.png")
                # display predictions
                eye_pred_header_empty.subheader("MODEL PREDICTIONS")
                eye_pred_empty.text(f"Probab. Right Eye Open:{np.round(r_eye_pred[0][0],2)} \n\nProbab. Left Eye Open:{np.round(l_eye_pred[0][0],2)}")

                if eye_visualize:
                    df = pd.DataFrame([round(r_eye_pred[0][0],2),round(l_eye_pred[0][0],2)],["Right Open Prob.","Left Open Prob"])
                    prob_graph.bar_chart(df,width=250,height=250)


            except Exception as e:
                print(e)
