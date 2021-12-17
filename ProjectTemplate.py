from tensorflow.keras.models import load_model
import mediapipe as mp
import numpy as np
import cv2

class ProjectTemplate:
    def __init__(self):
        self.face_landmarks_model = mp.solutions.face_mesh.FaceMesh(max_num_faces=1)
        self.eye_status_model = load_model("Models/eye_status_model2")


    def detect_and_draw_eye_statusV2(self,frame):
        img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        height,width = img.shape[0],img.shape[1]
        face_landmarks = self.face_landmarks_model.process(img)


        if face_landmarks.multi_face_landmarks:
            for landmarks in face_landmarks.multi_face_landmarks:
                try:
                    # P Q
                    right_start_x = int(landmarks.landmark[68].x * width)
                    right_start_y = int(landmarks.landmark[68].y * height)
                    right_end_x = int(landmarks.landmark[195].x * width)
                    right_end_y = int(landmarks.landmark[195].y * height)

                    # R S
                    right_width_x  = int(landmarks.landmark[9].x * width)
                    right_width_y  = int(landmarks.landmark[9].y * height)
                    right_height_x = int(landmarks.landmark[118].x * width)
                    right_height_y = int(landmarks.landmark[118].y * height)

                    #r_eye_img = cv2.cvtColor(frame[right_start_y:right_height_y,right_start_x:right_width_x],cv2.COLOR_BGR2RGB)
                    r_eye_img = frame[right_start_y:right_height_y, right_start_x:right_width_x]

                    # Improve resolution of image
                    #r_eye_img = HighRes.improve_res(r_eye_img)
                    r_eye_img = cv2.resize(r_eye_img,(100,100))
                    r_eye_img = cv2.cvtColor(r_eye_img,cv2.COLOR_BGR2GRAY)
                    r_eye = r_eye_img
                    r_eye_img = np.expand_dims(r_eye_img, axis=0)
                except Exception as e:
                    r_eye_img = np.array([])
                    continue

                try:
                    # P Q
                    left_start_x = int(landmarks.landmark[9].x * width)
                    left_start_y = int(landmarks.landmark[9].y * height)
                    left_end_x = int(landmarks.landmark[346].x * width)
                    left_end_y = int(landmarks.landmark[345].y * height)


                    # R S
                    left_width_x  = int(landmarks.landmark[298].x * width)
                    left_width_y  = int(landmarks.landmark[298].y * height)
                    left_height_x = int(landmarks.landmark[195].x * width)
                    left_height_y = int(landmarks.landmark[195].y * height)

                    #l_eye_img = cv2.cvtColor(frame[left_start_y:left_height_y,left_start_x:left_width_x],cv2.COLOR_BGR2RGB)
                    l_eye_img = frame[left_start_y:left_height_y, left_start_x:left_width_x]


                    # Improve resolution of image
                    #l_eye_img = HighRes.improve_res(l_eye_img)
                    l_eye_img = cv2.resize(l_eye_img,(100,100))
                    l_eye_img = cv2.cvtColor(l_eye_img,cv2.COLOR_BGR2GRAY)
                    l_eye = l_eye_img
                    l_eye_img = np.expand_dims(l_eye_img, axis=0)
                except Exception as e:
                    l_eye_img = np.array([])
                    continue

                try:
                    cv2.rectangle(frame,(right_start_x,right_start_y),(right_end_x,right_end_y),(255,0,0),2)
                except Exception as e:
                    print("Left Eye Not Detected Or : \n\n\n", e)

                try:
                    cv2.rectangle(frame, (left_start_x, left_start_y), (left_end_x, left_end_y), (255, 0, 0), 2)
                except Exception as e:
                    print("Left Eye Not Detected Or : \n\n\n",e)

                # Return Statements Acc to Situation
                return r_eye_img,l_eye_img



    def predict_eye_status(self,r_eye,l_eye):
        if r_eye.any() and l_eye.any():
            r_eye_pred = self.eye_status_model.predict(r_eye)
            l_eye_pred = self.eye_status_model.predict(l_eye)
        elif r_eye.any() and not l_eye.any():
            r_eye_pred = self.eye_status_model.predict(r_eye)
            l_eye_pred = 0
        elif l_eye.any() and not r_eye.any():
            l_eye_pred = self.eye_status_model.predict(l_eye)
            r_eye_pred = 0
        else:
            r_eye_pred = 0
            l_eye_pred = 0

        return r_eye_pred,l_eye_pred


    def display_status(self,frame,overall_status=""):
        if overall_status == "ANALYSING ...":
            color = (0,255,0)
        else:
            color = (0,0,255)
        cv2.putText(frame,text=overall_status,color=color,org=(50,50),thickness=2,fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=2)











