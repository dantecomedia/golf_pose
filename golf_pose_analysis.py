import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder 
import time

'''
lb=LabelEncoder()
lb.classes_=np.array(['A1 - Address', 'Front A1 - Address',
       'A2 - Shaft Horizontal (Backswing)',
       'A3 - Lead Arm Horizontal (Backswing)',
       'A3 - Lead Arm Horizontal (Backswing)',
       'Front A3 - Lead Arm Horizontal (Backswing)',
       'A4 - Top of the Backswing', 'A4 - Top of the Backswing',
       'A5 - Lead Arm Horizontal (Downswing)',
       'A5 - Lead Arm Horizontal (Downswing)',
       'A5 - Lead Arm Horizontal (Downswing)', 'A7 - Impact',
       'A7 - Impact', 'A7 - Impact',
       'A9 - Lead Arm Horizontal to the Ground on Follow-Through',
       'A10 - End of Swing', 'A10 - End of Swing', 'A10 - End of Swing'])


model=xgb.XGBClassifier(max_depth=10,n_job=-1)
model.load_model('golf_weights2.bst')
'''

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 


def return_pose(img):
    

    img = cv2.imread(img, 1) 
    cv2.imshow("m",img)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:


        # Recolor image to RGB
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            #print(landmarks)
        except Exception as e:
            return "no_landmarks_detected"
            


        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )               

        cv2.imshow('Mediapipe Feed', image)
        time.sleep(6)



    #address position

    #address position

    right_shoulder=[landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    right_elbow=[landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
    right_wrist=[landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
    right_ankle=[landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

    left_shoulder=[landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    left_elbow=[landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    left_wrist=[landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
    left_ankle=[landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

    angle_rs_re_rw=calculate_angle(right_shoulder, right_elbow, right_wrist)   #angle between rs, re,rw
    angle_rs_rh_rk=calculate_angle(right_shoulder, right_hip, right_knee)      #angle between rs,rh,rk
    angle_ls_le_lw=calculate_angle(left_shoulder, left_elbow, left_wrist)      #angle between ls,le,lw
    angle_ls_lh_lk=calculate_angle(left_shoulder, left_hip, left_knee)         #angle between ls,lh,lk

    angle_ls_rh_re=calculate_angle(left_shoulder,right_shoulder, right_elbow)  #angle between ls,rh,re
    angle_rs_lh_le=calculate_angle(right_shoulder,left_shoulder, left_elbow)   #angle between rs,lh,le
    angle_re_rw_le=calculate_angle(right_elbow,right_wrist, left_elbow)        #angle between re,rw,le
    angle_re_lw_le=calculate_angle(right_elbow,left_wrist, left_elbow)         #angle between re,lw,le

    angle_rh_rk_ra=calculate_angle(right_hip,right_knee, right_ankle)          #angle between rh,rk,ra
    angle_lh_lk_la=calculate_angle(left_hip,left_knee, left_ankle)             #angle between lh,lk,la




    if (angle_rs_re_rw>=145 and angle_rs_re_rw<=155) or (angle_rs_rh_rk>=119 and angle_rs_rh_rk<=123) and (angle_ls_le_lw>=145 and angle_ls_le_lw<=155) or (angle_ls_lh_lk>=119 and angle_ls_lh_lk<=123): 
        return ("Address_position")

    if (angle_ls_rh_re>=75 and angle_ls_rh_re<=99) and (angle_rs_lh_le>=75 and angle_rs_lh_le<=99) and (angle_re_rw_le>=58 and angle_re_rw_le<=81) and (angle_re_lw_le>=58 and angle_re_lw_le<=81) and ((angle_ls_le_lw>=150 and angle_ls_le_lw<=163)) and ((angle_rs_re_rw>=150 and angle_rs_re_rw<=162)):
        return ("Front Facing Address Position")

    if (angle_rh_rk_ra>=148 and angle_rh_rk_ra<=170) and (angle_lh_lk_la>=148 and angle_lh_lk_la<=170) and (angle_rs_re_rw>=133 and angle_rs_re_rw<=162) and (angle_ls_le_lw>=133 and angle_ls_le_lw<=162) and (angle_rs_rh_rk>=110 and angle_rs_rh_rk<=135) and (angle_ls_lh_lk>=110 and angle_ls_lh_lk<=135):
        return ("A2 - Shaft Horizontal (Backswing)")

    if (angle_rs_rh_rk>=158 and angle_rs_rh_rk<=161) and (angle_ls_lh_lk>=158 and angle_ls_lh_lk<=160) and (angle_rs_lh_le>=65 and angle_rs_lh_le<=68) and (angle_ls_rh_re>=103 and angle_rs_lh_le<=105) or (angle_ls_rh_re>=65 and angle_ls_rh_re<=68) and (angle_rs_lh_le>=103 and angle_rs_lh_le<=105) and (angle_rs_re_rw>=165 and angle_rs_re_rw<=169) and (angle_ls_le_lw>=149 and angle_ls_le_lw<=151):
        return ("Front A2 - Shaft Horizontal (Backswing)")

    if angle_ls_le_lw<=142 and angle_ls_le_lw>=138 and angle_rs_re_rw>=90 and angle_rs_re_rw<=94 and angle_rs_lh_le>=42 and angle_rs_lh_le<=45 and angle_ls_lh_lk>=109 and angle_ls_lh_lk<=111 and angle_rs_rh_rk>=121 and angle_rs_rh_rk<=126 and angle_lh_lk_la>=139 and angle_lh_lk_la<=141 and angle_rh_rk_ra>=159 and angle_rh_rk_ra<=162:
        return("Lead Arm Horizontal (Backswing)")

    if angle_rs_re_rw>=140 and angle_rs_re_rw<=148 and angle_rs_rh_rk>=100 and angle_rs_rh_rk<=110 and angle_ls_le_lw>=110 and angle_ls_le_lw<=118 and angle_ls_lh_lk>=120 and angle_ls_lh_lk<=125 and angle_ls_rh_re>=45 and angle_ls_rh_re<=50 and angle_rs_lh_le>=145 and angle_rs_lh_le<=150 and angle_re_rw_le>173 and angle_re_rw_le<=175 and angle_re_lw_le>=75 and angle_re_lw_le<=78 and angle_rh_rk_ra>=135 and angle_rh_rk_ra<=145 and angle_lh_lk_la>=145 and angle_lh_lk_la<=150:
        return("A3 - Lead Arm Horizontal (Backswing)")

    if angle_rs_re_rw>=115 and angle_rs_re_rw<=120 and angle_rs_rh_rk>=148 and angle_rs_rh_rk<=155 and angle_ls_le_lw>=148 and angle_ls_le_lw<=155 and angle_ls_lh_lk>=170 and angle_ls_lh_lk<=180 and angle_ls_rh_re>=115 and angle_ls_rh_re<=120 and angle_rs_lh_le>=30 and angle_rs_lh_le<=35 and angle_re_rw_le>=0 and angle_re_rw_le<=2 and angle_re_lw_le>=0 and angle_re_lw_le<=2 and angle_rh_rk_ra>=170 and angle_rh_rk_ra<=180 and angle_lh_lk_la>=150 and angle_lh_lk_la<=160:
        return("Front A3 - Lead Arm Horizontal (Backswing)")

    if angle_rs_re_rw>=120 and angle_rs_re_rw<=130 and angle_rs_rh_rk>=148 and angle_rs_rh_rk<=155 and angle_ls_le_lw>=120 and angle_ls_le_lw<=130 and angle_ls_lh_lk>=170 and angle_ls_lh_lk<=180 and angle_ls_rh_re>=155 and angle_ls_rh_re<=162 and angle_rs_lh_le>=9 and angle_rs_lh_le<=15 and angle_re_rw_le>=20 and angle_re_rw_le<=30 and angle_re_lw_le>=20 and angle_re_lw_le<=30 and angle_rh_rk_ra>=167 and angle_rh_rk_ra<=175 and angle_lh_lk_la>=166 and angle_lh_lk_la<=174:
        return("Front A4 - Top of the Backswing")

    if angle_rs_re_rw>=90 and angle_rs_re_rw<=95 and angle_rs_rh_rk>=120 and angle_rs_rh_rk<=128 and angle_ls_le_lw>=139 and angle_ls_le_lw<=146 and angle_ls_lh_lk>=102 and angle_ls_lh_lk<=110 and angle_ls_rh_re>=120 and angle_ls_rh_re<=130 and angle_rs_lh_le>=25 and angle_rs_lh_le<=30 and angle_re_rw_le>=15 and angle_re_rw_le<=25 and angle_re_lw_le>=10 and angle_re_lw_le<=18 and angle_rh_rk_ra>=160 and angle_rh_rk_ra<=168 and angle_lh_lk_la>=130 and angle_lh_lk_la<=140:
        return("A4 - Top of the Backswing")


    if angle_rs_re_rw>=115 and angle_rs_re_rw<=120 and angle_rs_rh_rk>=145 and angle_rs_rh_rk<=155 and angle_ls_le_lw>=145 and angle_ls_le_lw<=155 and angle_ls_lh_lk>=170 and angle_ls_lh_lk<=180 and angle_ls_rh_re>=115 and angle_ls_rh_re<=120 and angle_rs_lh_le>=30 and angle_rs_lh_le<=35 and angle_re_rw_le>=0 and angle_re_rw_le<=2 and angle_re_lw_le>=0 and angle_re_lw_le<=2 and angle_rh_rk_ra>=170 and angle_rh_rk_ra<=180 and angle_lh_lk_la>=150 and angle_lh_lk_la<=160:
        return("Front A5 - Lead Arm Horizontal (Downswing)")


    if angle_rs_re_rw>=115 and angle_rs_re_rw<=120 and angle_rs_rh_rk>=145 and angle_rs_rh_rk<=155 and angle_ls_le_lw>=145 and angle_ls_le_lw<=155 and angle_ls_lh_lk>=170 and angle_ls_lh_lk<=180 and angle_ls_rh_re>=115 and angle_ls_rh_re<=120 and angle_rs_lh_le>=30 and angle_rs_lh_le<=35 and angle_re_rw_le>=0 and angle_re_rw_le<=2 and angle_re_lw_le>=0 and angle_re_lw_le<=2 and angle_rh_rk_ra>=170 and angle_rh_rk_ra<=180 and angle_lh_lk_la>=150 and angle_lh_lk_la<=160:
        return("Front A5 - Lead Arm Horizontal (Downswing)")

    if angle_rs_re_rw>=90 and angle_rs_re_rw<=100 and angle_rs_rh_rk>=160 and angle_rs_rh_rk<=170 and angle_ls_le_lw>=160 and angle_ls_le_lw<=170 and angle_ls_lh_lk>=160 and angle_ls_lh_lk<=170 and angle_ls_rh_re>=69 and angle_ls_rh_re<=80 and angle_rs_lh_le>=60 and angle_rs_lh_le<=70 and angle_re_rw_le>=20 and angle_re_rw_le<=30 and angle_re_lw_le>=30 and angle_re_lw_le<=40 and angle_rh_rk_ra>=130 and angle_rh_rk_ra<=140 and angle_lh_lk_la>=170 and angle_lh_lk_la<=180:
        return("A10 - End of Swing")

    if angle_rs_re_rw>=150 and angle_rs_re_rw<=160 and angle_rs_rh_rk>=140 and angle_rs_rh_rk<=150 and angle_ls_le_lw>=60 and angle_ls_le_lw<=75 and angle_ls_lh_lk>=150 and angle_ls_lh_lk<=160 and angle_ls_rh_re>=0 and angle_ls_rh_re<=2 and angle_rs_lh_le>=140 and angle_rs_lh_le<=150 and angle_re_rw_le>=75 and angle_re_rw_le<=80 and angle_re_lw_le>=60 and angle_re_lw_le<=70 and angle_rh_rk_ra>=170 and angle_rh_rk_ra<=180 and angle_lh_lk_la>=170 and angle_lh_lk_la<=180:
        return("BACK A10 - End of Swing")

    if angle_rs_re_rw>=150 and angle_rs_re_rw<=160 and angle_rs_rh_rk>=140 and angle_rs_rh_rk<=150 and angle_ls_le_lw>=60 and angle_ls_le_lw<=75 and angle_ls_lh_lk>=150 and angle_ls_lh_lk<=160 and angle_ls_rh_re>=0 and angle_ls_rh_re<=2 and angle_rs_lh_le>=140 and angle_rs_lh_le<=150 and angle_re_rw_le>=75 and angle_re_rw_le<=80 and angle_re_lw_le>=60 and angle_re_lw_le<=70 and angle_rh_rk_ra>=170 and angle_rh_rk_ra<=180 and angle_lh_lk_la>=170 and angle_lh_lk_la<=180:
        return("BACK A10 - End of Swing")

    if angle_rs_re_rw>=170 and angle_rs_re_rw<=180 and angle_rs_rh_rk>=130 and angle_rs_rh_rk<=150 and angle_ls_le_lw>=150 and angle_ls_le_lw<=168 and angle_ls_lh_lk>=170 and angle_ls_lh_lk<=180 and angle_ls_rh_re>=105 and angle_ls_rh_re<=115 and angle_rs_lh_le>=55 and angle_rs_lh_le<=65 and angle_re_rw_le>=50 and angle_re_rw_le<=65 and angle_re_lw_le>=60 and angle_re_lw_le<=70 and angle_rh_rk_ra>=155 and angle_rh_rk_ra<=175 and angle_lh_lk_la>=155 and angle_lh_lk_la<=165:
        return("Front A6 - Shaft Horizontal (Downswing)")

    if angle_rs_re_rw>=115 and angle_rs_re_rw<=125 and angle_rs_rh_rk>=110 and angle_rs_rh_rk<=120 and angle_ls_le_lw>=165 and angle_ls_le_lw<=175 and angle_ls_lh_lk>=115 and angle_ls_lh_lk<=125 and angle_ls_rh_re>=80 and angle_ls_rh_re<=100 and angle_rs_lh_le>=120 and angle_rs_lh_le<=130 and angle_re_rw_le>=60 and angle_re_rw_le<=70 and angle_re_lw_le>=45 and angle_re_lw_le<=55 and angle_rh_rk_ra>=135 and angle_rh_rk_ra<=145 and angle_lh_lk_la>=145 and angle_lh_lk_la<=155:
        return("A6 - Shaft Horizontal (Downswing)")

    if angle_rs_re_rw>=20 and angle_rs_re_rw<=30 and angle_rs_rh_rk>=127 and angle_rs_rh_rk<=135 and angle_ls_le_lw>=85 and angle_ls_le_lw<=95 and angle_ls_lh_lk>=140 and angle_ls_lh_lk<=145 and angle_ls_rh_re>=140 and angle_ls_rh_re<=150 and angle_rs_lh_le>=50 and angle_rs_lh_le<=60 and angle_re_rw_le>=75 and angle_re_rw_le<=85 and angle_re_lw_le>=130 and angle_re_lw_le<=140 and angle_rh_rk_ra>=150 and angle_rh_rk_ra<=165 and angle_lh_lk_la>=165 and angle_lh_lk_la<=185:
        return("A9 - Lead Arm Horizontal to the Ground on Follow-Through")

    if angle_rs_re_rw>=159 and angle_rs_re_rw<=165 and angle_rs_rh_rk>=130 and angle_rs_rh_rk<=145 and angle_ls_le_lw>=170 and angle_ls_le_lw<=180 and angle_ls_lh_lk>=155 and angle_ls_lh_lk<=170 and angle_ls_rh_re>=62 and angle_ls_rh_re<=70 and angle_rs_lh_le>=95 and angle_rs_lh_le<=105 and angle_re_rw_le>=35 and angle_re_rw_le<=45 and angle_re_lw_le>=30 and angle_re_lw_le<=40 and angle_rh_rk_ra>=140 and angle_rh_rk_ra<=150 and angle_lh_lk_la>=165 and angle_lh_lk_la<=185:
        return("Front A9 - Lead Arm Horizontal to the Ground on Follow-Through")
    
    
    return "Nothing Detected"


return_pose("address for the image")



