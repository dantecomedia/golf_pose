import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder 
import google.auth
from datetime import datetime
import pandas
import pytz
import joblib

import os
import datetime
from xgboost import XGBClassifier
import xgboost

import requests

from google.cloud import storage
#from sklearn.externals import joblib
from io import BytesIO

import tensorflow as tf

from flask import Flask, redirect, url_for, request
app = Flask(__name__)


lb=LabelEncoder()
lb.classes_=np.array(['A1 - Address',
 'A10 - End of Swing',
 'A2 - Shaft Horizontal (Backswing)',
 'A3 - Lead Arm Horizontal (Backswing)',
 'A4 - Top of the Backswing',
 'A5 - Lead Arm Horizontal (Downswing)',
 'A6 - Shaft Horizontal (Downswing)',
 'A9 - Lead Arm Horizontal to the Ground on Follow-Through',
 'Front A1 - Address',
 'Front A10 - End of Swing',
 'Front A2 - Shaft Horizontal (Backswing)',
 'Front A3 - Lead Arm Horizontal (Backswing)',
 'Front A4 - Top of the Backswing',
 'Front A5 - Lead Arm Horizontal (Downswing)',
 'Front A6 - Shaft Horizontal (Downswing)',
 'Front A9 - Lead Arm Horizontal to the Ground on Follow-Through',
 'Lead Arm Horizontal (Backswing)'])

#print(lb.classes_)


model=xgb.XGBClassifier(max_depth=10,n_job=-1)
#xgb_model=joblib.load(tf.io.gfile.GFile("gs://xgbweights/xgb_golf.bin",'rb'))
model.load_model("golf_weights (2).bst")

#model.load_model('golf_weights (2).bst')
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

@app.route('/return_pose/<img>',methods=['POST'])
def return_pose(img):
    #res = img.get_json()
    #img=res['image']

    img = cv2.imread(img, 1) 
    #cv2.imshow("m",img)
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
    
    global angle_rs_re_rw,angle_rs_rh_rk,angle_ls_le_lw,angle_ls_lh_lk,angle_ls_rh_re,angle_rs_lh_le,angle_re_rw_le,angle_re_lw_le,angle_rh_rk_ra,angle_lh_lk_la
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
        print("Address_position")

    if (angle_ls_rh_re>=75 and angle_ls_rh_re<=99) and (angle_rs_lh_le>=75 and angle_rs_lh_le<=99) and (angle_re_rw_le>=58 and angle_re_rw_le<=81) and (angle_re_lw_le>=58 and angle_re_lw_le<=81) and ((angle_ls_le_lw>=150 and angle_ls_le_lw<=163)) and ((angle_rs_re_rw>=150 and angle_rs_re_rw<=162)):
        print("Front Facing Address Position")

    if (angle_rh_rk_ra>=148 and angle_rh_rk_ra<=170) and (angle_lh_lk_la>=148 and angle_lh_lk_la<=170) and (angle_rs_re_rw>=133 and angle_rs_re_rw<=162) and (angle_ls_le_lw>=133 and angle_ls_le_lw<=162) and (angle_rs_rh_rk>=110 and angle_rs_rh_rk<=135) and (angle_ls_lh_lk>=110 and angle_ls_lh_lk<=135):
        print("A2 - Shaft Horizontal (Backswing)")

    if (angle_rs_rh_rk>=158 and angle_rs_rh_rk<=161) and (angle_ls_lh_lk>=158 and angle_ls_lh_lk<=160) and (angle_rs_lh_le>=65 and angle_rs_lh_le<=68) and (angle_ls_rh_re>=103 and angle_rs_lh_le<=105) or (angle_ls_rh_re>=65 and angle_ls_rh_re<=68) and (angle_rs_lh_le>=103 and angle_rs_lh_le<=105) and (angle_rs_re_rw>=165 and angle_rs_re_rw<=169) and (angle_ls_le_lw>=149 and angle_ls_le_lw<=151):
        print("Front A2 - Shaft Horizontal (Backswing)")

    if angle_ls_le_lw<=142 and angle_ls_le_lw>=138 and angle_rs_re_rw>=90 and angle_rs_re_rw<=94 and angle_rs_lh_le>=42 and angle_rs_lh_le<=45 and angle_ls_lh_lk>=109 and angle_ls_lh_lk<=111 and angle_rs_rh_rk>=121 and angle_rs_rh_rk<=126 and angle_lh_lk_la>=139 and angle_lh_lk_la<=141 and angle_rh_rk_ra>=159 and angle_rh_rk_ra<=162:
        print("Lead Arm Horizontal (Backswing)")


    df=pd.DataFrame(columns=['angle_rs_re_rw','angle_rs_rh_rk','angle_ls_le_lw','angle_ls_lh_lk','angle_ls_rh_re','angle_rs_lh_le','angle_re_rw_le','angle_re_lw_le','angle_rh_rk_ra','angle_lh_lk_la'])
    val={}
    cols=list(df.columns)
    for i in cols:
        val[i]=globals()[i]
    #val['target']='Front A10 - End of Swing'
    df=df.append(val,ignore_index=True)
    df=df.astype(int)
    #print(df)
    y_pred=model.predict(df.values.reshape(1,-1))
    print(y_pred[0])
    xxx=lb.inverse_transform([y_pred[0]])
    return str(xxx)
    

if __name__ == '__main__':
    app.run(debug=True)

    #return_pose('Screenshot 2022-09-29 at 7.25.27 PM.png')
