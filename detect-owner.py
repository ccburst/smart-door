#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.preprocessing import LabelEncoder
import numpy as np
import imutils
import cv2
import time
import datetime

#宣告影像所需變數
time.sleep(0.5)
TargetName = "owner"
monitor_winSize = (640, 480)                            #監控視窗大小
cam_resolution = (1024,768)                             #攝影機解析度

cascade = "haarcascade_frontalface_default.xml"         #使用haar的特徵分類器
                                                        # lbpcascade_frontalface.xml / haarcascade_frontalface_default.xml
#Face detect
face_size = (47, 62)
scaleFactor = 1.3
minNeighbors = 10

#####################################################################

#載入target資訊
face_cascade = cv2.CascadeClassifier('objects\\' + cascade)
target = np.loadtxt('objects\\target.out', delimiter=',', dtype=np.str)

#將target資料透過encoder由字串轉為數字格式
le = LabelEncoder()
le.fit_transform(target)
print("The model is loading , please wait...")

#載入辨識器 命名為recognizer 並將訓練完成的owner.yaml模型載入辨識器
recognizer = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=16, grid_x=8, grid_y=8)
recognizer.read("objects\\owner.yaml")

#調整攝影機設定
cam_id = 700                                             #啟動攝影機ID
camera = cv2.VideoCapture(cam_id)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

#將文字資訊與圖框回傳 , 顯示在監控視窗frame中
def putText(image, text, x, y, color=(255,255,255), thickness=1, size=1.2):
    if x is not None and y is not None:
        cv2.putText( image, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness)   #cv2.putText(影像,文字,座標,字型,大小,顏色,線條寬度,線條種類)
    return image

while True:

    # 擷取影像轉為灰階
    (ret, img) = camera.read()                      #從攝影機擷取一張影像  ret回傳True/False  img回傳值為影像畫面
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 擷取灰階影像中的人臉資訊
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor= scaleFactor,
        minNeighbors=minNeighbors,
        minSize=face_size,
    )

    i = 0
    # 將人臉框起後處理
    for (x,y,w,h) in faces:
        roi_color = img[y:y+h, x:x+w]
        
        if(w>face_size[0] and h>face_size[1]):
            # 將人臉資訊裁減後 , 套入模型判斷是否為目標
            roi_gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(roi_gray, face_size)
        
            (prediction, conf) = recognizer.predict(face)                           #將人臉資訊套入模型預測結果
            namePredict = le.inverse_transform(prediction)                          #將預測結果轉換為其label名稱
                
            #若判斷出的人臉為target
            if(TargetName in namePredict):
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
                putText(img, namePredict, x, y-20, (0,0,255), thickness=2, size=2)  #將偵測到的人臉框起並告知對象為owner
            else:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
                putText(img, namePredict, x, y-20, (0,255,0), thickness=2, size=2)  #將偵測到的人臉框起並告知對象為guest

    # 將擷取的影像顯示至監控視窗
    cv2.imshow("Frame", img)
    key = cv2.waitKey(1)
