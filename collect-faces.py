#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob, os
from sklearn.preprocessing import LabelEncoder
from imutils import paths
from scipy import io
import numpy as np
import imutils
import cv2
import sys
import time
import datetime

#宣告影像所需變數
savePath = "collected"
face_size = (47, 62)                                            #蒐集人臉尺寸
monitor_winSize = (640, 480)                                    #電腦中監控視窗大小
cam_resolution = (1080,960)                                     #相機解析度設定

#攝影機格式宣告
cam_id = 700                                                    #設定相機編號
camera = cv2.VideoCapture(cam_id)                               #開啟相機裝置(引數) , 引數為相機裝置ID
camera.set(cv2.CAP_PROP_FRAME_WIDTH, cam_resolution[0])         #設定影像寬段為1080
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_resolution[1])        #設定影像高度為960    #xxx.set(propID, value) 設定影像尺寸大小

#宣告cv2使用的辨識器及路徑
face_cascade = cv2.CascadeClassifier('objects\\haarcascade_frontalface_default.xml')     #可使用lbpcascade_frontalface.xml速度較快,但能處理的人臉數量較少
                                                                                         #利用cv2.CascadeClassifier導入辨識工具,選擇臉部辨識工具
                                                                                         #Haar為特徵分類器的XML檔 , 描述各個部位的Haar特徵值

if (os.path.exists(savePath) ==0 ):     os.makedirs(savePath)                            #若collected資料夾不存在 則建立collected資料夾

#LOOP 擷取人臉照片
while True:

    #擷取影像轉為灰階
    (ret , img) = camera.read()                                 #從攝影機擷取一張影像  ret回傳True/False  img回傳值為影像畫面
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                #將 img的色調 彩色rgb 轉為 灰階gray

    #擷取灰階影像中的人臉資訊
    faces = face_cascade.detectMultiScale(                      #此函數檢測圖片中所有人臉,將人臉以vector保存各個人臉的座標,大小(以矩形表示)
        gray,                                                   #待檢測圖片,以灰階圖像加快檢測速度
        scaleFactor= 1.3,                                       #每次圖像尺寸減小的比例
        minNeighbors=10,                                        #類似區域最少重複抓到幾個以上才認定為人臉
        minSize=face_size,                                      #限制取得的人臉大小
    )
    
    #將人臉資訊裁減後輸出
    i = 0
    for (x,y,w,h) in faces:                                     #將faces取得的人臉向量帶入(X,Y,W,H)  x,y為裁減區域的座標 , w,h為裁減區域的長寬
	
        if(w>face_size[0] and h>face_size[1]):                  #若蒐集到的人臉超出人臉範圍
            roi_color = img[y:y+h, x:x+w]                       #裁減圖片
            now=datetime.datetime.now()
            faceName = '%s_%s_%s_%s_%s_%s_%s.jpg' % (now.year, now.month, now.day, now.hour, now.minute, now.second, i)
            cv2.imwrite(savePath+"\\" + faceName, roi_color)    #寫入圖片檔案 指定輸出位置及名稱 使用numpy陣列
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)      #描繪方框(影像,頂點座標,對角頂點座標,顏色,線條寬度)

    #將擷取的影像顯示至監控視窗
    cv2.imshow("Frame", img)                                    #顯示img圖像至frame中
    key = cv2.waitKey(1)                                        #持續刷新圖像 delay 1ms

