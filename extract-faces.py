import numpy as np
import cv2
import glob
import os

#宣告影像所需變數
sourePath = "photos"
savePath = "collected"
face_size_min = (47, 62)                                        #擷取的最小的臉部尺寸,小於此尺寸忽略不取
i = 0

#宣告cv2使用的辨識器及路徑
face_cascade = cv2.CascadeClassifier('objects\\lbpcascade_frontalface.xml')

if (os.path.exists(savePath) ==0 ):     os.makedirs(savePath)   #若存放資料夾不存在則建立


#對相片進行人臉蒐集
for filePath in glob.glob(sourePath+"\\*"):
    print("Load {} ...".format(filePath))                       #  .format() 格式化字符函數 , 依序讀取圖片

    # 擷取影像轉為灰階
    img = cv2.imread(filePath)                                  # 讀取圖像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                # 將圖像由RGB轉為灰階
    faces = face_cascade.detectMultiScale(gray, 1.1, 8)         # 設定人臉資訊格式

    #將人臉資訊裁減後輸出
    for (x,y,w,h) in faces:
        if(w>=face_size_min[0] and h>=face_size_min[1]):
            roi_color = img[y:y+h, x:x+w]
            cv2.imwrite(savePath+"/face-"+str(i)+".jpg", roi_color)
            i+=1
			
print("All faces has been extracted to {}".format(savePath))
