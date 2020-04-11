#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.datasets.base import Bunch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from imutils import paths
from scipy import io
import numpy as np
import random
import imutils
import cv2

#宣告影像所需變數
facePath = "datasets"
faces_min = 59			#定義每個類型最多要幾張臉孔才能訓練
face_size = (47, 62)	#擷取的最小的臉部尺寸,小於此尺寸忽略不取
test_size = 0.25		#定義dataset要分出多少比例做為test dataset
						#equal_sample 定義所有不同人物的訓練樣本數目是否要相同,與min_faces參數搭配使用
						#flatten是否要將讀入的圖片資料攤平為一維

#載入dataset,將數據轉為(training, testing, names) 三份datasets
def load_sunplusit_faces(datasetPath, min_faces=10, face_size=face_size, equal_samples=True , test_size=test_size , seed=42 ):
	imagePaths = sorted(list(paths.list_images(datasetPath)))  #排列資料夾名稱與內部照片

	random.seed(seed)
	data = []
	labels = []

	for (i, imagePath) in enumerate(imagePaths):
		#  將圖像轉換為灰階
		print(imagePath)
		face = cv2.imread(imagePath)
		face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
		face = cv2.resize(face, face_size)

		# 更新數據矩陣和相關標籤
		data.append(face)

		labels.append(imagePath.split("\\")[-2])  #表示資料夾名稱順序

	data = np.array(data)
	labels = np.array(labels)

	# 所有不同人物的訓練樣本數目是否要相同
	if equal_samples:
		# initialize the list of sampled indexes
		sampledIdxs = []

		# loop over the unique labels
		for label in np.unique(labels):		#去除重複的元素後排列
			# grab the indexes into the labels array where labels equals the current label
			labelIdxs = np.where(labels == label)[0]

			# only proceed if the required number of minimum faces can be met
			if len(labelIdxs) >= min_faces:
				# randomly sample the indexes for the current label, keeping only minumum amount, then update the list of sampled idnexes
				labelIdxs = random.sample(list(labelIdxs), min_faces)
				sampledIdxs.extend(labelIdxs)

		# use the sampled indexes to select the appropriate data points and labels
		random.shuffle(sampledIdxs)    	# 將sampledIdxs隨機排列
		data = data[sampledIdxs]
		labels = labels[sampledIdxs]

	# compute the training and testing split index
	idxs = range(0, len(data))
	random.shuffle(list(idxs))
	split = int(len(idxs) * (1.0 - test_size))

	# split the data into training and testing segments
	(trainData, testData) = (data[:split], data[split:])
	(trainLabels, testLabels) = (labels[:split], labels[split:])

	# create the training and testing bunches
	training = Bunch(name="training", data=trainData, target=trainLabels)
	testing = Bunch(name="testing", data=testData, target=testLabels)

	# return a tuple of the training, testing bunches, and original labels
	return (training, testing, labels) 

#將return的結果傳至(training, testing, names)
(training, testing, names) = load_sunplusit_faces(facePath, min_faces=faces_min, test_size=test_size)

#將training dataset的label透過encoder由字串轉為數字格式 , 並製成.target
le = LabelEncoder()
le.fit_transform(training.target)

#載入辨識器 命名為recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=16, grid_x=8, grid_y=8)

#將training得到的dataset匯入recongnizer進行訓練
print("training face recognizer...")
recognizer.train(training.data, le.transform(training.target))
#定義存放預測testing dataset結果與信心指數的兩個變數
predictions = []
confidence = []

for i in range(0, len(testing.data)):
	#分類臉孔,並更新預測結果和可信度分數
	(prediction, conf) = recognizer.predict(testing.data[i])
	predictions.append(prediction)
	confidence.append(conf)
	
# show the classification report
print(classification_report(le.transform(testing.target), predictions, target_names=np.unique(names)))

#將training.target分析的結果輸出為target.out
np.savetxt('objects\\target.out', training.target, delimiter=',', fmt="%s") 
print("Exporting model...")

#輸出recognizer
recognizer.write("objects\\owner.yaml")


