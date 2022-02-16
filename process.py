import cv2
import torch
import glob
import json
import time
import numpy as np
import pandas as pd
import seaborn as sn
import torch.nn as nn
from PIL import Image
from pyspark.ml import Pipeline
import matplotlib.pyplot as plt
import torchvision.models as models
from torch.autograd import Variable
from pyspark.sql import SparkSession
from pyspark import SparkContext,SparkConf
import torchvision.transforms as transforms
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.sql.types import StringType, ArrayType,StructType,StructField, FloatType
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier, LogisticRegression, OneVsRest


def main():

	spark = SparkSession.builder.appName('Artifact Classification').getOrCreate()
	outfile = '/home/antonello/Scrivania/Artifact-Classification-Apache-Spark/dataset.npz'

	npzfile = np.load(outfile)

	dataset = []

	for key in npzfile.files:
		for feature_vector in npzfile[key]:
			dataset.append({"label" : key, "features" : feature_vector.tolist()})

	df = spark.createDataFrame(dataset)
	# df.printSchema()

	list_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT())
	df = df.select(
		df["label"], 
		list_to_vector_udf(df["features"]).alias("vector_features")
	)

	'''
	###############################
	RANDOM FOREST CLASSIFIER
	###############################
	'''

	labelIndexer = StringIndexer(inputCol="label", outputCol="label_indexed").fit(df)

	(trainingData, testData) = df.randomSplit([0.7, 0.3])

	rf = RandomForestClassifier(labelCol="label_indexed", featuresCol="vector_features", numTrees=10)

	labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
		                       labels=labelIndexer.labels)
		                       
	pipeline = Pipeline(stages=[labelIndexer, rf, labelConverter])

	rf_model = pipeline.fit(trainingData)

	predictions = rf_model.transform(testData)

	predictions.show(testData.count())
	
	cm_rf = get_confusion_matrix_rf(predictions)

	evaluator = MulticlassClassificationEvaluator(
	    labelCol="label_indexed", predictionCol="prediction", metricName="accuracy")
	rf_accuracy = evaluator.evaluate(predictions)


	'''
	###############################
	MULTINOMIAL LOGISTIC REGRESSION
	###############################
	'''

	df = df.select(
	    df["label"].alias("string_labels"), 
	    df["vector_features"].alias("features")
	)

	labelIndexer = StringIndexer(inputCol="string_labels", outputCol="label").fit(df)
	(trainingData, testData) = df.randomSplit([0.7, 0.3])

	lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

	labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
		                       labels=labelIndexer.labels)
	pipeline = Pipeline(stages=[labelIndexer, lr, labelConverter])
	lr_model = pipeline.fit(trainingData)
	predictions = lr_model.transform(testData)

	predictions.show(50)
	
	cm_lr = get_confusion_matrix_lr(predictions)

	evaluator = MulticlassClassificationEvaluator(
	    labelCol="label", predictionCol="prediction", metricName="accuracy")
	lr_accuracy = evaluator.evaluate(predictions)

	'''
	###############################
	ONE VS ALL
	###############################
	'''
	lr = LogisticRegression(maxIter=10, tol=1E-6, fitIntercept=True)
	ovr = OneVsRest(classifier=lr)

	pipeline = Pipeline(stages=[labelIndexer, ovr, labelConverter])
	ova_model = pipeline.fit(trainingData)
	predictions = ova_model.transform(testData)
	
	cm_ova = get_confusion_matrix_lr(predictions)

	evaluator = MulticlassClassificationEvaluator(
	    labelCol="label", predictionCol="prediction", metricName="accuracy")
	ova_accuracy = evaluator.evaluate(predictions)

	print("Random forest accuracy is: " + str(rf_accuracy))
	print("Logistic regression accuracy is: " + str(lr_accuracy))
	print("One vs All accuracy is: " + str(ova_accuracy))
	
	sn.heatmap(cm_rf, annot=True)
	plt.show()
	sn.heatmap(cm_lr, annot=True)
	plt.show()
	sn.heatmap(cm_ova, annot=True)
	plt.show()
	
	
	
	# acquire_and_predict(spark, rf_model, lr_model, ova_model)
	#acquire_and_predict(spark, rf_model, None, None)

	spark.stop()

def acquire_and_predict(spark, rf_model, lr_model, ova_model):
	#camera = cv2.VideoCapture("/home/antonello/Scrivania/Artifact-Classification-Apache-Spark/door.mp4")
	camera = cv2.VideoCapture("/home/antonello/Scrivania/Artifact-Classification-Apache-Spark/anto.mp4")
	model = models.resnet18(pretrained=True)

	# Use the model object to select the desired layer
	layer = model._modules.get('avgpool')

	model.eval()

	scaler = transforms.Resize((224, 224))
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
									std=[0.229, 0.224, 0.225])
	to_tensor = transforms.ToTensor()
	
	while True:
		_, image = camera.read()
		
		#cv2.imshow("image", image)
		
		#cv2.waitKey(0) 
		
		img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		im_pil = Image.fromarray(img)
		img = im_pil
		t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))
		img_feature_vector = torch.zeros(512)
		
		def copy_data(m, i, o):
			img_feature_vector.copy_(o.data.reshape(o.data.size(1)))
		
		h = layer.register_forward_hook(copy_data) 
		model(t_img)    
		h.remove()

		np_arr = img_feature_vector.cpu().detach().numpy()
		
		current = [{"label" : "face", "features" : np_arr.tolist()}]
		current_img_df = spark.createDataFrame(current)

		list_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT())

		current_img_df = current_img_df.select(
		current_img_df["label"], 
		list_to_vector_udf(current_img_df["features"]).alias("vector_features"))

		predictions = rf_model.transform(current_img_df)

		predictions.select("prediction", "predictedLabel").show()
		
	#cv2.destroyAllWindows() 
	
	
def get_confusion_matrix_rf(df):
	
	pandasDF = df.toPandas()
	confusion_matrix = pd.crosstab(pandasDF['label_indexed'], pandasDF['prediction'], rownames=['Actual'], colnames=['Predicted'], margins = True)
	return confusion_matrix

	# sn.heatmap(confusion_matrix, annot=True)
	# plt.show()
	
	
def get_confusion_matrix_lr(df):

	pandasDF = df.toPandas()
	confusion_matrix = pd.crosstab(pandasDF['label'], pandasDF['prediction'], rownames=['Actual'], colnames=['Predicted'], margins = True)
	return confusion_matrix

	

if __name__ == "__main__":
	main()



