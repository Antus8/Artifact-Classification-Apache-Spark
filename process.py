import cv2
import torch
import glob
import json
import time
import numpy as np
import pandas as pd
import torch.nn as nn
from PIL import Image
from pyspark.ml import Pipeline
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

	predictions.show(50)


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

	evaluator = MulticlassClassificationEvaluator(
	    labelCol="label", predictionCol="prediction", metricName="accuracy")
	ova_accuracy = evaluator.evaluate(predictions)

	print("Random forest accuracy is: " + str(rf_accuracy))
	print("Logistic regression accuracy is: " + str(lr_accuracy))
	print("One vs All accuracy is: " + str(ova_accuracy))
	
	
	acquire_and_predict(rf_model, lr_model, ova_model)

	spark.stop()

def acquire_and_predict(rf_model, lr_model, ova_model):
	camera = cv2.VideoCapture("/home/antonello/Scrivania/Artifact-Classification-Apache-Spark/video.mp4")
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
		
		print(np_arr)
		time.sleep(10)
		
    	
	

def get_vector(image, normalize, to_tensor, scaler,layer):
	img = image   

	t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))
	img_feature_vector = torch.zeros(512)  # The 'avgpool' layer has an output size of 512, this is an empty vector that will hold features

	def copy_data(m, i, o):
		img_feature_vector.copy_(o.data.reshape(o.data.size(1)))

	h = layer.register_forward_hook(copy_data)    

	model(t_img)    
	h.remove() # Detach copy function from the layer

	return img_feature_vector
	

if __name__ == "__main__":
	main()



