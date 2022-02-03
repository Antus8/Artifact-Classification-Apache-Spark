import torch
import glob
import json
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
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier, LogisticRegression

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

'''
Solo pandas riesce a leggere bene i numpy arrays ma poi ci sono problemi con il modello di ML
Quindi convertiamo i numpy arrays in liste, così spark riesce a inferire lo schema
df = pd.DataFrame(dataset)
print(df)

indexer = StringIndexer(inputCol="label", outputCol="label_indexed")
indexed_df = indexer.fit(df).transform(df)
indexed.show()

IL MODELLO NON ACCETTA ARRAY MA SOLO VECTOR TYPE, QUINDI DOBBIAMO RICONVERTIRE
'''

labelIndexer = StringIndexer(inputCol="label", outputCol="label_indexed").fit(df)

(trainingData, testData) = df.randomSplit([0.7, 0.3])

rf = RandomForestClassifier(labelCol="label_indexed", featuresCol="vector_features", numTrees=10)

labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                               labels=labelIndexer.labels)
                               
pipeline = Pipeline(stages=[labelIndexer, rf, labelConverter])

model = pipeline.fit(trainingData)

predictions = model.transform(testData)

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
model = pipeline.fit(trainingData)
predictions = model.transform(testData)

predictions.show(50)

evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy")
lr_accuracy = evaluator.evaluate(predictions)

print("Random forest accuracy is: " + str(rf_accuracy))
print("Logistic regression accuracy is: " + str(lr_accuracy))


spark.stop()

