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
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.sql.types import StringType, ArrayType,StructType,StructField, FloatType

spark = SparkSession.builder.appName('Artifact Classification').getOrCreate()
outfile = '/home/antonello/spark-3.2.0-bin-hadoop3.2/Classification/dataset.npz'

npzfile = np.load(outfile)

dataset = []
    	
for key in npzfile.files:
    for feature_vector in npzfile[key]:
    	dataset.append({"label" : key, "features" : feature_vector.tolist()})

print(len(dataset))

'''
Solo pandas riesce a leggere bene i numpy arrays ma poi ci sono problemi con il modello di ML
Quindi convertiamo i numpy arrays in liste, così spark riesce a inferire lo schema
df = pd.DataFrame(dataset)
print(df)
'''

df = spark.createDataFrame(dataset)
df.printSchema()

list_to_vector_udf = udf(lambda l: Vectors.dense(l), VectorUDT())
df = df.select(
    df["label"], 
    list_to_vector_udf(df["features"]).alias("vector_features")
)

print("ML ALGORITHM RANDOM FOREST CLASSIFIER")
'''
indexer = StringIndexer(inputCol="label", outputCol="label_indexed")
indexed_df = indexer.fit(df).transform(df)
indexed.show()

IL MODELLO NON ACCETTA ARRAY MA SOLO VECTOR TYPE, QUINDI DOBBIAMO RICONVERTIRE
'''

labelIndexer = StringIndexer(inputCol="label", outputCol="label_indexed").fit(df)

(trainingData, testData) = df.randomSplit([0.7, 0.3])

rf = RandomForestClassifier(labelCol="label_indexed", featuresCol="features", numTrees=10)

labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                               labels=labelIndexer.labels)
                               
pipeline = Pipeline(stages=[labelIndexer, rf, labelConverter])

model = pipeline.fit(trainingData)

predictions = model.transform(testData)

predictions.select("predictedLabel", "label", "vector_features").show(5)


evaluator = MulticlassClassificationEvaluator(
    labelCol="label_indexed", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g" % (1.0 - accuracy))

spark.stop()

