import torch
import glob
import json
import numpy as np
import pandas as pd
import torch.nn as nn
from PIL import Image
import torchvision.models as models
from torch.autograd import Variable
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark import SparkContext,SparkConf
import torchvision.transforms as transforms
from pyspark.sql.types import StringType, ArrayType,StructType,StructField, FloatType


spark = SparkSession.builder.appName('Artifact Classification').getOrCreate()
outfile = '/home/antonello/spark-3.2.0-bin-hadoop3.2/Classification/dataset.npz'

npzfile = np.load(outfile)

dataset = []
    	
for key in npzfile.files:
    for feature_vector in npzfile[key]:
    	dataset.append({"label" : key, "features" : feature_vector})

print(len(dataset))

df = pd.DataFrame(dataset)
print(df)

spark.stop()

