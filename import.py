import torch
import glob
import json
import numpy as np
import torch.nn as nn
from PIL import Image
import torchvision.models as models
from torch.autograd import Variable
from pyspark.ml.linalg import Vectors
import torchvision.transforms as transforms

outfile = 'dataset.npz'

npzfile = np.load(outfile)
print(npzfile.files)

for key in npzfile.files:
    print("Processing key: " + str(key))
    print("number of elements: ")
    print(len(npzfile[key]))
    print("length of elements: ")
    print(len(npzfile[key][0]))


