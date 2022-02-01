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
    for element in npzfile[key]:
        print(element)
    print("number of elements: ")
    print(len(npzfile[key]))


'''
# Using PyTorch Cosine Similarity
cos = nn.CosineSimilarity(dim=1, eps=1e-6)
door_face_sim = cos(pic_one_vector.unsqueeze(0), pic_two_vector.unsqueeze(0))
door_stairs_sim = cos(pic_one_vector.unsqueeze(0), pic_three_vector.unsqueeze(0))
stairs_face_sim = cos(pic_two_vector.unsqueeze(0), pic_three_vector.unsqueeze(0))
'''
# print('\nCosine similarity DOOR FACE: {0}\n'.format(door_face_sim ))
# print('\nCosine similarity DOOR STAIRS: {0}\n'.format(door_stairs_sim))
# print('\nCosine similarity FACE STAIRS: {0}\n'.format(stairs_face_sim ))

