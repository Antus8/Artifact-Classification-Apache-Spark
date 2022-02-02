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


dataset_feature_vectors = {"door":[], "face" : [], "stairs" : [], "pedestrian" : [],}

doors_paths = glob.glob("/home/antonello/Scrivania/Dataset_complete/Doors/*")
faces_paths = glob.glob("/home/antonello/Scrivania/Dataset_complete/Faces/*")
stairs_paths = glob.glob("/home/antonello/Scrivania/Dataset_complete/Stairs/*")
pedestrian_paths = glob.glob("/home/antonello/Scrivania/Dataset_complete/Pedestrians/*")

# output_file = open("output_doors.txt", "w")

# Load the pretrained model
model = models.resnet18(pretrained=True)

# Use the model object to select the desired layer
layer = model._modules.get('avgpool')

model.eval()

scaler = transforms.Resize((224, 224))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()

def get_vector(image_name):
    img = Image.open(image_name)    
    
    t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))
    img_feature_vector = torch.zeros(512)  # The 'avgpool' layer has an output size of 512, this is an empty vector that will hold features
    
    def copy_data(m, i, o):
    	img_feature_vector.copy_(o.data.reshape(o.data.size(1)))
    
    h = layer.register_forward_hook(copy_data)    
    
    model(t_img)    
    h.remove() # Detach copy function from the layer
    
    return img_feature_vector
print("Processing doors")
for img_path in doors_paths:
    image_feature_vector = get_vector(img_path)
    np_arr = image_feature_vector.cpu().detach().numpy()
    dataset_feature_vectors["door"].append(np_arr)

print("Processing faces")
for img_path in faces_paths:
    image_feature_vector = get_vector(img_path)
    np_arr = image_feature_vector.cpu().detach().numpy()
    dataset_feature_vectors["face"].append(np_arr)

print("Processing stairs")
for img_path in stairs_paths:
    image_feature_vector = get_vector(img_path)
    np_arr = image_feature_vector.cpu().detach().numpy()
    dataset_feature_vectors["stairs"].append(np_arr)

print("Processing pedestrians")
for img_path in pedestrian_paths:
    image_feature_vector = get_vector(img_path)
    np_arr = image_feature_vector.cpu().detach().numpy()
    dataset_feature_vectors["pedestrian"].append(np_arr)   

outfile = 'dataset.npz'
np.savez(outfile, **dataset_feature_vectors)

'''
# Using PyTorch Cosine Similarity
cos = nn.CosineSimilarity(dim=1, eps=1e-6)
door_sim = cos(npzfile["door"][0], npzfile["door"][1])
face_sim = cos(npzfile["face"][0], npzfile["face"][0])
stairs_sim = cos(npzfile["stairs"][0], npzfile["stairs"][0])
print('\nCosine similarity DOOR FACE: {0}\n'.format(door_sim ))
print('\nCosine similarity DOOR STAIRS: {0}\n'.format(face_sim))
print('\nCosine similarity FACE STAIRS: {0}\n'.format(stairs_sim ))


door_face_sim = cos(npzfile["door"][0].unsqueeze(0), npzfile["face"][0].unsqueeze(0))
door_stairs_sim = cos(npzfile["door"][0].unsqueeze(0), npzfile["stairs"][0].unsqueeze(0))
face_stairs_sim = cos(npzfile["face"][0].unsqueeze(0), npzfile["stairs"][0].unsqueeze(0))
print('\nCosine similarity DOOR FACE: {0}\n'.format(door_face_sim ))
print('\nCosine similarity DOOR STAIRS: {0}\n'.format(door_stairs_sim))
print('\nCosine similarity FACE STAIRS: {0}\n'.format(face_stairs_sim ))
'''