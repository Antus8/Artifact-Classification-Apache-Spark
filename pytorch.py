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


dataset_feature_vectors = {"door":[], "face" : [], "stairs" : []}

doors_paths = glob.glob("/home/antonello/Scrivania/Dataset/Doors/*")
faces_paths = glob.glob("/home/antonello/Scrivania/Dataset/Faces/*")
stairs_paths = glob.glob("/home/antonello/Scrivania/Dataset/Stairs/*")

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

for img_path in doors_paths:
    image_feature_vector = get_vector(img_path)
    np_arr = image_feature_vector.cpu().detach().numpy()
    dataset_feature_vectors["door"].append(np_arr)

for img_path in faces_paths:
    image_feature_vector = get_vector(img_path)
    np_arr = image_feature_vector.cpu().detach().numpy()
    dataset_feature_vectors["face"].append(np_arr)

for img_path in stairs_paths:
    image_feature_vector = get_vector(img_path)
    np_arr = image_feature_vector.cpu().detach().numpy()
    dataset_feature_vectors["stairs"].append(np_arr)    

outfile = 'dataset.npz'
np.savez(outfile, **dataset_feature_vectors)


