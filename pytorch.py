import torch
import glob
import torch.nn as nn
from PIL import Image
import torchvision.models as models
from torch.autograd import Variable
import torchvision.transforms as transforms

doors_feature_vectors = {}
faces_feature_vectors = {}
stairs_feature_vectors = {}

doors_paths = glob.glob("/home/antonello/Scrivania/Dataset/Doors/*")
print(doors_paths)

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
    


'''
pic_one = "/home/antonello/Scrivania/Dataset/Doors/Door0007.png"
pic_two = "/home/antonello/Scrivania/Dataset/Faces/1_0_0_20161219190045155.jpg.chip.jpg"
pic_three = "/home/antonello/Scrivania/Dataset/Stairs/829365.jpg"

pic_one_vector = get_vector(pic_one)
pic_two_vector = get_vector(pic_two)
pic_three_vector = get_vector(pic_three)

print("VECTORS!")
print(pic_one_vector)
print("len")
print(len(pic_one_vector))
print(pic_two_vector)
print("END VECTORS!")

# Using PyTorch Cosine Similarity
cos = nn.CosineSimilarity(dim=1, eps=1e-6)
door_face_sim = cos(pic_one_vector.unsqueeze(0), pic_two_vector.unsqueeze(0))
door_stairs_sim = cos(pic_one_vector.unsqueeze(0), pic_three_vector.unsqueeze(0))
stairs_face_sim = cos(pic_two_vector.unsqueeze(0), pic_three_vector.unsqueeze(0))'''
# print('\nCosine similarity DOOR FACE: {0}\n'.format(door_face_sim ))
# print('\nCosine similarity DOOR STAIRS: {0}\n'.format(door_stairs_sim))
# print('\nCosine similarity FACE STAIRS: {0}\n'.format(stairs_face_sim ))


