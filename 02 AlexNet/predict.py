import json
import os.path

import torch
from torch import nn
from torchvision import transforms
from model import AlexNet
from PIL import Image
import matplotlib.pyplot as plt

device="cuda" if torch.cuda.is_available() else "cpu"

data_transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
#load data
img_path="./tulip.jpg"
assert os.path.exists(img_path), "file: '{}' does not exist".format(img_path)
img=Image.open(img_path)#打开
plt.imshow(img)#在绘图区添加一个对象

img=data_transform(img)#[C,H,W]
img=torch.unsqueeze(img,dim=0)#增加一个维度->[N,C,H,W]

# read class_indice
json_path='./class_indices.json'
assert os.path.exists(json_path), "file: '{}' does not exist".format(json_path)

with open(json_path,"r") as file:
    class_indice=json.load(file)

#create model
model=AlexNet(num_classes=5).to(device)

#load model weights
weights_path="./AlexNet.pth"
assert os.path.exists(weights_path), "file: '{}' does not exist".format(weights_path)
model.load_state_dict(torch.load(weights_path))

model.eval()
with torch.no_grad():
    output1=model(img.to(device))
    output2=torch.squeeze(output1)
    output=torch.squeeze(model(img.to(device)))#二维数组[1,num_classes]->一维[num_classes] ：tensor([-2.1923, -4.5021,  1.2517, -2.1776,  1.8861])
    predict=torch.softmax(output,dim=0)#tensor([0.0108, 0.0011, 0.3386, 0.0110, 0.6385])
    predict_cla=torch.argmax(predict).numpy()#4
# print(class_indice[str(predict_cla)],predict[predict_cla].item())
# plt.show()#显示创建的对象
print_res = "class: {}   prob: {:.3}".format(class_indice[str(predict_cla)],
                                             predict[predict_cla].numpy())
plt.title(print_res)
for i in range(len(predict)):
    print("class: {:10}   prob: {:.3}".format(class_indice[str(i)],
                                              predict[i].numpy()))
plt.show()
