import json
import os.path
import sys

import torch
from torch import nn
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import resnet34,resnet101

from torchvision.models import resnet

device="cuda" if torch.cuda.is_available() else "cpu"
print("using {} device".format(device))

data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    "val": transforms.Compose([transforms.Resize(256),#原图片的长宽比固定不变，将其最短边长缩放为256
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
#load data
train_data=datasets.ImageFolder(root="../data/flower_data/"+"train",
                                transform=data_transform["train"])
train_loader=DataLoader(train_data,batch_size=32,shuffle=True,num_workers=0)
print(len(train_data))

validate_data=datasets.ImageFolder(root="../data/flower_data/"+"val",
                                transform=data_transform["val"])
validate_loader=DataLoader(validate_data,batch_size=32,shuffle=False,num_workers=0)
print(len(validate_data))

#write json
#获取 类名-idx
flower_list=train_data.class_to_idx#{'daisy': 0, 'dandelion': 1, 'roses': 2, 'sunflowers': 3, 'tulips': 4}
flower_dict=dict((val,key) for key,val in flower_list.items())
json_str=json.dumps(flower_dict,indent=4)
with open('class_indices.json','w') as json_file:
    json_file.write(json_str)


net=resnet34()
net=net.to(device)
model_weight_path="./resnet34-pre.pth"
assert os.path.exists(model_weight_path),"file {} does not exists".format(model_weight_path)
net.load_state_dict(torch.load(model_weight_path,map_location=device))#载入模型权重

# change fc layer
in_channel=net.fc.in_features#输入特征矩阵的深度
net.fc=nn.Linear(in_channel,5)

loss_fn=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(net.parameters(),lr=0.0001)

epoches=1
train_steps = len(train_loader)
val_num=len(validate_data)
save_path='./ResNet.pth'
best_acc=0.0
for epoch in range(epoches):
    #train +running_loss in epoch
    net.train()
    running_loss=0.0;
    train_bar=tqdm(train_loader,file=sys.stdout)
    for step,data in enumerate(train_bar):
        # data=data.to(device)#AttributeError: 'list' object has no attribute 'to'
        images,labels=data

        optimizer.zero_grad()
        outputs=net(images.to(device))
        loss = loss_fn(outputs, labels.to(device))
        loss.backward()
        optimizer.step()

        running_loss+=loss
        # print("train epoch[{}/{}] running loss:{:.3f}").format(epoch+1,epoches,running_loss)
        train_bar.desc="train epoch[{}/{}] running loss:{:.3f}".format(epoch+1,epoches,loss)


    #validate
    net.eval()
    acc = 0.0
    val_bar=tqdm(validate_loader,file=sys.stdout)
    with torch.no_grad():
        for val_data in (val_bar):
            val_images,val_labels=val_data
            outputs=net(val_images.to(device))
            pred=torch.max(outputs,dim=1)[1]
            acc+=torch.eq(pred,val_labels.to(device)).sum().item()
        val_accuracy=acc/val_num
        val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                   epoches)

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))
    if val_accuracy > best_acc:
        best_acc = val_accuracy
        torch.save(net.state_dict(), save_path)
print("Finished Training!")
