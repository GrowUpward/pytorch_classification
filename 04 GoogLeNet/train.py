import json
import sys

import torch
from torch import nn
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import GoogLeNet

device="cuda" if torch.cuda.is_available() else "cpu"
print("using {} device".format(device))

data_transform={
    "train":transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ]),
    "val":transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
}
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

# model_name="vgg16"
# net=vgg(model_name=model_name,num_classes=5,init_weights=True)
net=GoogLeNet(num_classes=5,aux_logits=True,init_weights=True)
net=net.to(device)
loss_fn=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(net.parameters(),lr=0.0001)

epoches=1
val_num=len(validate_data)
save_path='./GoogLeNet.pth'
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
        # outputs=net(images.to(device))
        # loss = loss_fn(outputs, labels.to(device))
        logits,aux_logits2,aux_logits1=net(images.to(device))#有三个输出
        loss0=loss_fn(logits,labels.to(device))
        loss1=loss_fn(aux_logits1,labels.to(device))
        loss2=loss_fn(aux_logits2,labels.to(device))
        loss=loss0+0.3*loss1+0.3*loss2#对于辅助分类器的损失，按0.3的权重加入到总损失中
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
        val_bar.desc="[epoch {}] training loss:{:.3f} val accuracy:{:.3f}".format(epoch+1,running_loss,val_accuracy)

    if val_accuracy > best_acc:
        best_acc = val_accuracy
        torch.save(net.state_dict(), save_path)
print("Finished Training!")
