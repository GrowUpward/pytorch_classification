import json
import os.path
import sys

import torch
import torchvision
from torch import nn
from torchvision import transforms,datasets
from tqdm import tqdm

from model import AlexNet

import matplotlib.pyplot as plt
import numpy as np


device="cuda" if torch.cuda.is_available() else"cpu"
print("using {} device".format(device))

data_transform={
    "train":transforms.Compose([transforms.RandomResizedCrop(224),#随机裁剪
                                transforms.RandomHorizontalFlip(),#随机翻转
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ]),
    "val":transforms.Compose([transforms.Resize((224,224)),# cannot 224, must (224, 224)
                              transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                              ])
}
data_root=os.path.abspath(os.path.join(os.getcwd(),'../'))# get data root path
image_path=os.path.join(data_root,"data","flower_data") # get flower dataset
assert os.path.exists(image_path),"{} path does not exist".format(image_path)
train_dataset=datasets.ImageFolder(root=os.path.join(image_path,"train"),
                                   transform=data_transform["train"])

train_num=len(train_dataset)#训练集中图片的数量
#{'daisy': 0, 'dandelion': 1, 'roses': 2, 'sunflowers': 3, 'tulips': 4}
flower_list=train_dataset.class_to_idx
#转变为-> {0: 'daisy', 1: 'dandelion', 2: 'roses', 3: 'sunflowers', 4: 'tulips'}
cla_dict=dict((val,key)for key,val in flower_list.items())
# write dict into json file
json_str=json.dumps(cla_dict,indent=4)
with open('class_indices.json','w') as json_file:
    json_file.write(json_str)

batch_size=32
train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=0)

validate_dataset=datasets.ImageFolder(root=image_path+"/val",
                                  transform=data_transform["val"])
val_num=len(validate_dataset)
validate_loader=torch.utils.data.DataLoader(validate_dataset,batch_size=batch_size,shuffle=False,num_workers=0)

# test_data_iter = iter(validate_loader)
# test_image, test_label = next(test_data_iter)
#
# def imshow(img):
#     img = img / 2 + 0.5  # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
#
# print(' '.join('%5s' % cla_dict[test_label[j].item()] for j in range(4)))
# imshow(torchvision.utils.make_grid(test_image))

net=AlexNet(num_classes=5,init_weight=True)
net.to(device)
loss_function=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(net.parameters(),lr=0.002)

epoches=3
best_acc=0.0
save_path='./AlexNet.pth'
train_steps = len(train_loader)
for epoch in range(epoches):
    # train
    net.train()
    running_loss=0.0
    train_bar=tqdm(train_loader,file=sys.stdout)
    for step,data in enumerate(train_bar):
        images,labels=data
        optimizer.zero_grad()
        outputs=net(images.to(device))
        loss=loss_function(outputs,labels.to(device))
        loss.backward()
        optimizer.step()

        # print statistic
        running_loss += loss.item()
        train_bar.desc="train epoch[{}/{}] loss:{:.3f}".format(epoch+1,epoches,loss)

    #validate
    net.eval()
    acc=0.0
    with torch.no_grad():
        val_bar=tqdm(validate_loader,file=sys.stdout)
        for val_data in val_bar:
            val_images,val_labels=val_data
            outputs=net(val_images.to(device))#[batch,num_classes]
            pred=torch.max(outputs,dim=1)[1]
            acc+=torch.eq(pred,val_labels.to(device)).sum().item()
    val_accurate=acc/val_num
    print('[epoch %d] training loss:%.3f val_accuracy:%.3f' %
          (epoch+1,running_loss/ train_steps,val_accurate))

    if val_accurate>best_acc:
        best_acc=val_accurate
        torch.save(net.state_dict(),save_path)
print("Finished Training!")

