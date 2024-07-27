import torch
import torchvision
from torchvision import transforms
from model import LeNet
from torch import nn
# import matplotlib.pyplot as plt
# import numpy as np
def main():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    #50000张训练图片
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=False, transform=transform)
    trainloader=torch.utils.data.DataLoader(trainset,batch_size=36,shuffle=True,num_workers=0)

    #10000张测试图片
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=False, transform=transform)
    testloader=torch.utils.data.DataLoader(testset,batch_size=5000,shuffle=False,num_workers=0)

    # #获取标签
    test_data_iter=iter(testloader)
    test_image,test_label=next(test_data_iter)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    model=LeNet()
    loss_function=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=0.001)


    epoches=5
    for epoch in range(epoches):
        running_loss=0.0
        for step,data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs,labels=data
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs=model(inputs)
            loss=loss_function(outputs,labels)
            loss.backward()
            optimizer.step()#参数更新

            #print statistics
            running_loss+=loss.item()
            if step%500 == 499:
                with torch.no_grad():
                    outputs=model(test_image)#[batch,10]
                    predict_y=torch.max(outputs,dim=1)[1]#找到预测类别最可能值的下标
                    accuracy=torch.eq(predict_y,test_label).sum().item()/test_label.size(0)

                    print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                          (epoch + 1, step + 1, running_loss / 500, accuracy))
                    running_loss = 0.0
    print('Finished Training')

    save_path = './LeNet.pth'
    torch.save(model.state_dict(), save_path)

if __name__ == '__main__':
    main()
