import torch
from PIL import Image

from model import LeNet
from torchvision import transforms
def main():
    transform=transforms.Compose([
        transforms.Resize([32,32]),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    model=LeNet()
    model.load_state_dict(torch.load('LeNet.pth'))

    img=Image.open('airplane.jpg')
    img=transform(img)#[C,H,W]
    img=torch.unsqueeze(img,dim=0)#增加一个维度 [N,C,H,W]

    with torch.no_grad():
        output=model(img)
        predict=torch.max(output,dim=1)[1].numpy()
        print(predict) #[0]
    print(classes[(predict)])#classes[0]='airplane'

if __name__ == '__main__':
    main()