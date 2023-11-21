

from cgi import test
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn


from torchvision.io import read_image
import os
import re

from util import progress_bar
from resnet_model import ResNet18


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def test(net, testloader):
    criterion = nn.CrossEntropyLoss()
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            targets = targets.type(torch.LongTensor)
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.labels = list()
        self.images = list()
        for filename in os.listdir(self.img_dir):
          f = os.path.join(self.img_dir, filename)
          if os.path.isfile(f):
              label = re.findall(r'\d+', filename)
              self.labels.append(torch.tensor(int(label[0]), dtype=torch.int8))
              
              image = read_image(f)
              if self.transform:
                  self.images.append(self.transform(image))
              else:
                  self.images.append(image)
                  
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


img_dir = "../imagen-pytorch/fake_images_text/"
transform_fake = transforms.Compose([
    # transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float32),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.Resize(32, antialias=True)
])

testset_fake = CustomImageDataset(img_dir = img_dir,transform = transform_fake)
testloader_fake = torch.utils.data.DataLoader(testset_fake, batch_size=128, shuffle=True, num_workers=2)


transform_real = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
testset_real = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_real)
testloader_real = torch.utils.data.DataLoader(
    testset_real, batch_size=128, shuffle=False, num_workers=2)



# model = torchvision.models.resnet18()
model = ResNet18()
model = model.to(device)
if device == 'cuda':
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True
model.load_state_dict(torch.load("./checkpoint/ckpt_s.pth")['net'])



test(model, testloader_fake)
test(model, testloader_real)