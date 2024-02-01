import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing

########################################
# Class for CNN
########################################

class Net(nn.Module):
    def __init__(self):
        super().__init__();
        self.conv1 = nn.Conv2d(3, 6, 5);
        self.pool = nn.MaxPool2d(2, 2);
        self.conv2 = nn.Conv2d(6, 16, 5);
        self.fc1 = nn.Linear(16 * 5 * 5, 120);
        self.fc2 = nn.Linear(120, 84);
        self.fc3 = nn.Linear(84, 10);

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)));
        x = self.pool(F.relu(self.conv2(x)));
        x = torch.flatten(x, 1);
        x = F.relu(self.fc1(x));
        x = F.relu(self.fc2(x));
        x = self.fc3(x);
        return x;



########################################
# Helper functions 
########################################

def img_show(image):
    image = (image / 2) + 0.5;
    np_image = image.numpy()
    plt.imshow(np.transpose(np_image, (1, 2, 0)));
    plt.show();



########################################
# Loading and running data sets
########################################

def main():
    torchvision.datasets.CIFAR10.url="http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) 

    print("Loading Data");
    batch_size = 4;

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
        shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
        shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    print("Data Loaded");
    print();

    print("Testing Images / Data");
    data_iter = iter(trainloader);
    images, labels = next(data_iter);
    img_show(torchvision.utils.make_grid(images))
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))
    print();

    print("Setting up NN");
    PATH = './cifar_net.pth'

    net = Net();
    net.load_state_dict(torch.load(PATH));
    
    """ Already trained
    criterion = nn.CrossEntropyLoss();
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9);

    for epoch in range(2): # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 2000 == 1999: 
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
                print('Finished Training')

    torch.save(net.state_dict(), PATH)
    """



########################################
# Script run protection
########################################

if __name__ == "__main__":
    multiprocessing.freeze_support();
    print("Lab Four");
    main();

