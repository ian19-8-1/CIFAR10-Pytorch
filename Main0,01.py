import torch
import torchvision
import torchvision.transforms as transforms
from Dataset import CIFAR10

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

### Hyperparameters

batch_size = 32
num_workers = 2
lr = 0.001
wd = 0.01
num_epochs = 400



### Load and normalize CIFAR10

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data',
                                        train=True,
                                        download=True,
                                        transform=transform)
# trainset = CIFAR10(root='./data',
#                    train=True,
#                    transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,                 # ???
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=num_workers)

testset = torchvision.datasets.CIFAR10(root='.data',
                                       train=False,
                                       download=True,
                                       transform=transform)
# testset = CIFAR10(root='./data',
#                   train=False,
#                   transform=transform)
testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         num_workers=num_workers)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



### Neural Network model

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = Net()


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)


writer = SummaryWriter()



### Loss function and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)



### Training

def test(epoch, model):

    running_loss = 0.0
    correct = 0
    total = 0
    accuracy = 0.0
    tot_loss = 0.0

    with torch.no_grad():
        for i, data in enumerate(testloader):
            images, labels = data[0].to(device), data[1].to(device)
            # images, labels = data

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            loss = criterion(outputs, labels)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            accuracy = correct / total

            running_loss += loss.item()
            tot_loss = running_loss / 2000

            if i % 100 == 99:
                # print("TEST CHECKPT")
                running_loss = 0.0  # ???
                correct = 0
                total = 0
    # print("total: ", total)
    # print("correct: ", correct)
    print('Test accuracy: %.3f' % accuracy)

    writer.add_scalar("Loss/test", tot_loss, epoch)
    writer.add_scalar("Accuracy/test", accuracy, epoch)





for epoch in range(num_epochs):

    running_loss = 0.0
    total = 0
    correct = 0
    epoch_acc = 0.0
    epoch_loss = 0.0
    for i, data in enumerate(trainloader, 0):

        inputs, labels = data[0].to(device), data[1].to(device)     # ???
        # inputs, labels = data

        optimizer.zero_grad()

        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        loss = criterion(outputs, labels)       # ???
        loss.backward()                         # ???
        optimizer.step()                        # ???

        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        epoch_acc = correct / total

        running_loss += loss.item()
        epoch_loss = running_loss / 2000
        if i % 1000 == 999:
            # print("TRAIN CHECKPT")
            running_loss = 0.0      # ???
            correct = 0
            total = 0

        # print("Index: ", i)

    print('epoch: %d loss: %.3f, accuracy: %.3f' %
          (epoch + 1,  epoch_loss, epoch_acc))
    writer.add_scalar("Loss/train", epoch_loss, epoch)
    writer.add_scalar("Accuracy/train", epoch_acc, epoch)

    test(epoch, model)


writer.flush()
writer.close()
print('Finished')