import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from torchvision import datasets, transforms
import argparse
import matplotlib.pyplot as plt
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--epochs", help="the number of training epochs to run", type=int, default=10)
parser.add_argument("--lr", help="the learning rate of the optimizer", type=float, default=0.001)
parser.add_argument("--momentum", help="the momentum of the optimizer", type=float, default=0.9)
parser.add_argument("--batchsize", help="The size of the training batches", type=int, default=64)
args = parser.parse_args()


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        print(x.shape, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def plot_data(data, interval, file_name, title="Title"):
    fig = plt.figure(figsize=(8, 8))


    ax = plt.axes()
    x = np.linspace(interval, len(data) * interval, num=len(data))
    ax.plot(x, data)

    ax.set_xlabel("Training Samples", fontsize=30)
    ax.set_ylabel("Crossentropy Loss", fontsize=30)
    plt.title(title, fontsize=50)

    plt.savefig(file_name)


def main():
    device_ = torch.device("cuda")
    transform = transforms.Compose([transforms.ToTensor()])

    train_set = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=args.batchsize, shuffle=True, pin_memory=True)

    test_set = datasets.MNIST(root="data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=args.batchsize, shuffle=True, pin_memory=True)

    model = Net().to(device_)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)


    losses = []
    report_interval = 300
    for epoch in range(args.epochs):
        running_loss = 0.
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader, start=1):
            data, target = data.to(device_), target.to(device_)
            optimizer.zero_grad()
            outputs = model(data)
            loss = loss_function(outputs, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if batch_idx % report_interval == 0:
                print(f"loss: {running_loss}")
                losses.append(running_loss)
                running_loss = 0.

    plot_data(losses, report_interval, "results.png", "Loss")

    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            data, target = data
            data, target = data.to(device_), target.to(device_)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    print(f"Accuracy of the network on the test images: {100 * correct / total}")

if __name__ == "__main__":
    main()