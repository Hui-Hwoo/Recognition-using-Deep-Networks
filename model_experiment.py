# ================================== #
#        Created by Hui Hu           #
#   Recognition using Deep Networks  #
# ================================== #

import torch
import torch.nn as nn
import torchvision

from model_training import drawNLLL, test, train_network


# Class definitions for experiment
class expNetwork(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(expNetwork, self).__init__()
        # Convolutional layer with 10 5x5 filters
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)

        # Max pooling layer with a 2x2 window and a ReLU function applied
        # Create the max pooling layer
        maxpool_layer = nn.MaxPool2d(kernel_size=2)
        # Create the ReLU activation function
        relu = nn.ReLU()
        # Combine the max pooling and ReLU layers using nn.Sequential
        self.pool1 = nn.Sequential(maxpool_layer, relu)

        # Convolutional layer with 20 5x5 filters
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)

        # Dropout layer with a 0.5 dropout rate (50%)
        self.dropout = nn.Dropout(p=dropout_rate)

        # Max pooling layer with a 2x2 window and a ReLU function applied
        self.pool2 = nn.Sequential(maxpool_layer, relu)

        # A flattening operation followed by a fully connected Linear layer with 50 nodes and a ReLU function
        # Flattening operation
        flatten = nn.Flatten()
        # Fully connected linear layer with 50 nodes
        linear1 = nn.Linear(in_features=320, out_features=50)
        self.fc1 = nn.Sequential(flatten, linear1, relu)

        # Fully connected linear layer with 10 nodes and the log_softmax function
        linear2 = nn.Linear(in_features=50, out_features=10)
        # Log_softmax function
        log_softmax = nn.LogSoftmax(dim=1)
        self.fc2 = nn.Sequential(linear2, log_softmax)

    # computes a forward pass for the network
    def forward(self, x):
        # Pass input through the first convolutional layer
        x = self.conv1(x)
        # Apply max pooling and ReLU function
        x = self.pool1(x)
        # Pass through the second convolutional layer
        x = self.conv2(x)
        # Apply dropout
        x = self.dropout(x)
        # Apply max pooling with ReLU function
        x = self.pool2(x)
        # Flatten, pass Linear layer and apply ReLU function
        x = self.fc1(x)
        # Linear layer and apply log_softmax function
        x = self.fc2(x)

        return x


def model_experiment():
    # ================================ #
    #            Load Dataset          #
    # ================================ #

    # Load MNIST data set
    train_set = torchvision.datasets.MNIST(
        root="./data/",
        train=True,
        download=True,
        transform=torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )
    test_set = torchvision.datasets.MNIST(
        root="./data/",
        train=False,
        download=True,
        transform=torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )

    # Define the training and test data loaders with the specified batch size
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=64,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=1000,
        shuffle=True,
    )

    # ================================ #
    #    Find optimal epoch numbers    #
    # ================================ #

    model = expNetwork().to("cpu")
    train_losses = []
    train_counter = []
    test_losses = []

    # Loop over the epochs
    epoch = 0
    test(model, test_loader, test_losses)
    while epoch < 2 or (test_losses[-1] < test_losses[-2]):
        epoch += 1
        train_network(model, train_loader, train_losses, train_counter, epoch)
        test(model, test_loader, test_losses)

    test_counter = [i * len(train_loader.dataset) for i in range(epoch + 1)]
    drawNLLL(train_losses, train_counter, test_losses, test_counter, "experiment_optimal_epoch")

    print("Optimal epoch is", epoch - 1)

    # ================================ #
    #    Find optimal dropout rates    #
    # ================================ #

    result = []
    drop_rate = 0.1
    while drop_rate < 1:
        print("Drop rate:", drop_rate)
        model = expNetwork(dropout_rate=drop_rate).to("cpu")
        epoch = 0
        test(model, test_loader, test_losses)
        while epoch < 7:
            epoch += 1
            train_network(model, train_loader, train_losses, train_counter, epoch)
            test(model, test_loader, test_losses)
        result.append(test_losses[-1])
        drop_rate += 0.1

    print("Optimal drop rate is", (result.index(max(result)) + 1) * 0.1)


if __name__ == "__main__":
    model_experiment()
