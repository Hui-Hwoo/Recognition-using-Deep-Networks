# ================================== #
#        Created by Hui Hu           #
#   Recognition using Deep Networks  #
# ================================== #

# import statements
import os
import sys

import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision


# Class definitions
class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
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
        self.dropout = nn.Dropout(p=0.5)

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


# Draw negative log likelihood loss chart
def drawNLLL(train_losses, train_counter, test_losses, test_counter, image_name):
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color="blue")
    plt.scatter(test_counter, test_losses, color="red")
    plt.legend(["Train Loss", "Test Loss"], loc="upper right")
    plt.xlabel("number of training examples seen")
    plt.ylabel("negative log likelihood loss")
    fig.savefig(f"results/{image_name}.png")


# Train model for one epoch
def train_network(model, train_loader, train_losses, train_counter, epoch, batch_size=64, log_interval=10):
    # Define the loss function and optimizer
    loss_fn = nn.NLLLoss()
    learning_rate = 0.01
    momentum = 0.5
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    # Define lists to store the training loss for this epoch
    epoch_train_losses = []
    epoch_train_counter = []

    # Train the model for one epoch
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # Forward pass
        output = model(data)
        loss = loss_fn(output, target)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    min(batch_idx * batch_size, len(train_loader.dataset)),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            train_losses.append(loss.item())
            train_counter.append((batch_idx * batch_size) + ((epoch - 1) * len(train_loader.dataset)))
            epoch_train_losses.append(loss.item())
            epoch_train_counter.append((batch_idx * batch_size) + ((epoch - 1) * len(train_loader.dataset)))

    return epoch_train_losses, epoch_train_counter


# Test model for one epoch
def test(model, test_loader, test_losses):
    model.eval()
    test_loss = 0
    correct = 0
    loss_fn = nn.NLLLoss(reduction="sum")  # loss function

    # Evaluate the model on the test sets
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += loss_fn(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)

    test_losses.append(test_loss)
    print(
        "\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )


# Training model
def model_training():
    # configure
    epochs = 5

    # Create results directory to save images if not exists
    directory = "results"
    os.makedirs(directory, exist_ok=True)

    # Set random seed for repeatability
    torch.manual_seed(42)
    torch.backends.cudnn.enabled = False  # turn off CUDA

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

    # Show first six example digits
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(8, 6))
    for i in range(6):
        img, label = train_set[i]
        row = i // 3
        col = i % 3
        axs[row, col].imshow(cv2.UMat(img[0].numpy()).get(), cmap="gray")
        axs[row, col].set_title(f"Label: {label}")
    fig.savefig("results/digit_data_example.png")

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # model = MyNetwork().to(device)
    # print(model)

    model = MyNetwork().to("cpu")
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i * len(train_loader.dataset) for i in range(epochs + 1)]

    # Loop over the epochs
    test(model, test_loader, test_losses)
    for epoch in range(1, epochs + 1):
        epoch_train_losses, epoch_train_counter = train_network(model, train_loader, train_losses, train_counter, epoch)
        test(model, test_loader, test_losses)
        epoch_test_losses = test_losses[: epoch + 1]
        if epoch > 1:
            epoch_test_losses[0] = 0
        epoch_test_counter = [i * len(train_loader.dataset) for i in range(epoch + 1)]
        drawNLLL(epoch_train_losses, epoch_train_counter, epoch_test_losses, epoch_test_counter, f"digit_epoch_{epoch}")

    drawNLLL(train_losses, train_counter, test_losses, test_counter, "digit_epoch_all")

    # Save the model to a file
    torch.save(model.state_dict(), "my_network.pth")

    return


if __name__ == "__main__":
    model_training()
