# ================================== #
#        Created by Hui Hu           #
#   Recognition using Deep Networks  #
# ================================== #


import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision

from model_training import MyNetwork, drawNLLL, test, train_network


# greek data set transform
class GreekTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        x = torchvision.transforms.functional.affine(x, 0, (0, 0), 28 / 3024, 0)
        x = torchvision.transforms.functional.center_crop(x, (28, 28))
        return torchvision.transforms.functional.invert(x)


def model_extension():
    # Configure
    epochs = 5
    batch_size = 5
    greek_training_set_path = "greek_extension/train"
    greek_test_set_path = "greek_extension/test"

    # =============================== #
    #      Load Model and DataSet     #
    # =============================== #

    # Load the saved model from a file
    greek_model = MyNetwork()
    greek_model.load_state_dict(torch.load("my_network.pth"))

    # Freezes the parameters for the whole network
    for param in greek_model.parameters():
        param.requires_grad = False

    # Replace the last layer
    linear_greek = nn.Linear(in_features=50, out_features=10)
    log_softmax = nn.LogSoftmax(dim=1)
    fc2 = nn.Sequential(linear_greek, log_softmax)
    greek_model.fc2 = fc2

    # DataLoader for the Greek data set
    greek_train_set = torchvision.datasets.ImageFolder(
        greek_training_set_path,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                GreekTransform(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    )

    greek_test_set = torchvision.datasets.ImageFolder(
        greek_test_set_path,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                GreekTransform(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    )

    greek_train_loader = torch.utils.data.DataLoader(
        greek_train_set,
        batch_size=batch_size,
        shuffle=True,
    )

    greek_test_loader = torch.utils.data.DataLoader(
        greek_test_set,
        batch_size=1,
        shuffle=False,
    )

    # Show first 9 example Greek letters
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(8, 6))
    for i in range(9):
        img, label = greek_train_set[i * 10]
        row = i // 3
        col = i % 3
        axs[row, col].imshow(cv2.UMat(img[0].numpy()).get(), cmap="gray")
        # axs[row, col].set_title(f"Label: {label}")
    fig.savefig("results/greek_data_example.png")

    # =============================== #
    #            Train Model          #
    # =============================== #
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i * len(greek_train_loader.dataset) for i in range(epochs + 1)]

    # Train model
    greek_model.train()
    test(greek_model, greek_test_loader, test_losses)
    for epoch in range(1, epochs + 1):
        train_network(
            greek_model, greek_train_loader, train_losses, train_counter, epoch, batch_size=batch_size, log_interval=1
        )
        test(greek_model, greek_test_loader, test_losses)
    drawNLLL(train_losses, train_counter, test_losses, test_counter, "extension_greek_epoch_all")


if __name__ == "__main__":
    model_extension()
