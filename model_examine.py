# ================================== #
#        Created by Hui Hu           #
#   Recognition using Deep Networks  #
# ================================== #

import imghdr  # check image file
import os
from os import walk

import cv2
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms

from model_training import MyNetwork


def model_examine():
    # ======================================== #
    #             Get necessary data           #
    # ======================================== #

    # Load the saved model from a file
    model = MyNetwork()
    model.load_state_dict(torch.load("my_network.pth"))

    # Load test dataset
    test_set = torchvision.datasets.MNIST(
        root="./data/",
        train=False,
        download=True,
        transform=torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
    )

    # Create results directory to save images if not exists
    directory = "results"
    os.makedirs(directory, exist_ok=True)

    # ======================================== #
    #         Predict on Test Dataset          #
    # ======================================== #

    model.eval()
    fig0, axs0 = plt.subplots(3, 3)

    for batch_idx, (data, target) in enumerate(test_loader):
        if batch_idx < 9:
            image = cv2.UMat(test_set[batch_idx][0][0].numpy())
            output = model(data)
            lst = list(output.detach().numpy()[0])
            predication = lst.index(max(lst))
            row = batch_idx // 3
            col = batch_idx % 3
            axs0[row, col].title.set_text(f"Prediction: {predication}")
            axs0[row, col].imshow(image.get(), cmap="gray")
            axs0[row, col].set_xticks([])
            axs0[row, col].set_yticks([])

            rounded_lst = [round(value, 2) for value in lst]
            print(
                f"{batch_idx+1}:\n\tOutput values: {rounded_lst}\n\tindex of the max output value: {predication}\n\tcorrect label of the digit: {target}"
            )
        else:
            break

    # Display the first 9 predictions
    fig0.subplots_adjust(hspace=0.3)  # modify the space between two image
    fig0.savefig("results/predict_test_digit.png")
    # plt.show()

    # ======================================== #
    #          Predict on New Digits           #
    # ======================================== #

    # Get all image files and corresponding labels
    dir_path = "new_digits"
    new_digits = []

    mean_sum = 0
    std_sum = 0
    count = 0
    convert_tensor = transforms.ToTensor()
    for dir_path, dir_names, file_names in walk(dir_path):
        for file_name in file_names:
            # Only collect image file
            if imghdr.what(dir_path + "/" + file_name):
                image = cv2.imread(dir_path + "/" + file_name, cv2.IMREAD_GRAYSCALE)
                data = convert_tensor(cv2.subtract(1, image))

                mean_sum += torch.mean(data)
                std_sum += torch.std(data)
                count += 1

                name = file_name.split(".")[0].split("_")[0]
                new_digits.append((data, int(name)))

    # Normalize image files
    normalize = transforms.Normalize((mean_sum / count,), (std_sum / count,))
    for idx, (data, label) in enumerate(new_digits):
        new_digits[idx] = (normalize(data), label)

    # Load dataset
    new_digits_loader = torch.utils.data.DataLoader(
        new_digits,
        batch_size=1,
        shuffle=False,
    )

    model.eval()
    fig1, axs1 = plt.subplots(4, 3)

    for idx, (data, target) in enumerate(new_digits_loader):
        image = cv2.UMat(new_digits[idx][0][0].numpy())
        output = model(data)
        lst = list(output.detach().numpy()[0])
        predication = lst.index(max(lst))
        if idx < 9:
            row = idx // 3
            col = idx % 3
            axs1[row, col].title.set_text(f"Prediction: {predication}")
            axs1[row, col].imshow(image.get(), cmap="gray")
            axs1[row, col].set_xticks([])
            axs1[row, col].set_yticks([])
        else:
            axs1[3, 1].title.set_text(f"Prediction: {predication}")
            axs1[3, 1].imshow(image.get(), cmap="gray")
            axs1[3, 1].set_xticks([])
            axs1[3, 1].set_yticks([])

    fig1.delaxes(axs1[3][0])  # remove empty image
    fig1.delaxes(axs1[3][2])

    # Display the predictions
    fig1.subplots_adjust(hspace=0.5)  # modify the space between two image
    fig1.savefig("results/predict_new_digits.png")
    # plt.show()

    # ======================================== #
    #          Analyze the first layer         #
    # ======================================== #

    # Get the weights of the first layer
    weights = model.conv1.weight

    # print(model)          # Print the model structure
    # print(weights.shape)  # should be [10, 1, 5, 5]

    # Visualize the filters
    fig, axs = plt.subplots(3, 4)
    for i in range(10):
        row = i // 4
        col = i % 4
        axs[row, col].title.set_text(f"Filter {i}")
        axs[row, col].imshow(weights[i, 0].detach().numpy())
        axs[row, col].set_xticks([])
        axs[row, col].set_yticks([])

    fig.subplots_adjust(hspace=0.3)  # modify the space between two image
    fig.delaxes(axs[2][2])  # remove empty image
    fig.delaxes(axs[2][3])

    fig.savefig("results/filter.png")
    # plt.show()

    # ======================================== #
    #      Show the effect of the filters      #
    # ======================================== #

    # Get the image
    index = 0
    img, _ = test_set[index]
    image = cv2.UMat(img[0].numpy())

    # Apply the 10 filters using OpenCV's filter2D function
    filtered_images = []
    for i in range(weights.shape[0]):
        filter = weights[i, 0].detach().numpy()
        filtered_image = cv2.filter2D(image, -1, filter)
        filtered_images.append(filtered_image)

    # Generate a plot of the 10 filtered images
    figs, axes = plt.subplots(nrows=5, ncols=4, figsize=(10, 10))
    for i in range(10):
        row = i // 2
        col = (i % 2) * 2
        # filter
        axes[row, col].imshow(weights[i, 0].detach().numpy(), cmap="gray")
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])
        # filtered image
        next_col = 1 if (i % 2 == 0) else 3
        axes[row, next_col].imshow(filtered_images[i].get(), cmap="gray")
        axes[row, next_col].set_xticks([])
        axes[row, next_col].set_yticks([])
    figs.savefig("results/filtered_image.png")
    # plt.show()


if __name__ == "__main__":
    model_examine()
