# ================================== #
#        Created by Hui Hu           #
#   Recognition using Deep Networks  #
# ================================== #

import os
import sys

from model_examine import model_examine
from model_experiment import model_experiment
from model_extension import model_extension
from model_training import model_training
from model_transfer import model_transfer


# Main function
def main(argv):
    # handle any command line arguments in argv
    modes = ["train", "examine", "transfer", "experiment", "extension"]
    if len(argv) != 2 or argv[1] not in modes:
        print("Please choose one mode:")
        for mode in modes:
            print(f"    {mode}")
    else:
        mode = argv[1]
        trained_model_path = "my_network.pth"
        trained = os.path.exists(trained_model_path)
        if mode == "train":
            model_training()
        elif mode == "examine":
            if not trained:
                model_training()
            model_examine()
        elif mode == "transfer":
            if not trained:
                model_training()
            model_transfer()
        elif mode == "extension":
            if not trained:
                model_training()
            model_extension()
        else:
            model_experiment()


if __name__ == "__main__":
    main(sys.argv)
