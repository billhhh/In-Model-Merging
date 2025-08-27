"""
In-Model-Merging
"""

import cxr_dataset as CXR
import eval_model as E
import model as M
import argparse
import warnings
import torch
import random
import numpy as np

warnings.filterwarnings("ignore")


def get_arguments():
    """
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default='in_model_merge_vgg19_b324')
    parser.add_argument("--model", type=str, default='vgg19')
    parser.add_argument("--b", type=int, default=324)
    parser.add_argument("--tolerance", type=int, default=8)
    return parser


# NIH images
parser = get_arguments()
print(parser)
args = parser.parse_args()

PATH_TO_IMAGES = "path_to_data/chestxray/all_images"
WEIGHT_DECAY = 1e-4
LEARNING_RATE = 0.1
NUM_EPOCHS = 30
BATCH_SIZE = args.b

preds, aucs = M.train_cnn(args, PATH_TO_IMAGES, LEARNING_RATE, WEIGHT_DECAY,
                          NUM_EPOCHS, BATCH_SIZE)
