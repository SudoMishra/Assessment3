from tensorflow import keras
import pandas as pd
from glob import glob
import numpy as np
from keras.layers import (
    Input,
    Dense,
    LSTMCell,
    RNN,
    Conv2D,
    BatchNormalization,
    MaxPool2D,
    Dropout,
    Flatten,
    Concatenate,
    AvgPool2D,
)
from keras.regularizers import L2
from keras.applications.mobilenet_v3 import MobileNetV3Small
from keras.applications.mobilenet_v3 import preprocess_input as mnet_small_preprocess
from keras.applications.efficientnet import EfficientNetB0
from keras.applications.efficientnet import preprocess_input as effnet_preprocess
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input as mnet_preprocess
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import os
import argparse
from math import prod

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import seaborn as sns

parser = argparse.ArgumentParser(description="Action Recognition on UCF 101 Model")
parser.add_argument(
    "--model", choices=["LSTM", "ANN"], default="LSTM", help="Model Type"
)
parser.add_argument(
    "--pretrained",
    choices=["effnet", "mobile", "mobilenet"],
    default="mobile",
    help="Pretrained Feature Extractor Type",
)
parser.add_argument(
    "--features",
    choices=["effnet", "mobile", "mobilenet"],
    default="effnet",
    help="Extracted Feature Type",
)
parser.add_argument("--lr", type=float, default=1e-3, help="Value of Learning Rate")
parser.add_argument("--epochs", type=int, default=10, help="Number of Epochs")
parser.add_argument("--batch_size", type=int, default=32, help="Batch Size")
parser.add_argument("--callbacks", action="store_true", help="Use Callbacks")
parser.add_argument("--extract_features", action="store_true", help="Extract Features")
args = parser.parse_args()

IMAGE_DIMS = (128, 128)


def get_data(label_file, feature_file, mode):
    features = np.load(feature_file)
    feature_shape = features.shape
    features = features.reshape(-1, 3, prod(feature_shape[2:]))
    if mode == "Test":
        labels = None
    else:
        df = pd.read_csv(label_file)
        labels = np.array(df["labels"])
        labels -= 1.0
        labels = keras.utils.to_categorical(labels, num_classes=10, dtype="float32")
    # data = zip(features,labels)
    return features, labels


def main(args):
    x_test, _ = get_data(
        "test.csv", f"./data/{args.features}_Test_img_features.npy", "Test"
    )
    model = keras.models.load_model(f"saved_models/{args.model}_{args.features}_Model")
    y_preds = model.predict(x_test)
    y_preds = np.argmax(y_preds, axis=1)
    y_preds += 1
    y_preds = y_preds.astype(np.int32)
    df = pd.DataFrame()
    # df["labels"] = y_preds
    df["Id"] = np.arange(0, len(y_preds))
    df["Class"] = y_preds
    df.to_csv(f"{args.features}_{args.model}_submission.csv", index=False)


if __name__ == "__main__":
    main(args)
