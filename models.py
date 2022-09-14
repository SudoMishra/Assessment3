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
parser.add_argument("--viz", action="store_true", help="Visualise Embeddings")
args = parser.parse_args()

IMAGE_DIMS = (128, 128)


def pretrain_model(pretrained):
    inp_shape = (128, 128, 3)
    inp_seq = Input(shape=inp_shape)
    if pretrained == "mobile":
        print("Loading Mobile Net V3 Small")
        base_model = MobileNetV3Small(
            weights="imagenet", include_top=False, pooling="avg", input_shape=inp_shape
        )
    if pretrained == "effnet":
        print("Loading Efficient Net")
        base_model = EfficientNetB0(
            weights="imagenet", include_top=False, pooling="avg", input_shape=inp_shape
        )
    if pretrained == "mobilenet":
        print("Loading Mobile Net")
        base_model = MobileNet(
            weights="imagenet", include_top=False, pooling="avg", input_shape=inp_shape
        )
    for layer in base_model.layers:
        layer.trainable = False
    y = base_model(inp_seq)
    model = keras.models.Model(inputs=inp_seq, outputs=y)
    return model


def get_img(f):
    img = Image.open(f)
    img = img.resize(IMAGE_DIMS)
    return np.asarray(img)


def extract_features(file, mode, args):
    df = pd.read_csv(file)
    features = []
    labels = []
    model = pretrain_model(args.pretrained)
    if args.pretrained == "effnet":
        preprocess = effnet_preprocess
    if args.pretrained == "mobile":
        preprocess = mnet_small_preprocess
    if args.pretrained == "mobilenet":
        preprocess = mnet_preprocess
    for index, row in tqdm(df.iterrows()):
        folder_path = f'{row["name"]}*.jpg'
        video_label = row["labels"]
        fnames = sorted(glob(folder_path, recursive=True))
        imgs = preprocess(np.array([get_img(f) for f in fnames]))
        img_features = model.predict(imgs, verbose=False)
        features.append(img_features)
        labels.append(video_label)
    features = np.array(features)
    print(f"Features Shape : {features.shape}")
    with open(
        os.path.join(".", "data", f"{args.pretrained}_{mode}_img_features.npy"), "wb"
    ) as f:
        np.save(f, features)


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
    return features, labels


def fine_tune_classifier(feature_shape):
    input_shape = feature_shape
    inp_seq = Input(shape=input_shape)
    x = RNN(
        LSTMCell(128),
    )(inp_seq)
    x = Dropout(0.5)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    y = Dense(10, activation="softmax")(x)
    model = keras.models.Model(inputs=inp_seq, outputs=y)
    return model


def fine_tune_classifier_ANN(feature_shape):
    input_shape = feature_shape
    inp_seq = Input(shape=input_shape)
    x = Flatten()(inp_seq)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)
    y = Dense(10, activation="softmax")(x)
    model = keras.models.Model(inputs=inp_seq, outputs=y)
    return model


def plot_loss(train_loss, val_loss, args):
    fig, ax = plt.subplots()
    ax.plot(train_loss, label="Train")
    ax.plot(val_loss, label="Val")
    ax.set_title("Train Vs Val Loss")
    fig.savefig(f"imgs/loss/{args.features}_Train_Val_Loss.png")
    ax.legend()
    plt.close(fig)


def plot_acc(train_acc, val_acc, args):
    fig, ax = plt.subplots()
    ax.plot(train_acc, label="Train")
    ax.plot(val_acc, label="Val")
    ax.set_title("Train Vs Val Accuracy")
    fig.savefig(f"imgs/acc/{args.features}_Train_Val_Acc.png")
    ax.legend()
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, mode, args):
    cm = confusion_matrix(
        y_true, y_pred, normalize="true", labels=[i for i in range(10)]
    )
    fig, ax = plt.subplots()
    ax = sns.heatmap(cm, vmin=0, vmax=1)
    plt.savefig(f"imgs/confusion_matrix/{mode} {args.features}.png")


def visualize_tsne(x, y, mode, args):
    tsne = TSNE(n_components=2, verbose=1)
    tsne_result = tsne.fit_transform(x)
    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x=tsne_result[:, 0],
        y=tsne_result[:, 1],
        hue=y,
        palette=sns.color_palette("hls", 10),
        data=x,
        legend="full",
        alpha=0.3,
    )
    # plt.show()
    plt.savefig(f"imgs/viz/{args.features} {mode} Data TSNE.png")
    plt.close()


def visualize_pca(x, y, mode, args):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(x)
    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x=pca_result[:, 0],
        y=pca_result[:, 1],
        hue=y,
        palette=sns.color_palette("hls", 10),
        data=x,
        legend="full",
        alpha=0.3,
    )
    plt.savefig(f"imgs/viz/{args.features} {mode} Data PCA.png")
    plt.close()


def plot_cms(x, y, mode, model, args):
    y_pred = model.predict(x)
    y_true = np.argmax(y, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    plot_confusion_matrix(y_true, y_pred, mode, args)


def main(args):
    x_train, y_train = get_data(
        "train.csv", f"./data/{args.features}_Train_img_features.npy", "Train"
    )
    x_val, y_val = get_data(
        "val.csv", f"./data/{args.features}_Val_img_features.npy", "Val"
    )
    if args.viz:
        visualize_pca(
            x_train.reshape(-1, prod(x_train.shape[1:])),
            np.argmax(y_train, axis=1),
            "Train",
            args,
        )
        visualize_tsne(
            x_train.reshape(-1, prod(x_train.shape[1:])),
            np.argmax(y_train, axis=1),
            "Train",
            args,
        )
        visualize_pca(
            x_val.reshape(-1, prod(x_val.shape[1:])),
            np.argmax(y_val, axis=1),
            "Val",
            args,
        )
        visualize_tsne(
            x_val.reshape(-1, prod(x_val.shape[1:])),
            np.argmax(y_val, axis=1),
            "Val",
            args,
        )
    # exit()
    if args.model == "LSTM":
        model = fine_tune_classifier(x_train.shape[1:])
        print("Loading LSTM Model")
        print(model.summary())
    else:
        model = fine_tune_classifier_ANN(x_train.shape[1:])
        print("Loading ANN Model")

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.9, patience=5, min_lr=1e-7
    )
    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)
    if args.callbacks:
        callbacks = [reduce_lr, early_stopping]
    else:
        callbacks = []
    print(args.batch_size, args.epochs, args.lr)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    if args.model == "LSTM":
        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            shuffle=True,
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=callbacks,
        )
    else:
        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            shuffle=True,
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=callbacks,
        )
    model.save(f"saved_models/{args.model}_{args.features}_Model")

    train_loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    train_acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]

    plot_loss(train_loss, val_loss, args)
    plot_acc(train_acc, val_acc, args)
    plot_cms(x_train, y_train, "Train", model, args)
    plot_cms(x_val, y_val, "Val", model, args)


if __name__ == "__main__":
    if args.extract_features:
        extract_features("train.csv", "Train", args)
        extract_features("val.csv", "Val", args)
        extract_features("test.csv", "Test", args)
    else:
        main(args)
