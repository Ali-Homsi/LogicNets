import tensorflow as tf
import datetime
import numpy as np
from argparse import ArgumentParser
from models import precalculation
from models import traditional_bnn
from models import traditional_bnn_without_depthwise

from tensorboard.plugins.hparams import api as hp
from dataloader import extract_ecg_file_list, extract_data

import sys
import os
from matplotlib import pyplot as plt


def main(args):
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--model", help="select the model to train",
                            choices=["precalculation",
                                     "traditional_bnn",
                                     "traditional_bnn_without_depthwise"], required=True)
    arg_parser.add_argument("--datapath", help="absolute path to the folder which contains the csv files with ECG data", required=True)
    args = arg_parser.parse_args(args)
    
    x_train, y_train = extract_data(extract_ecg_file_list(args.datapath, "train.csv"))
    x_valid, y_valid = extract_data(extract_ecg_file_list(args.datapath, "validation.csv"))

    x_train_chI = np.delete(x_train, 1, axis=2)
    x_valid_chI = np.delete(x_valid, 1, axis=2)

    metadata = {
        "learning_rate": 45e-4,
        "epochs": 400,
        "architecture": args.model,
    }

    models = {
        "precalculation": precalculation,
        "traditional_bnn": traditional_bnn,
        "traditional_bnn_without_depthwise": traditional_bnn_without_depthwise,
    }
    build_model = models[metadata['architecture']]

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = "runs/{}".format(timestamp)
    model = build_model(input_shape=np.shape(x_train_chI[0]), metadata=metadata)
    model.fit(
        x=x_train_chI,
        y=y_train,
        epochs=metadata['epochs'],
        batch_size=512,
        validation_data=(x_valid_chI, y_valid),
        callbacks=[
            tf.keras.callbacks.TensorBoard(log_dir=log_dir),
            hp.KerasCallback(log_dir, metadata),
            tf.keras.callbacks.ModelCheckpoint(
                filepath='{}.hdf5'.format(timestamp),
                monitor='val_loss',
                save_best_only=True,
            )],
        verbose=1
    )


if __name__ == "__main__":
    main(sys.argv[1:])
