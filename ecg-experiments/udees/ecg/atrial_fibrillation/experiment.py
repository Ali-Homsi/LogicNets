from udees.datasets.mitdb import AtrialFibrillation
from sklearn.model_selection import train_test_split
import tensorflow as tf
import datetime
import numpy as np
from argparse import ArgumentParser
from udees.ecg.atrial_fibrillation.models import binary_depthwise, precalculated_binary_depthwise, binary, precalculated_binary, ternary_depthwise, precalculated_ternary_depthwise, ternary, precalculated_ternary
from tensorboard.plugins.hparams import api as hp
import sys


def _prepare_data():
    records = AtrialFibrillation.get_records()
    samples = []
    labels = []
    forty_two_seconds = 250*42
    window_size = forty_two_seconds
    for record in records:
        _samples, _labels = record.as_uniformly_sized_examples(
                         downsampling_factor=2,
                         example_window_size=window_size,
                         offset_factor=0
                  )
        samples.extend(_samples)

        def to_categorical(label):
            if label == "(N":
                return 0
            else:
                return 1

        _labels = map(to_categorical, _labels)
        labels.extend(_labels)
    labels = np.array(labels)
    samples = np.array(samples)
    samples = samples[:, :, 0:1]
    print(np.shape(samples))
    mean = np.mean(samples)
    std = np.std(samples)
    samples = (samples - mean)/std
    print(np.shape(samples))
    arrhythmia_percentage = np.sum(labels)/np.shape(labels)[0]
    print("arrhythmia percentage", arrhythmia_percentage * 100)
    x_train, x_valid, y_train, y_valid = train_test_split(samples, labels,
                                                          test_size=0.4)
    x_test, x_valid, y_test, y_valid = train_test_split(x_valid, y_valid,
                                                        test_size=0.5)
    return x_train, x_test, x_valid, y_train, y_test, y_valid,


def run(metadata):
    x_train, x_test, x_valid, y_train, y_test, y_valid = _prepare_data()
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = "runs/{}".format(timestamp)
    build_model = metadata['architecture']
    model = build_model(input_shape=np.shape(x_train[0]), metadata=metadata)
    model.fit(
        x=x_train,
        y=y_train,
        epochs=metadata['epochs'],
        batch_size=512,
        validation_data=(x_valid, y_valid),
        callbacks=[
            tf.keras.callbacks.TensorBoard(log_dir=log_dir),
            tf.keras.callbacks.ModelCheckpoint(
                filepath='{}.hdf5'.format(timestamp),
                monitor='val_loss',
                save_best_only=True,
            )],
        verbose=2
    )
    model = tf.keras.models.load_model('{}.hdf5'.format(timestamp))
    _, accuracy = model.evaluate(x=x_test, y=y_test)
    metadata.update({'test_accuracy': accuracy, 'architecture': model.name})
    with tf.summary.create_file_writer(log_dir).as_default():
        hp.hparams(metadata)
        tf.summary.scalar('test_accuracy', accuracy, step=1)
    tf.keras.backend.clear_session()


def loops():
    epochs = 400
    for architecture in [
        binary_depthwise,
        precalculated_binary_depthwise,
        ternary,
        precalculated_ternary,
        ternary_depthwise,
        precalculated_ternary_depthwise,
        binary,
        precalculated_binary,
    ]:
        for learning_rate in [x*1e-5 for x in range(400, 466, 6)]:
            hparams = {
                "learning_rate": learning_rate,
                "epochs": epochs,
                "architecture": architecture,
            }
            run(hparams)

def main(args):
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--model", help="select the model to train",
                            choices=["precalculation",
                                     "traditional_bnn",
                                     "traditional_bnn_without_depthwise"], required=True)
    arg_parser.add_argument("--epochs", help="number of epochs to train", type=int, required=True)
    arg_parser.add_argument("--learning_rate", type=float, required=True)
    args = arg_parser.parse_args(args)
    run(args)

if __name__ == "__main__":
    # tf.config.threading.set_inter_op_parallelism_threads(16)
    # tf.config.threading.set_intra_op_parallelism_threads(8)
    for _ in range(0, 3):
        loops()
