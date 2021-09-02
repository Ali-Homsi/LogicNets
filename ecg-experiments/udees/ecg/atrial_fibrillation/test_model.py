import csv
import sys
import os
from os.path import join, split, splitext, basename
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
import numpy as np
from argparse import ArgumentParser
from models import precalculation
from models import traditional_bnn
from models import traditional_bnn_without_depthwise
from dataloader import extract_ecg_file_list, extract_data


def make_report(y_actual, y_pred, thresh):
    tn, fp, fn, tp = confusion_matrix(y_actual, np.greater_equal(y_pred, thresh)).ravel()

    accuracy = (tp + tn) / (tn + fp + fn + tp)
    sensitivity = tp / (tp + fn)
    fallout = fp / (tn + fp)

    print('- accuracy: %.4f' % accuracy)
    print('- sensitivity: %.4f' % sensitivity)
    print('- fallout: %.4f' % fallout)
    print(' ')

    return accuracy, sensitivity, fallout


def main(args):
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--modelpath", help="specify the model to test", required=True)
    arg_parser.add_argument("--datapath", help="absolute path to the folder which contains the csv files with ECG data", required=True)
    arg_parser.add_argument("--savepath", help="absolute path to the folder the test results shall be written to", required=True)
    args = arg_parser.parse_args(args)

    # Load data set
    file_list = extract_ecg_file_list(args.datapath, "test.csv")
    x_test, y_test = extract_data(file_list)
    x_test_chI = np.delete(x_test, 1, axis=2)

    # Load model
    saved_model = load_model(args.modelpath)
    saved_model.summary()

    # Infer predictions
    predictions = saved_model.predict(x_test_chI, verbose=1)
    accuracy, sensitivity, fallout = make_report(y_test, predictions, thresh=0.5)

    # Build results file path
    results_folder = join(args.savepath, saved_model.name)
    _, model_file = split(args.modelpath)
    model_name, _ = splitext(model_file)
    results_file_name =  model_name + "_testresults.csv"
    results_path = join(results_folder, results_file_name)

    # Write results
    with open(results_path, 'wt') as results_file:
        wtr = csv.writer(results_file, delimiter=' ', lineterminator='\n')
        wtr.writerow(["ecg-path", "ground-truth", "model-prediction"])

        def get_last_directory_and_basename(path):
            dir_name = basename(split(path)[0])
            return join(dir_name, basename(path))

        results = list(
            zip(
                map(get_last_directory_and_basename, file_list),
                (int(y) for y in y_test),
                (int(p > 0.5) for [p] in predictions)
            )
        )
        wtr.writerows(results)


if __name__ == "__main__":
    main(sys.argv[1:])
