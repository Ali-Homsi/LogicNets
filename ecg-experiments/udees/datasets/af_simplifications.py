from udees.datasets.mitdb import AtrialFibrillation
from udees.datasets.mitdb import interleave_record
from sklearn.model_selection import train_test_split
from types import SimpleNamespace
from pathlib import Path
import os

"""
This module contains code to produce simplified ecg records from
the physionet's afdb dataset. Each record is split into 42second
subrecords, converted to uint16 little-endian with interleaved channels.
Each subrecord is then saved to a folder corresponding to it's label.
"""

OUT_FOLDER = Path("simplified_atrial_fibrillation")


def split_uint16(number):
    upper_byte = int(number/2**8)
    lower_byte = int(number - upper_byte*2**8)
    split = SimpleNamespace(upper_byte=upper_byte, lower_byte=lower_byte)
    return split


def apply_12bit_ecg_offset(number):
    return number + 2**11


def save_simplifications():

    def save_sample(sample, label, id):
        sample = interleave_record(sample)
        if label == '(N':
            subdir = "sinus"
        else:
            subdir = "atrial_fibrillation"
        out_folder = OUT_FOLDER.joinpath(subdir)
        os.makedirs(out_folder, exist_ok=True)
        out_folder = out_folder.joinpath("{}.ecg".format(id))
        with open(out_folder, 'wb') as file:
            def save_as_little_endian(number):
                split = split_uint16(number)
                file.write(bytes([split.lower_byte, split.upper_byte]))

            for point in sample:
                point = apply_12bit_ecg_offset(point)
                save_as_little_endian(point)

    AtrialFibrillation.download() #ALI: need this only for the first time run
    for record in AtrialFibrillation.get_records():
        print("processing record:", record.id)
        samples, labels = record.as_uniformly_sized_examples(
                                   downsampling_factor=2,
                                   example_window_size=250*42,
                                   offset_factor=0
                                 )
        for sub_index, (sample, label) in enumerate(zip(samples, labels)):
            save_sample(sample, label, "{}_{}".format(record.id, sub_index))


def get_train_test_val(out_folder=OUT_FOLDER):
    def get_files(subfolder):
        files = [f for f in OUT_FOLDER.joinpath(subfolder).glob("*.ecg")]
        return files

    sinus_files = get_files("sinus")
    sinus_labels = ["sinus"] * len(sinus_files)
    af_files = get_files("atrial_fibrillation")
    af_labels = ["atrial_fibrillation"] * len(af_files)
    files = sinus_files + af_files
    labels = sinus_labels + af_labels
    x_train, x_test, y_train, y_test = train_test_split(files, labels,
                                                        test_size=0.33,
                                                        random_state=42)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test,
                                                    test_size=0.5,
                                                    random_state=42)
    return x_train, x_test, x_val, y_train, y_test, y_val

def save_train_test_val():
    x_train, x_test, x_val, y_train, y_test, y_val = get_train_test_val()

    def save_filenames(csv_file, xs, ys):
        with open(csv_file, "w") as f:
            for x, y in zip(xs, ys):
                f.write("{}, {}\n".format(x, y))

    save_filenames("train.csv", x_train, y_train)
    save_filenames("test.csv", x_test, y_test)
    save_filenames("validation.csv", x_val, y_val)


if __name__ == "__main__":
    # save_simplifications() #ALI: run this first
    save_train_test_val()

    # x_train, x_test, x_val, y_train, y_test, y_val = get_train_test_val()
    # print(x_train[0])
