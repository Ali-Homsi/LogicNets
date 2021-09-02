import numpy as np
from pandas import DataFrame
from tensorflow.keras.preprocessing.sequence import pad_sequences
import csv
import os

class ECGDataIterator:

    def __init__(self, f, subsample=1):
        self.ifd = open(f, "rb")
        self._ss = subsample
        self._offset = 2048

    def __next__(self):
        chIb = self.ifd.read(2)
        if chIb == b'':
            self.ifd.close()
            raise StopIteration
        chI = int.from_bytes(chIb, byteorder="little") - self._offset
        chIII = int.from_bytes(self.ifd.read(2), byteorder="little") - self._offset
        # move pointer 4 bytes times subsampling to next data
        if self._ss > 1:
            self.ifd.seek(4*(self._ss-1), 1)

        return chI, chIII

    def __iter__(self):
        return self


def extract_data(file_list, sample_rate=125, subsample=1, duration=42):
    fs = sample_rate // subsample
    num_samples = duration * fs
    length = len(file_list)
    x = np.empty((length, num_samples, 2))
    y = np.empty(length)
    for i, f in enumerate(file_list):
        data = DataFrame(ECGDataIterator(f, subsample)).to_numpy().T
        data = pad_sequences(data, maxlen=num_samples, padding='pre', truncating='pre')
        x[i, :] = data.T
        y[i] = 0 if 'sinus' in f else 1
    return x, y


def extract_ecg_file_list(datafolder, csvfile):
    file_list = list()
    with open(os.path.join(datafolder, csvfile)) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            file_list.append(os.path.join(datafolder, row[0]))
    return file_list
