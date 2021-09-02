import numpy as np
from pandas import DataFrame
# from tensorflow.keras.preprocessing.sequence import pad_sequences
import csv
import os

import six #ALI : added to remove tensorflow dependency

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
    # y = np.empty(length) # original code
    y = np.empty((length, 2)) #y is 2 dimensional, to make y ouptut shape as a one_hot vector
    for i, f in enumerate(file_list):
        data = DataFrame(ECGDataIterator(f, subsample)).to_numpy().T
        data = pad_sequences(data, maxlen=num_samples, padding='pre', truncating='pre')
        x[i, :] = data.T
        # y[i] = 0 if 'sinus' in f else 1 #original code

        #ALI:  SINUS_RYTHM one_hot vector with argmax = 0 [1,0] return np.eye(2)[1]  # AF one_hot vector with argmax = 1 [0,1]
        y[i] = np.eye(2)[0] if 'sinus' in f else np.eye(2)[1]
    return x, y


def extract_ecg_file_list(datafolder, csvfile):
    file_list = list()
    with open(os.path.join(datafolder, csvfile)) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            file_list.append(os.path.join(datafolder, row[0]))
    return file_list


# copied from tensorflow source code, to remove tensorflow dependency
def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    num_samples = len(sequences)

    lengths = []
    sample_shape = ()
    flag = True

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.

    for x in sequences:
        try:
            lengths.append(len(x))
            if flag and len(x):
                sample_shape = np.asarray(x).shape[1:]
                flag = False
        except TypeError:
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))

    if maxlen is None:
        maxlen = np.max(lengths)

    is_dtype_str = np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.unicode_)
    if isinstance(value, six.string_types) and dtype != object and not is_dtype_str:
        raise ValueError("`dtype` {} is not compatible with `value`'s type: {}\n"
                         "You should set `dtype=object` for variable length strings."
                         .format(dtype, type(value)))

    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" '
                             'not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s '
                             'is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x
