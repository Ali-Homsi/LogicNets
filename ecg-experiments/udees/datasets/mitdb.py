import wfdb
import numpy
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Sequence


def interleave_record(sample):
    acc = []
    for point in sample:
        for channel in point:
            acc.append(channel)
    return acc


class Record:

    def __init__(self, id=None, db=None):
        self.id = id
        self._db = db

    def __eq__(self, other):
        def compare_field(field):
            return self.__dict__[field] == other.__dict__[field]

        if isinstance(other, Record):
            return all(map(compare_field, [
                          'id',
                          'db',
                       ]))
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def as_frame(self):
        samples = self._db.construct_sample_frame_from_record_id(self.id)
        labels = self._db.construct_annotations_frame_from_record_id(self.id)
        samples = samples.join(labels, on=["sample"])
        return samples

    @property
    def samples(self):
        """
        Returns a numpy array containing the records samples.
        Typically the shape of that array is (num_samples, num_channels).
        """
        return self._db.read_samples_for_record_id(self.id)

    def as_example(self, resampling_factor=2):
        """
        Returns labeled sections resampled by `resampling_factor`
        and their corresponding labels as a tuple.
        The tuple has the form (sections, labels)
        where `len(sections) == len(labels)` but we do not necessarily
        have `len(sections[i]) == len(sections[j])` with `i != j`.
        To retrieve sections of equal size use `as_uniformly_sized_examples`.
        """
        pass

    def as_uniformly_sized_examples(self,
                                    downsampling_factor=2,
                                    example_window_size=16000,
                                    offset_factor=0):
        """
        Similar to `as_example` but slices the examples so they all have equal
        size.
        Args:
            downsampling_factor: specifies how to downsample. E.g. specifying 2
                                 will half the sample rate
            example_window_size: the number of samples that are extracted
                                 per example, the number refers to the
                                 window with the original sampling rate,
                                 ie. before downsampling
            offset_factor: the factor by which to offset the position of the
                           first
                           example_window in relation to the position of the
                           annotation. E.g. using -0.5 will ensure that the
                           annotation position is in the middle of the first
                           example_window, while the default of 0 will result
                           in the window starting at the position of the
                           annotation.
        """
        samples = self.samples
        labels, labeled_samples = self.labels
        data = ([], [])

        def labeled_sections():
            return zip(labeled_samples,
                       np.concatenate([labeled_samples[1:],
                                       [len(samples)]]))

        for label, (section_start, section_end) in zip(labels,
                                                       labeled_sections()):
            section_size = section_end - section_start
            number_of_windows = section_size//example_window_size

            def offset(index):
                return section_start + example_window_size * index

            def _resample(window):
                return window[::downsampling_factor]

            for current_window_index in range(number_of_windows):
                window_slice = slice(offset(current_window_index),
                                     offset(current_window_index+1))
                window = _resample(samples[window_slice])

                data[0].append(window)
                data[1].append(label)

        return data

    @property
    def labels(self):
        """
        Returns the label annotations in form of a tuple.
        Where the first item is the annotated symbols and the second
        item lists the sample that the symbol was annotated to.
        """
        annotations = self._db.read_annotations_for_record_id(self.id)
        return annotations.aux_note, annotations.sample.astype('int64')


class MITBIH:
    """ Base class for downloading and simplified access to physionet ecg data.
    """

    @classmethod
    def download(cls):
        wfdb.dl_database(cls._db_id_on_physionet, dl_dir=cls._db_name)

    @classmethod
    def read_annotations_for_record_id(cls, record_id):
        annotations = wfdb.rdann(record_name='{}/{}'.format(cls._db_name,
                                                            record_id),
                                 return_label_elements=['symbol'],
                                 extension='atr')
        return annotations

    @classmethod
    def get_record_ids(cls):
        db_folder = cls._db_name
        assert Path(db_folder).exists(), ("you have to download the data"
               "set {} first".format(cls._db_name))
        record_ids = wfdb.get_record_list(db_dir=cls._db_id_on_physionet)
        return record_ids

    @classmethod
    def get_records(cls):
        record_ids = cls.get_record_ids()
        return [Record(id, cls) for id in record_ids]

    @classmethod
    def save_from_raw_to_csv(cls):
        frame = cls.get_records().as_frame()
        frame.to_csv("{}.csv".format(cls._db_name))

    @classmethod
    def from_csv_as_frame(cls):
        frame = pd.read_csv("{}.csv".format(cls._db_name))
        return frame

    @classmethod
    def read_samples_for_record_id(cls, record_id, return_res=16):
        record = wfdb.rdrecord('{}/{}'.format(cls._db_name,
                                              record_id),
                               physical=False,
                               return_res=return_res)
        samples = record.d_signal
        return samples

    @classmethod
    def construct_annotations_frame_from_record_id(cls, record_id):
        annotations = cls.read_annotations_for_record_id(record_id)
        index = pd.Index(
                  annotations.sample.astype('int64'),
                  names="sample"
                )
        annotations = pd.DataFrame(annotations.symbol,
                                   columns=["labels"],
                                   index=index)
        return annotations

    @classmethod
    def construct_sample_frame_from_record_id(cls, record_id):
        data = cls.read_samples_for_record_id(record_id)
        shape = np.shape(data)
        number_of_channels = shape[1]
        number_of_samples = shape[0]
        columns = ["Channel{}".format(i) for i in range(number_of_channels)]
        index = pd.Index(
                  list(range(number_of_samples)),
                  name="sample"
                )
        samples = pd.DataFrame(
                     data,
                     columns=columns,
                     index=index
                  )
        return samples

    @classmethod
    def _get_samples_sequence(cls) -> Sequence:
        record_ids = cls.get_record_ids()
        yield from (cls.read_samples_for_record_id(id) for id in record_ids)

    @classmethod
    def _get_annotations_sequence(cls) -> Sequence:
        record_ids = cls.get_record_ids()
        yield from (cls.read_annotations_for_record_id(id)
                    for id in record_ids)

    @classmethod
    def _get_numpy_samples_from_raw_data(cls) -> np.ndarray:
        samples = np.stack(cls._get_samples_sequence())
        return samples

    @classmethod
    def _get_numpy_annotations_from_raw_data(cls) -> np.ndarray:
        annotations = cls._get_annotations_sequence()
        return np.stack(([annotation.aux_note, annotation.sample]
                         for annotation in annotations))


class Arrhythmia(MITBIH):
    """https://www.physionet.org/content/mitdb/1.0.0/
    """
    _db_id_on_physionet = 'mitdb'
    _db_name = 'arrhythmia'


class NormalSinusRhythm(MITBIH):
    """https://www.physionet.org/content/nsrdb/1.0.0/
    """
    _db_id_on_physionet = 'nsrdb'
    _db_name = 'normal_sinus_rhythm'


class AtrialFibrillation(MITBIH):
    """https://www.physionet.org/content/afdb/1.0.0/
    """
    _db_id_on_physionet = 'afdb'
    _db_name = 'atrial_fibrillation'

    _notes_on_records = {
        '00735': "Signals unavailable",
        '03665': "Signals unavailable",
        '04043': "Block 39 is unreadable",
        '05091': "Corrected QRS annotations are available (annotator qrs)",
        '06453': "Recording ends after 9 hours, 15 minutes",
        '08378': "No start time",
        '08405': "No start time; block 1067 is unreadable",
        '08434': "Blocks 648, 857, and 894 are unreadable",
        '08455': "No start time",
    }

    @classmethod
    def get_record_ids(cls):
        """Override super classes method to filter out inconsistent
           records.
        """
        record_files = super().get_record_ids()
        record_files = list((record for record in record_files
                             if record not in cls._notes_on_records))
        return record_files
