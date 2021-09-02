import udees.datasets.mitdb as mitdb
from udees.datasets.mitdb import Record
from udees.datasets.mitdb import interleave_record
import unittest
from unittest.mock import Mock
from unittest.mock import patch


@patch("udees.datasets.mitdb.np")
@patch("udees.datasets.mitdb.pd")
@patch("udees.datasets.mitdb.Path")
@patch("udees.datasets.mitdb.wfdb")
@unittest.skip("redesigned records, tests need to be fixed")
class MITDB_Test(unittest.TestCase):

    def rdann_mock(symbols, samples):
        annotations = Mock()
        annotations.symbols.return_value = symbols
        annotations.samples.return_value = samples
        return annotations

    def test_download_arrhythmia_calls_wfdb(self, wfdb, *args):
        mitdb.Arrhythmia.download()
        wfdb.dl_database.assert_called_once_with('mitdb', dl_dir='arrhythmia')

    def test_get_record_list_called_with_correct_parameter(self, wfdb, *args):
        mitdb.Arrhythmia.get_record_data()
        wfdb.get_record_list.assert_called_once_with(db_dir='mitdb')

    def test_get_empty_record_list(self, wfdb, *args):
        wfdb.get_record_list.return_value = []
        wfdb.rdsamp.return_value = (None, None)
        wfdb.rdann.return_value = None
        data = mitdb.Arrhythmia.get_record_data()
        self.assertSequenceEqual([], data, msg="data={}".format(data))

    def test_get_one_record(self, wfdb, Path, pd, np):
        wfdb.get_record_list.return_value = ['record_file_path']
        wfdb.rdsamp.return_value = ("some data", 'field descriptors')
        data = mitdb.Arrhythmia.get_record_data()[0]
        expected = mitdb.Record(
                           data='some data',
                           file_name='record_file_path',
                           labels=np.stack()
                         )
        self.assertEqual(expected, data,
                         msg="expected: {}, actual: {}"
                             .format(expected, data))

    def test_raise_assert_exception(self, wfdb, Path, pd, np):
        path = unittest.mock.Mock()
        path.exists.return_value = False
        Path.return_value = path
        with self.assertRaises(AssertionError) as error:
            mitdb.Arrhythmia.get_record_data()
        self.assertEqual(
            "you have to download the dataset arrhythmia first",
            str(error.exception)
        )

    def test_rdann_called_correctly_once(self, wfdb, Path, pd, np):
        wfdb.get_record_list.return_value = ['file_name']
        wfdb.rdsamp.return_value = ('', '')
        mitdb.Arrhythmia.get_record_data()
        wfdb.rdann.assert_called_once_with(
            record_name='{}/{}'.format('arrhythmia', 'file_name'),
            extension='atr'
        )

    def test_rdann_called_correctly_twice(self, wfdb, Path, pd, np):
        wfdb.get_record_list.return_value = ['first', 'second']
        wfdb.rdsamp.return_value = ('', '')
        mitdb.Arrhythmia.get_record_data()
        for name in ['first', 'second']:
            wfdb.rdann.assert_any_call(record_name='{}/{}'
                                       .format('arrhythmia', name),
                                       extension='atr')

    def test_np_stack_called_correctly(self, wfdb, Path, pd, np):
        wfdb.get_record_list.return_value = ['name']
        wfdb.rdsamp.return_value = ('', '')
        mitdb.Arrhythmia.get_record_data()
        annotations = wfdb.rdann()
        np.stack.assert_called_once_with([annotations.symbol,
                                         annotations.sample])

    def test_labels_read_correctly(self, wfdb, Path, pd, np):
        wfdb.get_record_list.return_value = ['first']
        wfdb.rdsamp.return_value = ('', '')

        def np_stack(items):
            return list(zip(items[0], items[1]))

        np.stack.side_effect = np_stack
        for label in ['label1', 'secondlabel']:
            with self.subTest():
                annotations = Mock()
                annotations.symbol = [label]
                annotations.sample = [5]
                wfdb.rdann.return_value = annotations
                expected = np_stack([annotations.symbol, annotations.sample])
                data = mitdb.Arrhythmia.get_record_data()[0]
                self.assertSequenceEqual(expected, data.labels)

    def test_read_samples(self, wfdb, Path, pd, np):
        wfdb.get_record_list.return_value = ['record_file_path']
        wfdb.rdsamp.return_value = ('some data', 'field descriptors')
        data = mitdb.Arrhythmia.get_record_data()[0]
        expected = Record(
                          data='some data',
                          file_name='record_file_path',
                          labels=np.stack()
                          )
        self.assertEqual(expected,
                         data,
                         msg="expected: {}, actual: {}"
                             .format(expected, data))


@unittest.skip("redesigned records, tests need to be fixed")
class Record_Test(unittest.TestCase):
    def test_inequality_for_different_data(self):
        r1 = Record()
        r2 = Record(data=1)
        self.assertFalse(r1 == r2)

    def test_equality_of_non_empty_records(self):
        self.assertEqual(Record(data=1, file_name="file", labels=["label"]),
                         Record(data=1, file_name="file", labels=["label"]))

    def test_equality_of_empty_records(self):
        self.assertEqual(Record(), Record())

    def test_inequality_for_different_file_name(self):
        r1 = Record(data=1, file_name="file", labels=["label"])
        r2 = Record(data=1, file_name="other", labels=["label"])
        self.assertFalse(r1 == r2)

    def test_inequality_for_different_labels(self):
        r1 = Record(data=1, file_name="file", labels=["labels"])
        r2 = Record(data=1, file_name="file", labels=["labels", "more labels"])
        self.assertFalse(r1 == r2)


class SimplifyNumpyExamplesFromMITDB(unittest.TestCase):
    def create_dummy_sample(self, length):
        sample = [[x, y] for x, y in zip(range(length), reversed(range(length)))]
        return sample

    def test_content_one_element(self):
        sample = self.create_dummy_sample(1)
        self.assertSequenceEqual([0, 0], interleave_record(sample))

    def test_content_two_elements(self):
        sample = self.create_dummy_sample(2)
        self.assertSequenceEqual([0, 1, 1, 0],
                                 interleave_record(sample))

    def test_content_three_elements(self):
        sample = self.create_dummy_sample(3)
        self.assertSequenceEqual(
                 [0, 2, 1, 1, 2, 0],
                 interleave_record(sample)
             )


if __name__ == "__main__":
    unittest.main()
