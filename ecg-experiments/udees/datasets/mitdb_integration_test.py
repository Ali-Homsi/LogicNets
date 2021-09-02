from udees.datasets.mitdb import AtrialFibrillation
from udees.datasets.mitdb import simplify_mitdb_sample
import unittest
from unittest import TestCase


class SimplifiedSamples(TestCase):
    def test_has_correct_number_of_bytes(self):
        record = AtrialFibrillation.get_records()[0]
        window_size = 250 * 42
        downsampling_factor = 2
        samples, _ = record.as_uniformly_sized_examples(
                                downsampling_factor=downsampling_factor,
                                example_window_size=window_size,
                                offset_factor=0
                            )
        sample = samples[0]
        sample = simplify_mitdb_sample(sample)
        number_of_channels = 2
        expected = window_size / downsampling_factor * number_of_channels
        expected = int(expected)
        actual = len(sample)
        self.assertEqual(expected, actual)




if __name__ == "__main__":
    unittest.main()
