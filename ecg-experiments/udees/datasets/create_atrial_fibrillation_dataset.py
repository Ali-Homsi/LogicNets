from udees.datasets.mitdb import AtrialFibrillation
import pandas as pd
from pathlib import Path
import pickle

records = AtrialFibrillation.get_records()

directory = Path("atrial_fibrillation")
two_minutes = 250*60*2
fortyfive_seconds = 250*45
fortytwo_seconds = 250*42
window_size = fortytwo_seconds
sinus_rhythm_directory = directory.joinpath("sinus_rhythm")
atrial_fibrillation_directory = directory.joinpath("atrial_fibriallation")

sinus_rhythm_directory.mkdir(exist_ok=True)
atrial_fibrillation_directory.mkdir(exist_ok=True)

for record in records:
    examples = record.as_uniformly_sized_examples(downsampling_factor=2,
                                                  example_window_size=window_size,
                                                  offset_factor=0)
    for (example_number_in_record,
         example) in enumerate(zip(*examples)):
        # example is a 2-tuple, example[0] is a nparray of features example[1] is the label
        # print(example[0])
        # print(example.type) #AttributeError: 'tuple' object has no attribute 'type'

        if example[1] == "(N":
            directory = sinus_rhythm_directory
        else: #example[1]="(AFIB"
            directory = atrial_fibrillation_directory
        example_file = directory.joinpath("{}_{}.pickle".format(record.id,
                                           example_number_in_record))
        with open(example_file, 'wb') as f:
            pickle.dump(example, f)
