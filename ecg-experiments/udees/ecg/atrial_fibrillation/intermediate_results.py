import tensorflow as tf
from tensorflow.keras.models import load_model
import sys
import qnn.layers
import qnn.constraints
from udees.ecg.atrial_fibrillation.dataloader import extract_data
import numpy as np
from pathlib import Path

model = load_model(sys.argv[1])
sample_path = sys.argv[2]
sample = extract_data([sample_path])[0]
sample = np.delete(sample, 1, axis=2)

intermediate_results = {}

def get_subgraph(model, subslice):
    def subgraph(x):
        for layer in model.layers[subslice]:
            x = layer(x)
        return x
    return subgraph

for final_layer_index in range(1, len(model.layers)):
    intermediate_results[final_layer_index] = get_subgraph(model, slice(0, final_layer_index))(sample)

for key, value in intermediate_results.items():
    filename = "intermediate_results/{}_layer_{}.txt".format(Path(sample_path).stem, key-1)
    value = value.numpy()[0]
    np.savetxt(filename, value)
