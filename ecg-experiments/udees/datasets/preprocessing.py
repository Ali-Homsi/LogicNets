from udees.datasets.mitdb import AtrialFibrillation
from sklearn.model_selection import train_test_split
import numpy as np

def save_training_data():
    records = AtrialFibrillation.get_records()
    samples = []
    labels = []
    features_file = "features.npy"
    labels_file = "labels.npy"
    forty_two_seconds = 250*42
    ten_seconds = 250*10

    # window_size = forty_two_seconds
    window_size = ten_seconds

    #len(records) = 16, each record has a number of samples which differ from record to record.
    #FOR WINDOW SIZE 10 SEC : record[0 and 1 and 2] have 3675 samples, record[3] has 3676 samples
    for record in records:
        _sample, _label = record.as_uniformly_sized_examples(
                         downsampling_factor=2,
                         example_window_size=window_size,
                         offset_factor=0
                  )
        samples.extend(_sample)


        def to_categorical(label):
            if label == "(N":
                return np.eye(2)[0]  # SINUS_RYTHM one_hot vector with argmax = 0 [1,0]

            else:
                return np.eye(2)[1]  # AF one_hot vector with argmax = 1 [0,1]

        _label = map(to_categorical, _label)
        labels.extend(_label)



    labels = np.array(labels)
    # print("labels shape",np.shape(labels))
    # print("labels length", len(labels))


    samples = np.array(samples)
    # print(len(samples)) # 58731
    # print(samples[0])
    # print(samples.shape) #windowsize 42 sec: (13848, 5250, 2)/// windowsize 10 sec: (58731, 1250, 2)


    samples = samples[:, :, 0:1]
    # print(samples[0])
    # print(samples.shape) #windowsize 42 sec: (13848, 5250, 1)/// windowsize 10 sec: (58731, 1250, 1)

    # print(samples)
    # print(samples.shape)

    #reshape the array to 1 channel input with 1250 features : (58731, 1250, 1) -> (58731, 1, 1250)
    samples = samples.reshape((len(samples), 1, 1250))

    # print(samples[0])
    # print(samples.shape)

    # print("labels shape", np.shape(labels)) # (58731, 2)
    # print("labels length", len(labels)) # 58731
    # print(labels[0]) # [1. 0.]

    print(samples[0])
    print(labels[0])

    print(samples[1])
    print(labels[1])
    input()

    mean = np.mean(samples)
    std = np.std(samples)
    samples = (samples - mean) / std
    # print("features shape after standarizing", np.shape(features))
    print("arrhythmia percentage", calc_arrhythmia_percentage(labels))

    with open(features_file, 'wb') as f:
        np.save(f, samples)

    with open(labels_file, 'wb') as f:
        np.save(f, labels)

    # np.save(features_file, samples , allow_pickle=False)
    # np.save(labels_file, labels,  allow_pickle=False)

    # x_train, x_valid, y_train, y_valid = train_test_split(features, labels,
    #                                                       test_size=0.4)
    # x_test, x_valid, y_test, y_valid = train_test_split(x_valid, y_valid,
    #                                                     test_size=0.5)
    # return x_train, x_test, x_valid, y_train, y_test, y_valid

def calc_arrhythmia_percentage(labels):
    sinus_count = 0
    af_count = 0
    for label in labels:
        if (label == np.eye(2)[0]).all():
            sinus_count += 1
        else: af_count += 1
    return af_count*100/np.shape(labels)[0]

def load_training_data():
    # features = np.load("features.npy", allow_pickle=False)
    # labels = np.load("labels.npy", allow_pickle=False)

    with open('features.npy', 'rb') as f:
        samples = np.load(f)

    with open('labels.npy', 'rb') as f:
        labels = np.load(f)

    # print("labels shape", np.shape(labels))
    # print("labels length", len(labels))
    # print("features shape", np.shape(features))
    # print("features length", len(features))

    print(samples[0])
    print(labels[0])

    print(samples[1])
    print(labels[1])



if __name__ == "__main__":
    save_training_data()
    load_training_data()



# x_train, x_test, x_valid, y_train, y_test, y_valid = save_training_data()
# print("xtrain shape=",np.shape(x_train[0]))
# print("xtrain = ",x_train[0])
# print("ytrain shape =",np.shape(y_train[0]))
# print("ytrain = ",y_train[0])
# print("len xtrain = ",len(x_train))
# print("len ytrain = ",len(y_train))
# print("len xtest = ",len(x_test))
# print("len ytest = ",len(y_test))
# print("len xval = ",len(x_valid))
# print("len ytrain = ",len(y_valid))
#
# input()
