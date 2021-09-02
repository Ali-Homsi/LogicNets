import tensorflow as tf
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPool1D
from qnn.layers import Binarize
from qnn.layers import Conv1D
from qnn.layers import Dense
from qnn.layers import Ternarize
from qnn.layers import QuantizeTwoBit
padding='valid',
from qnn.layers import DepthwiseConv1D
from qnn.constraints import ChebyshevDistance
from typing import Tuple, Dict

padding='valid',

def precalculated_ternary_depthwise(input_shape, metadata) -> Tuple[tf.keras.Model, Dict]:
    model = Sequential([
        Input(shape=input_shape),

        Conv1D(
            filters=6,
            kernel_size=10,
            strides=3,
            padding='valid',
            use_bias=False,
            quantizer=Binarize(),
            kernel_constraint=ChebyshevDistance(1.)
        ),
        BatchNormalization(),
        Ternarize(),

        MaxPool1D(
            pool_size=8,
            strides=6,
        ),

        DepthwiseConv1D(
            kernel_size=6,
            use_bias=False,
            padding='valid',
        ),
        BatchNormalization(),
        Ternarize(),

        Conv1D(
            kernel_size=1,
            filters=5,
            use_bias=False,
            padding='valid',
        ),
        BatchNormalization(),
        Ternarize(),

        DepthwiseConv1D(
            kernel_size=6,
            use_bias=False,
            padding='valid',
        ),
        BatchNormalization(),
        Ternarize(),

        Conv1D(
            kernel_size=1,
            filters=5,
            use_bias=False,
            padding='valid',
        ),
        BatchNormalization(),
        Ternarize(),

        MaxPool1D(
            pool_size=3,
            strides=2,
        ),

        DepthwiseConv1D(
            kernel_size=6,
            use_bias=False,
            padding='valid',
        ),
        BatchNormalization(),
        Ternarize(),

        Conv1D(
            kernel_size=1,
            filters=3,
            use_bias=False,
            padding='valid',
        ),
        BatchNormalization(),
        QuantizeTwoBit(),

        MaxPool1D(
            pool_size=4,
            strides=4,
        ),
        MaxPool1D(
            pool_size=3,
            strides=3,
        ),

        Flatten(),
        Dense(
            units=1,
            use_bias=False,
            quantizer=Binarize(),
            kernel_constraint=ChebyshevDistance(1.)
        ),
        BatchNormalization(),
        Activation('sigmoid')
    ],
        name="precalculated_ternary_depthwise"
    )
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            'accuracy',
        ],
        optimizer=tf.keras.optimizers.Adamax(metadata['learning_rate']),
    )

    model.summary()
    return model

def ternary_depthwise(input_shape, metadata) -> Tuple[tf.keras.Model, Dict]:
    model = Sequential([
        Input(shape=input_shape),

        Conv1D(
            filters=6,
            kernel_size=10,
            strides=3,
            padding='valid',
            use_bias=False,
            quantizer=Binarize(),
            kernel_constraint=ChebyshevDistance(1.)
        ),
        BatchNormalization(),
        Ternarize(),

        MaxPool1D(
            pool_size=8,
            strides=6,
        ),

        DepthwiseConv1D(
            kernel_size=6,
            quantizer=Binarize(),
            kernel_constraint=ChebyshevDistance(1.),
            padding='valid',
            use_bias=False
        ),
        BatchNormalization(),
        Ternarize(),

        Conv1D(
            kernel_size=1,
            filters=5,
            quantizer=Binarize(),
            padding='valid',
            kernel_constraint=ChebyshevDistance(1.),
            use_bias=False
        ),
        BatchNormalization(),
        Ternarize(),

        DepthwiseConv1D(
            kernel_size=6,
            quantizer=Binarize(),
            kernel_constraint=ChebyshevDistance(1.),
            padding='valid',
            use_bias=False
        ),
        BatchNormalization(),
        Ternarize(),

        Conv1D(
            kernel_size=1,
            quantizer=Binarize(),
            kernel_constraint=ChebyshevDistance(1.),
            padding='valid',
            filters=5,
            use_bias=False
        ),
        BatchNormalization(),
        Ternarize(),

        MaxPool1D(
            pool_size=3,
            strides=2,
        ),

        DepthwiseConv1D(
            kernel_size=6,
            quantizer=Binarize(),
            kernel_constraint=ChebyshevDistance(1.),
            padding='valid',
            use_bias=False
        ),
        BatchNormalization(),
        Ternarize(),

        Conv1D(
            kernel_size=1,
            filters=3,
            quantizer=Binarize(),
            padding='valid',
            kernel_constraint=ChebyshevDistance(1.),
            use_bias=False
        ),
        BatchNormalization(),
        QuantizeTwoBit(),

        MaxPool1D(
            pool_size=4,
            strides=4,
        ),
        MaxPool1D(
            pool_size=3,
            strides=3,
        ),

        Flatten(),
        Dense(
            units=1,
            use_bias=False,
            quantizer=Binarize(),
            kernel_constraint=ChebyshevDistance(1.)
        ),
        BatchNormalization(),
        Activation('sigmoid')
    ],
        name="ternary_depthwise"
    )
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            'accuracy',
        ],
        optimizer=tf.keras.optimizers.Adamax(metadata['learning_rate']),
    )

    model.summary()
    return model


def precalculated_binary_depthwise(input_shape, metadata) -> Tuple[tf.keras.Model, Dict]:
    model = Sequential([
        Input(shape=input_shape),

        Conv1D(
            filters=6,
            kernel_size=10,
            strides=3,
            padding='valid',
            use_bias=False,
            quantizer=Binarize(),
            kernel_constraint=ChebyshevDistance(1.)
        ),
        BatchNormalization(),
        Binarize(),

        MaxPool1D(
            pool_size=8,
            strides=6,
        ),

        DepthwiseConv1D(
            kernel_size=6,
            use_bias=False,
        padding='valid',
        ),
        BatchNormalization(),
        Binarize(),

        Conv1D(
            kernel_size=1,
            filters=5,
            padding='valid',
            use_bias=False,
        ),
        BatchNormalization(),
        Binarize(),

        DepthwiseConv1D(
            kernel_size=6,
            use_bias=False,
            padding='valid',
        ),
        BatchNormalization(),
        Binarize(),

        Conv1D(
            kernel_size=1,
            filters=5,
            padding='valid',
            use_bias=False,
        ),
        BatchNormalization(),
        Binarize(),

        MaxPool1D(
            pool_size=3,
            strides=2,
        ),

        DepthwiseConv1D(
            kernel_size=6,
            use_bias=False,
            padding='valid',
        ),
        BatchNormalization(),
        Binarize(),

        Conv1D(
            kernel_size=1,
            filters=3,
            padding='valid',
            use_bias=False,
        ),
        BatchNormalization(),
        Binarize(),

        MaxPool1D(
            pool_size=4,
            strides=4,
        ),
        MaxPool1D(
            pool_size=3,
            strides=3,
        ),

        Flatten(),
        Dense(
            units=1,
            use_bias=False,
            quantizer=Binarize(),
            kernel_constraint=ChebyshevDistance(1.)
        ),
        BatchNormalization(),
        Activation('sigmoid')
    ],
        name="precalculated_binary_depthwise"
    )
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            'accuracy',
        ],
        optimizer=tf.keras.optimizers.Adamax(metadata['learning_rate']),
    )

    model.summary()
    return model


def binary_depthwise(input_shape, metadata) -> Tuple[tf.keras.Model, Dict]:
    model = Sequential([
        Input(shape=input_shape),

        Conv1D(
            filters=6,
            kernel_size=10,
            strides=3,
            padding='valid',
            use_bias=False,
            quantizer=Binarize(),
            kernel_constraint=ChebyshevDistance(1.)
        ),
        BatchNormalization(),
        Binarize(),

        MaxPool1D(
            pool_size=8,
            strides=6,
        ),

        DepthwiseConv1D(
            kernel_size=6,
            use_bias=False,
            quantizer=Binarize(),
            padding='valid',
            kernel_constraint=ChebyshevDistance(1.)
        ),
        BatchNormalization(),

        Binarize(),
        Conv1D(
            kernel_size=1,
            filters=5,
            use_bias=False,
            padding='valid',
            quantizer=Binarize(),
            kernel_constraint=ChebyshevDistance(1.)
        ),
        BatchNormalization(),
        Binarize(),

        DepthwiseConv1D(
            kernel_size=6,
            use_bias=False,
            quantizer=Binarize(),
            padding='valid',
            kernel_constraint=ChebyshevDistance(1.)
        ),
        BatchNormalization(),
        Binarize(),

        Conv1D(
            kernel_size=1,
            filters=5,
            use_bias=False,
            padding='valid',
            quantizer=Binarize(),
            kernel_constraint=ChebyshevDistance(1.)
        ),
        BatchNormalization(),
        Binarize(),

        MaxPool1D(
            pool_size=3,
            strides=2,
        ),

        DepthwiseConv1D(
            kernel_size=6,
            use_bias=False,
            quantizer=Binarize(),
            padding='valid',
            kernel_constraint=ChebyshevDistance(1.)
        ),
        BatchNormalization(),
        Binarize(),

        Conv1D(
            kernel_size=1,
            filters=3,
            use_bias=False,
            padding='valid',
            quantizer=Binarize(),
            kernel_constraint=ChebyshevDistance(1.)
        ),
        BatchNormalization(),
        QuantizeTwoBit(),

        MaxPool1D(
            pool_size=4,
            strides=4,
        ),
        MaxPool1D(
            pool_size=3,
            strides=3,
        ),

        Flatten(),
        Dense(
            units=1,
            use_bias=False,
            quantizer=Binarize(),
            kernel_constraint=ChebyshevDistance(1.)
        ),
        BatchNormalization(),
        Activation('sigmoid')
    ],
        name="binary_depthwise"
    )
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            'accuracy',
        ],
        optimizer=tf.keras.optimizers.Adamax(metadata['learning_rate']),
    )

    model.summary()
    return model


def binary(input_shape, metadata) -> Tuple[tf.keras.Model, Dict]:
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(
            filters=6,
            kernel_size=10,
            strides=3,
            padding='valid',
            use_bias=False,
            quantizer=Binarize(),
            kernel_constraint=ChebyshevDistance(1.)
        ),
        BatchNormalization(),
        Binarize(),
        MaxPool1D(
            pool_size=8,
            strides=6,
        ),
        Conv1D(
            kernel_size=6,
            use_bias=False,
            filters=5,
            padding='valid',
            quantizer=Binarize(),
            kernel_constraint=ChebyshevDistance(1.)
        ),
        BatchNormalization(),
        Binarize(),
        Conv1D(
            kernel_size=6,
            use_bias=False,
            filters=5,
            padding='valid',
            quantizer=Binarize(),
            kernel_constraint=ChebyshevDistance(1.)
        ),
        BatchNormalization(),
        Binarize(),
        MaxPool1D(
            pool_size=3,
            strides=2,
        ),
        Conv1D(
            kernel_size=6,
            use_bias=False,
            filters=3,
            padding='valid',
            quantizer=Binarize(),
            kernel_constraint=ChebyshevDistance(1.)
        ),
        BatchNormalization(),
        QuantizeTwoBit(),
        MaxPool1D(
            pool_size=4,
            strides=4,
        ),
        MaxPool1D(
            pool_size=3,
            strides=3,
        ),
        Flatten(),
        Dense(
            units=1,
            use_bias=False,
            quantizer=Binarize(),
            kernel_constraint=ChebyshevDistance(1.)
        ),
        BatchNormalization(),
        Activation('sigmoid')
    ],
        name="binary"
    )
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            'accuracy',
        ],
        optimizer=tf.keras.optimizers.Adamax(metadata['learning_rate']),
    )

    model.summary()
    return model


def ternary(input_shape, metadata) -> Tuple[tf.keras.Model, Dict]:
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(
            filters=6,
            kernel_size=10,
            strides=3,
            padding='valid',
            use_bias=False,
            quantizer=Binarize(),
            kernel_constraint=ChebyshevDistance(1.)
        ),
        BatchNormalization(),
        Ternarize(),
        MaxPool1D(
            pool_size=8,
            strides=6,
        ),
        Conv1D(
            kernel_size=6,
            use_bias=False,
            filters=5,
            padding='valid',
            quantizer=Binarize(),
            kernel_constraint=ChebyshevDistance(1.)
        ),
        BatchNormalization(),
        Ternarize(),
        Conv1D(
            kernel_size=6,
            use_bias=False,
            filters=5,
            padding='valid',
            quantizer=Binarize(),
            kernel_constraint=ChebyshevDistance(1.)
        ),
        BatchNormalization(),
        Ternarize(),
        MaxPool1D(
            pool_size=3,
            strides=2,
        ),
        Conv1D(
            kernel_size=6,
            use_bias=False,
            filters=3,
            padding='valid',
            quantizer=Binarize(),
            kernel_constraint=ChebyshevDistance(1.)
        ),
        BatchNormalization(),
        QuantizeTwoBit(),
        MaxPool1D(
            pool_size=4,
            strides=4,
        ),
        MaxPool1D(
            pool_size=3,
            strides=3,
        ),
        Flatten(),
        Dense(
            units=1,
            use_bias=False,
            quantizer=Binarize(),
            kernel_constraint=ChebyshevDistance(1.)
        ),
        BatchNormalization(),
        Activation('sigmoid')
    ],
        name="ternary"
    )
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            'accuracy',
        ],
        optimizer=tf.keras.optimizers.Adamax(metadata['learning_rate']),
    )
    return model


def precalculated_binary(input_shape, metadata) -> Tuple[tf.keras.Model, Dict]:
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(
            filters=6,
            kernel_size=10,
            strides=3,
            padding='valid',
            use_bias=False,
            quantizer=Binarize(),
            kernel_constraint=ChebyshevDistance(1.)
        ),
        BatchNormalization(),
        Binarize(),
        MaxPool1D(
            pool_size=8,
            strides=6,
        ),
        Conv1D(
            kernel_size=6,
            use_bias=False,
            filters=5,
            padding='valid',
        ),
        BatchNormalization(),
        Binarize(),
        Conv1D(
            kernel_size=6,
            use_bias=False,
            filters=5,
            padding='valid',
        ),
        BatchNormalization(),
        Binarize(),
        MaxPool1D(
            pool_size=3,
            strides=2,
        ),
        Conv1D(
            kernel_size=6,
            use_bias=False,
            filters=3,
            padding='valid',
        ),
        BatchNormalization(),
        QuantizeTwoBit(),
        MaxPool1D(
            pool_size=4,
            strides=4,
        ),
        MaxPool1D(
            pool_size=3,
            strides=3,
        ),
        Flatten(),
        Dense(
            units=1,
            use_bias=False,
            quantizer=Binarize(),
            kernel_constraint=ChebyshevDistance(1.)
        ),
        BatchNormalization(),
        Activation('sigmoid')
    ],
        name="precalculated_binary"
    )
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            'accuracy',
        ],
        optimizer=tf.keras.optimizers.Adamax(metadata['learning_rate']),
    )

    model.summary()
    return model


def precalculated_ternary(input_shape, metadata) -> Tuple[tf.keras.Model, Dict]:
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(
            filters=6,
            kernel_size=10,
            strides=3,
            padding='valid',
            use_bias=False,
            quantizer=Binarize(),
            kernel_constraint=ChebyshevDistance(1.)
        ),
        BatchNormalization(),
        Ternarize(),
        MaxPool1D(
            pool_size=8,
            strides=6,
        ),
        Conv1D(
            kernel_size=6,
            use_bias=False,
            filters=5,
            padding='valid',
        ),
        BatchNormalization(),
        Ternarize(),
        Conv1D(
            kernel_size=6,
            use_bias=False,
            filters=5,
            padding='valid',
        ),
        BatchNormalization(),
        Ternarize(),
        MaxPool1D(
            pool_size=3,
            strides=2,
        ),
        Conv1D(
            kernel_size=6,
            use_bias=False,
            filters=3,
            padding='valid',
        ),
        BatchNormalization(),
        QuantizeTwoBit(),
        MaxPool1D(
            pool_size=4,
            strides=4,
        ),
        MaxPool1D(
            pool_size=3,
            strides=3,
        ),
        Flatten(),
        Dense(
            units=1,
            use_bias=False,
            quantizer=Binarize(),
            kernel_constraint=ChebyshevDistance(1.)
        ),
        BatchNormalization(),
        Activation('sigmoid')
    ],
        name="preclculated_ternary"
    )
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            'accuracy',
        ],
        optimizer=tf.keras.optimizers.Adamax(metadata['learning_rate']),
    )

    model.summary()
    return model

