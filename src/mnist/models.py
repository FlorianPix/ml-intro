import tensorflow as tf


def single_layer_model():
    return multiple_layer_model(1)


def multiple_layer_model(number_of_layers):
    if number_of_layers < 1:
        number_of_layers = 1

    layers = [tf.keras.layers.Flatten(input_shape=(28, 28))]

    for _ in range(number_of_layers):
        layers.append(tf.keras.layers.Dense(128, activation='relu'))

    layers.append(tf.keras.layers.Dropout(0.2))
    layers.append(tf.keras.layers.Dense(10))

    return tf.keras.models.Sequential(layers)
