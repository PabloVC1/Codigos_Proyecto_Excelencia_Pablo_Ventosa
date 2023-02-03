import tensorflow as tf
import tensorflow_gan as tfgan
import matplotlib.pyplot as plt
import numpy as np
import os
import pretty_midi

tf.config.run_functions_eagerly(False)
def load_midi_data(root_dir):
    matrices = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            name, ext = os.path.splitext(filename)
            if ext == ".mid":
                file_path = os.path.join(dirpath, filename)
                try:
                    midi_data = pretty_midi.PrettyMIDI(file_path)
                    piano_roll = midi_data.get_piano_roll(fs=100)
                except Exception as e:
                    print(f"No se pudo procesar el archivo {filename}: {e}")
                    continue
                piano_roll = (piano_roll + 1) / 2
                piano_roll = np.int32(piano_roll)
                matrices.append(piano_roll)

matrices = load_midi_data(r'C:\Users\Pablo\Desktop\clean_midi\A-I')
max_matrix = np.amax(matrices, axis=0)
for i in range(len(matrices)):
    matrices[i] = np.pad(matrices[i], ((0, max_matrix.shape[0] - matrices[i].shape[0]),
                                       (0, max_matrix.shape[1] - matrices[i].shape[1])), 'constant')
real_data = np.array(matrices)

def generator(noise_input, training=True):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation="relu", input_shape=(100,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(piano_roll.shape[1], activation="sigmoid")
    ])
    return model(noise_input)


def discriminator(input_data, training=True):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(512, activation="relu", input_shape=(piano_roll.shape[1],)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    return model(input_data)

with tf.device('/GPU:0'):
    noise_input = tf.keras.layers.Input(shape=(100,))
    real_data = tf.keras.layers.Input(shape=(max_matrix.shape[1],))

    gan = tfgan.gan_model(generator, discriminator, real_data, generator_inputs = [noise_input])

    gan.compile(optimizer=tf.keras.optimizers.Adam(0.0002, 0.5), loss="binary_crossentropy")
    gan.fit(x=[np.random.normal(size=(piano_roll.shape[0], 100)), piano_roll], y=np.ones((piano_roll.shape[0], 1)), batch_size=32, epochs=100)