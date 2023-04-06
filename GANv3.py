import tensorflow as tf
import wandb
import numpy as np
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, LeakyReLU
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D, UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam
import os
import pickle
import pretty_midi

cache_folder = r"C:\Users\Pablo\Desktop\Nueva carpeta\Falso_caché"
matrices_originales = os.path.join(cache_folder, 'cache.pickle')
with open(matrices_originales, 'rb') as f:
    datos = pickle.load(f)


def build_generator(latent_dim, input_shape):
    model = Sequential()

    model.add(Dense(128, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(np.prod(input_shape), activation='tanh'))
    model.add(Reshape(input_shape))

    model.summary()

    noise = Input(shape=(latent_dim,))
    note = model(noise)

    return Model(noise, note)


def build_discriminator(input_shape):
    model = Sequential()

    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    note = Input(shape=input_shape)
    validity = model(note)

    return Model(note, validity)


def load_data(datos):
    filtered_mats = []
    max_seq_length = 10000
    for matrix in datos:
        if matrix.shape[0] > max_seq_length:
            continue
        matriz_vacia = np.zeros((max_seq_length, 5))
        matriz_vacia[:matrix.shape[0], :matrix.shape[1]] = matrix
        filtered_mats.append(matriz_vacia)

    filtered_mats = list(filter(lambda x: len(x) > 0, datos))
    filtered_mats = np.concatenate(filtered_mats, axis=0)
    print('filtradas')
    min_val = np.min(filtered_mats)
    max_val = np.max(filtered_mats)
    norm_mats = (filtered_mats - min_val) / (max_val - min_val + 1e-7)
    norm_mats = np.clip(norm_mats, 0, 1)
    return norm_mats, (min_val, max_val)


def train_epoch(generator, discriminator, dataset, batch_size):
    noise = np.random.normal(0, 1, (batch_size, latent_dim))

    gen_notes = generator.predict(noise)

    idx = np.random.randint(0, dataset.shape[0], batch_size)
    real_notes = dataset[idx]

    notes = np.concatenate((gen_notes, real_notes))
    labels = np.concatenate((np.zeros((batch_size, 1)), np.ones((batch_size, 1))))

    d_loss = discriminator.train_on_batch(notes, labels)

    noise = np.random.normal(0, 1, (batch_size, latent_dim))

    misleading_targets = np.ones((batch_size, 1))

    g_loss = gan.train_on_batch(noise, misleading_targets)

    return d_loss, g_loss


def generate_midi(filename, length, original_range):
    noise = np.random.normal(0, 1, (1, latent_dim))

    generated_notes = generator.predict(noise)
    for _ in range(length - 1):
        noise = np.random.normal(0, 1, (1, latent_dim))
        note = generator.predict(noise)
        generated_notes = np.vstack((generated_notes, note))

    min_val, max_val = original_range
    generated_notes = generated_notes * (max_val - min_val) + min_val
    print(max_val, min_val)

    print(generated_notes)

    pm = pretty_midi.PrettyMIDI()
    pm_instrument = pretty_midi.Instrument(program=0)

    for note in generated_notes:
        pitch = int(note[2])
        program = int(note[4])
        pm_note = pretty_midi.Note(
            velocity=int(note[3]),
            pitch=pitch,
            start=note[0],
            end=note[1]
        )
        pm_instrument.notes.append(pm_note)
        pm_instrument.program = program

    pm.instruments.append(pm_instrument)

    pm.write(filename)


wandb.login(key="26ab38e8f6e471ce6662ff95ea15c50993b6d4a1")
wandb.init(project="MusikAI_V2")

latent_dim = 100
epochs = 1000

input_shape = (5,)

generator = build_generator(latent_dim, input_shape)
discriminator = build_discriminator(input_shape)

discriminator.compile(loss='binary_crossentropy',
                      optimizer=Adam(0.0002, 0.5),
                      metrics=['accuracy'])

z = Input(shape=(latent_dim,))
note = generator(z)
valid = discriminator(note)
gan = Model(z, valid)
gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

dataset, original_range = load_data(datos)
print(dataset)
batch_size = 32

for epoch in range(epochs):
    d_loss, g_loss = train_epoch(generator, discriminator, dataset, batch_size)

    wandb.log({'epoch': epoch, 'd_loss': d_loss, 'g_loss': g_loss})
    print(f"Epoch {epoch}: Discriminator loss = {d_loss}, Generator loss = {g_loss}")

generator.save(f"{wandb.run.dir}/generator.h5")

print('generando')

generate_midi(r"\Desktop\cancion_generada.midi", length=64, original_range=original_range)

print('Hecho')
