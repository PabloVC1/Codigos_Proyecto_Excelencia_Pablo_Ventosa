import tensorflow as tf
from keras.layers import Dense, LeakyReLU, Reshape, Flatten, Input
from keras.layers import BatchNormalization, Input
from keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
import wandb
import os
import numpy as np
import pickle
import pretty_midi

cache_file = r"C:\Users\Pablo\Desktop\Falso_caché\Big_Matrices.pickle"
with open(cache_file, 'rb') as f:
    datos = np.array(pickle.load(f))

max_values = [11055, 11311, 127, 127, 127]

wandb.login(key="26ab38e8f6e471ce6662ff95ea15c50993b6d4a1")
wandb.init(project='MusikAI_V4')

def load_data(datos, batch_size):
    filtered_mats = []
    max_seq_length = 10000
    for i in range(0, len(datos), batch_size):
        for m in datos:
            num_rows = m.shape[0]
            if num_rows <= max_seq_length:
                # Si la matriz tiene menos filas que el número de columnas deseado, agregar filas de ceros
                zeros = np.zeros((max_seq_length - num_rows, m.shape[1]), dtype=int)
                m = np.concatenate((m, zeros), axis=0)
                filtered_mats.append(m)
        batch = np.concatenate(datos[i:i+batch_size], axis=0)
        filtered_mats.append(batch)
        print(i)
    filtered_mats = np.concatenate(filtered_mats, axis=0)

    print(filtered_mats)
    filtered_mats = list(filter(lambda x: len(x) > 0, filtered_mats))
    filtered_mats = np.concatenate(filtered_mats, axis=0)
    print('filtradas')
    norm_mats = filtered_mats/ max_values[:5]
    norm_mats = np.clip(norm_mats, 0, 1)
    return norm_mats

# definir la arquitectura del generador
def make_generator_model(INPUT_SHAPE, LATENT_DIM):
    model = tf.keras.Sequential()
    model.add(Dense(500, input_dim=LATENT_DIM))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1000))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(5000))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(np.prod(INPUT_SHAPE), activation='sigmoid'))
    model.add(Reshape(INPUT_SHAPE))

    model.summary()

    return model

# definir la arquitectura del discriminador
def make_discriminator_model(INPUT_SHAPE):
    model = tf.keras.Sequential()
    model.add(Flatten(input_shape=INPUT_SHAPE))
    model.add(Dense(5000))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1000))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    return model

# Definir una función para guardar las muestras generadas
def save_samples(epoch, samples):
    filename = f'generated_music/sample_epoch{epoch}.csv'
    np.savetxt(filename, samples, delimiter=',', fmt='%.4f')

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


def generate_midi(filename, length):
    min_note = 0
    max_note = 127
    noise = np.random.normal(0, 1, (length, latent_dim))

    generated_notes = generator.predict(noise)

    print(generated_notes)

    generated_notes = generated_notes * max_values[:5]
    generated_notes[:, 4] = (generated_notes[:, 4] - min_note) / (max_note - min_note)
    generated_notes[:, 4] = generated_notes[:, 4] * 35 + 60
    generated_notes = np.round(generated_notes)

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
wandb.init(project="MusikAI_V4")

latent_dim = 100
epochs = 3000
num_samples = 10
input_shape = (5,)
learning_rate = 0.0001

generator = make_generator_model(input_shape, latent_dim)
discriminator = make_discriminator_model(input_shape)

discriminator.compile(loss='binary_crossentropy',
                      optimizer=Adam(learning_rate, 0.5),
                      metrics=['accuracy'])

z = Input(shape=(latent_dim,))
note = generator(z)
valid = discriminator(note)
gan = Model(z, valid)
gan.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate, 0.5))

dataset = load_data(datos, batch_size=1000)
print(dataset)
batch_size = 100

for epoch in range(epochs):
    d_loss, g_loss = train_epoch(generator, discriminator, dataset, batch_size)

    # Generar y guardar muestras
    if epoch % 50 == 0:
        noise = tf.random.normal([num_samples, latent_dim])
        generated_data = generator(noise, training=False)
        generated_data = generated_data * max_values
        generated_data = generated_data.numpy().astype(int)
        save_samples(epoch, generated_data)

    # Registrar métricas en wandb
    wandb.log({'epoch': epoch, 'd_loss': d_loss, 'g_loss': g_loss})
    print(f"Epoch {epoch}: Discriminator loss = {d_loss}, Generator loss = {g_loss}")

generator.save(f"{wandb.run.dir}/generator.h5")

print('generando')

generate_midi(r"C:\Users\Pablo\Documents\canción_de_prueba.mid", length=64)

print('Hecho')
