import tensorflow as tf
import wandb
import numpy as np
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, LeakyReLU
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D, UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam
import os
import pickle
from tensorflow.keras.utils import plot_model
import pretty_midi

cache_folder = r"C:\Users\Pablo\Desktop\Nueva carpeta\Falso_caché"
matrices_originales = os.path.join(cache_folder, 'cache.pickle')
with open(matrices_originales, 'rb') as f:
    datos = pickle.load(f)

# Función para construir el generador
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

# Función para construir el discriminador
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

# Función para cargar los datos
def load_data(datos):
    filtered_mats = []

    # Recorrer lista de matrices
    for mat in datos:
        # Comprobar número de filas de la matriz
        if mat.shape[0] < 10000:
            # Si tiene menos de 10000 filas, añadirla a la lista de matrices filtradas
            filtered_mats.append(mat)

    # Devolver lista de matrices filtradas
    return filtered_mats

# Función para entrenar una sola época de la GAN
def train_epoch(generator, discriminator, dataset, batch_size):
    # Creamos un vector de ruido para el generador
    noise = np.random.normal(0, 1, (batch_size, latent_dim))

    # Generamos notas musicales falsas con el generador
    gen_notes = generator.predict(noise)

    # Obtenemos notas musicales reales del conjunto de datos
    idx = np.random.randint(0, dataset.shape[0], batch_size)
    real_notes = dataset[idx]

    # Creamos una matriz de entrada para el discriminador
    # Creamos una matriz de entrada para el discriminador
    notes = np.concatenate((gen_notes, real_notes))
    labels = np.concatenate((np.zeros((batch_size, 1)), np.ones((batch_size, 1))))

    # Entrenamos el discriminador en esta época
    d_loss = discriminator.train_on_batch(notes, labels)

    # Creamos un vector de ruido para el generador
    noise = np.random.normal(0, 1, (batch_size, latent_dim))

    # Creamos una matriz de entrada para el generador
    misleading_targets = np.ones((batch_size, 1))

    # Entrenamos la GAN en esta época
    g_loss = gan.train_on_batch(noise, misleading_targets)

    return d_loss, g_loss

def note_sequence_to_pretty_midi(notes):
    # Crear un objeto PrettyMIDI vacío
    pm = pretty_midi.PrettyMIDI()
    # Crear un objeto Instrument para almacenar las notas
    instrument = pretty_midi.Instrument(program=0)

    # Convertir cada nota a un objeto Note de PrettyMIDI y agregarlo al instrumento
    for note in notes:
        pitch = int(note[3])
        start_time = note[1]
        end_time = note[2]
        velocity = int(note[0].item() * 127)
        # Crear un objeto Note y agregarlo al instrumento
        pm_note = pretty_midi.Note(
            velocity=velocity, pitch=pitch, start=start_time, end=end_time)
        instrument.notes.append(pm_note)

    # Agregar el instrumento al objeto PrettyMIDI y devolverlo
    pm.instruments.append(instrument)
    return pm


def generate_midi(model, filename, length):
    # Generar notas
    noise = np.random.normal(0, 1, (1, 100))
    generated_notes = model.predict(noise)[0]
    generated_notes = np.array([generated_notes[i:i + 5] for i in range(0, len(generated_notes), 5)])

    # Crear objeto PrettyMIDI
    pm = pretty_midi.PrettyMIDI()
    pm_instrument = pretty_midi.Instrument(program=0)

    # Transformar notas generadas a notas de midi
    for note in generated_notes:
        pitch = int(note[3])
        program = int(note[4])
        pm_note = pretty_midi.Note(
            velocity=int(note[0] * 127),
            pitch=pitch,
            start=note[1] * length,
            end=note[2] * length
        )
        pm_instrument.notes.append(pm_note)
        pm_instrument.program = program

    pm.instruments.append(pm_instrument)

    # Exportar archivo midi
    pm.write(r"C:\Users\Pablo\Desktop\cancion_generada.mid")


# Configuramos WandB con nuestra API key
wandb.login(key="26ab38e8f6e471ce6662ff95ea15c50993b6d4a1")
wandb.init(project="MusikAI_V1")

# Dimensiones de entrada para el generador
latent_dim = 100
epochs = 15000
# Dimensiones de entrada para las notas musicales
input_shape = (5,)

# Construimos el generador y el discriminador
generator = build_generator(latent_dim, input_shape)
discriminator = build_discriminator(input_shape)

# Compilamos el discriminador
discriminator.compile(loss='binary_crossentropy',
                      optimizer=Adam(0.0002, 0.5),
                      metrics=['accuracy'])

# Creamos la GAN conectando el generador y el discriminador
z = Input(shape=(latent_dim,))
note = generator(z)
valid = discriminator(note)
gan = Model(z, valid)
gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

# Cargamos los datos de entrenamiento
dataset = np.concatenate(load_data(datos), axis=0)

# Tamaño de los lotes para el entrenamiento
batch_size = 32

# Entrenamos la GAN
for epoch in range(epochs):
    d_loss, g_loss = train_epoch(generator, discriminator, dataset, batch_size)

    # Reportamos el progreso en WandB
    wandb.log({'epoch': epoch, 'd_loss': d_loss, 'g_loss': g_loss})
    print(f"Epoch {epoch}: Discriminator loss = {d_loss}, Generator loss = {g_loss}")

# Guardamos el modelo entrenado en WandB
generator.save(f"{wandb.run.dir}/generator.h5")

print('generando')
num_notes = 20

# Generar una secuencia de notas de música utilizando la función generate
generate_midi(generator, r"\Desktop\cancion_generada.midi", length=256)

print('Hecho')