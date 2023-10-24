import tensorflow as tf
from keras.layers import Dense, LeakyReLU, Reshape, Flatten, BatchNormalization, Input
from keras.models import Model
from keras.optimizers import Adam
import wandb
import random
import json
import numpy as np
import pretty_midi

datos = np.load("midi_matrices.npy")
max_values = np.array([5306767, 5429390, 127, 127, 127])
resolution = 1920

wandb.login(key="26ab38e8f6e471ce6662ff95ea15c50993b6d4a1")
wandb.init(project='MusikAI_V5')

def load_data(data, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(data).shuffle(data.shape[0])
    dataset = dataset.batch(batch_size, drop_remainder=True)
    print(f'Dataset completo: {data}')

    return dataset

lista_duraciones = [1.5, 1.0, 0.75, 0.5, 0.375, 0.25, 0.1875, 0.125, 0.0625]
n = 2.0
normalization_values = [20000, 2, 127, 127, 127]
num_compases = 4
midi_file_directory = '/content/drive/MyDrive/canción_de_prueba.mid'
notas_midi = {"C": 60, "C#": 61, "Db": 61, "D": 62, "D#": 63, "Eb": 63, "E": 64, "F": 65, "F#": 66, "Gb": 66, "G": 67, "G#": 68, "Ab": 68, "A": 69, "A#": 70, "Bb": 70, "B": 71}
escalas_circulo_quintas = {
    "C Mayor": ["C", "D", "E", "F", "G", "A", "B"],
    "G Mayor": ["G", "A", "B", "C", "D", "E", "F#"],
    "D Mayor": ["D", "E", "F#", "G", "A", "B", "C#"],
    "A Mayor": ["A", "B", "C#", "D", "E", "F#", "G#"],
    "E Mayor": ["E", "F#", "G#", "A", "B", "C#", "D#"],
    "B Mayor": ["B", "C#", "D#", "E", "F#", "G#", "A#"],
    "F Mayor": ["F", "G", "A", "Bb", "C", "D", "E"],
    "Bb Mayor": ["Bb", "C", "D", "Eb", "F", "G", "A"],
    "Eb Mayor": ["Eb", "F", "G", "Ab", "Bb", "C", "D"],
    "Ab Mayor": ["Ab", "Bb", "C", "Db", "Eb", "F", "G"],
    "Db Mayor": ["Db", "Eb", "F", "Gb", "Ab", "Bb", "C"]
}

tipos_acordes = {
    1: [0, 4, 7],
    2: [0, 3, 7],
    3: [0, 3, 7],
    4: [0, 4, 7],
    5: [0, 4, 7],
    6: [0, 3, 7],
    7: [0, 3, 6]
}

progresiones_mayor = {
    1: [1, 6, 3, 4],
    2: [1, 6, 4, 5],
    3: [2, 5, 1, 6],
    4: [1, 6, 2, 5],
    5: [1, 4, 6, 5],
    6: [1, 5, 2, 4],
    7: [1, 3, 4, 5],
    8: [6, 2, 5, 1],
    9: [1, 4, 2, 5]
}
def cambiar_disposicion(porgresiones_mayor):
    progresion_aleatoria = random.choice(list(progresiones_mayor.values()))
    indice_aleatorio = random.randint(0, 3)
    progresion_cambiada = progresion_aleatoria[indice_aleatorio:] + progresion_aleatoria[:indice_aleatorio]
    return progresion_cambiada, progresion_aleatoria

def generar_escala_aleatoria(diccionario_escalas, progresion_acordes):
    nombre_escala, notas_escala = random.choice(list(diccionario_escalas.items()))
    notas_progresion = [notas_escala[chord - 1] for chord in progresion_acordes]

    notas_acorde = [tipos_acordes[chord] for chord in progresion_acordes]

    progresion_final = dict(zip(notas_progresion, notas_acorde))

    return nombre_escala, progresion_final

notas_midi = {"C": 60, "C#": 61, "Db": 61, "D": 62, "D#": 63, "Eb": 63, "E": 64, "F": 65, "F#": 66, "Gb": 66, "G": 67, "G#": 68, "Ab": 68, "A": 69, "A#": 70, "Bb": 70, "B": 71}

def notas_de_escala_para_acorde(acorde, notas_midi_dict, notas_progresion):
    notas_acorde = notas_progresion[acorde]
    raiz = notas_midi_dict[acorde]
    notas_escala = []
    for semitono in notas_acorde:
            nota_midi = (raiz + semitono)
            notas_escala.append(nota_midi)

    return notas_escala

def get_notes(progresion_acordes):
    nombre_escala, notas_progresion = generar_escala_aleatoria(escalas_circulo_quintas, progresion_acordes)
    return notas_progresion, nombre_escala

def cambiar_parametros(matriz):
    primer_valor = 0

    for fila in matriz:
        fila[1] = lista_duraciones[int(fila[1] * (len(lista_duraciones) - 1))]
        fila[0] = primer_valor
        fila[4] = 0
        fila[3] = 100
        primer_valor = fila[0] + fila[1]

    return matriz


def change_matrix(matrix, progresion_acordes, compas_inicial=0):
    notas_progresion, nombre_escala = get_notes(progresion_acordes)
    matrix = cambiar_parametros(matrix)
    compas = compas_inicial

    for acorde in notas_progresion:
        notas_escala = notas_de_escala_para_acorde(acorde, notas_midi, notas_progresion)
        for fila in matrix:
            if fila[0] < compas + 2:
                indice = int(min(1.0, max(0.0, fila[2])) * (len(notas_escala) - 1))
                fila[2] = round(notas_escala[indice])
            else:
                break

        compas += 2
    matrix = matrix[:-12]

    nuevas_filas = crear_acordes_matrices(notas_progresion)
    matrix = np.vstack([matrix, nuevas_filas])

    return matrix

def crear_acordes_matrices(diccionario_acordes):
    matrices_acordes = []
    tiempo_inicio = 0

    for acorde, intervalos in diccionario_acordes.items():
        for intervalo in intervalos:
            fila = [tiempo_inicio, 2.0, notas_midi[acorde] + intervalo - 12, 100, 0]
            matrices_acordes.append(fila)

        tiempo_inicio += 2.0

    return np.array(matrices_acordes)

def make_generator_model(INPUT_SHAPE, LATENT_DIM):
    model = tf.keras.Sequential()
    model.add(Dense(64, input_dim=LATENT_DIM))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(128))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(np.prod(5), activation='sigmoid'))
    model.add(Reshape(INPUT_SHAPE))
    return model

def make_discriminator_model(INPUT_SHAPE):
    model = tf.keras.Sequential()
    model.add(Flatten(input_shape=INPUT_SHAPE))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(128))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(64))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    return model

def train_epoch(generator, discriminator, gan, dataset, batch_size):
    gen_notes = []
    for batch in dataset:
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        song = generator.predict(noise)
        progresion_acordes, progresion_aleatoria = cambiar_disposicion(progresiones_mayor)
        new_matrix = change_matrix(song, progresion_acordes)
        normalized_matrix = new_matrix / normalization_values[:5]
        gen_notes.append(normalized_matrix)

        if len(gen_notes) >= batch_size:
            break

    gen_notes = np.concatenate(gen_notes, axis=0)

    if len(gen_notes) < batch_size:
        print("Insufficient number of batches in the dataset.")
        return None, None

    real_notes = batch[:batch_size]

    notes = np.concatenate([gen_notes, real_notes], axis=0)
    labels = np.concatenate([np.zeros((10000, 1)), np.ones((batch_size, 1))], axis=0)

    d_loss = discriminator.train_on_batch(notes, labels)

    misleading_targets = np.ones((batch_size, 1))
    noise = np.random.normal(0, 1, (batch_size, 100))

    g_loss = gan.train_on_batch(noise, misleading_targets)

    return d_loss, g_loss


def generate_midi(midiname, filename, length, resolution=1):
    noise = np.random.normal(0, 1, (length, latent_dim))

    generated_notes = generator.predict(noise)
    progresion_acordes, progresion_aleatoria = cambiar_disposicion(progresiones_mayor)
    new_matrix = change_matrix(generated_notes, progresion_acordes)
    print(new_matrix)


    with open(filename + '.json', 'w') as json_file:
        json.dump(new_matrix.tolist(), json_file)

    pm = pretty_midi.PrettyMIDI()
    pm_instrument = pretty_midi.Instrument(program=0)

    for note in generated_notes:
        pitch = int(note[2])
        start_time = note[0]
        end_time = note[1] + note[0]
        program = 1
        pm_note = pretty_midi.Note(
            velocity=int(note[3]),
            pitch=pitch,
            start=start_time,
            end=end_time
        )
        pm_instrument.notes.append(pm_note)
        pm_instrument.program = program

    pm.instruments.append(pm_instrument)
    pm.write(midiname + '.midi')

latent_dim = 100
epochs = 4
num_samples = 10
input_shape = (5,)
learning_rate = 0.002

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

dataset = load_data(datos, batch_size=100)
print(dataset)
batch_size = 100

for epoch in range(epochs):
    d_loss, g_loss = train_epoch(generator, discriminator, gan, dataset=dataset, batch_size=batch_size)

    if epoch % 50 == 0:
        noise = np.random.normal(0, 1, (1, latent_dim))
        generated_data = generator.predict(noise)
        generated_data = generated_data * max_values
        generated_data = generated_data.astype(int)

    wandb.log({'epoch': epoch, 'd_loss': d_loss[0], 'g_loss': g_loss})
    print(f"Epoch {epoch}: Discriminator loss = {d_loss[0]}, Generator loss = {g_loss}")

generator.save(f"{wandb.run.dir}/generator.h5")

print('generating')

generate_midi('output_midi', 'output_json', length=100, resolution=0.1)

print('Done')
