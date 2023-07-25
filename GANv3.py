import tensorflow as tf
from tensorflow.keras.layers import Dense, LeakyReLU, Reshape, Flatten, BatchNormalization, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import wandb
import numpy as np
import pickle
import pretty_midi

cache_file = r"C:\Users\Pablo\Documents\MusikIA\Numpy\matrices10000.pickle"
with open(cache_file, 'rb') as f:
    datos = np.array(pickle.load(f))

max_values = np.array([2,1170,127,127,127])

wandb.login(key="26ab38e8f6e471ce6662ff95ea15c50993b6d4a1")
wandb.init(project='MusikAI_V4')

def load_data(data, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(data).shuffle(data.shape[0])
    dataset = dataset.batch(batch_size, drop_remainder=True)
    print(f'Dataset completo: {data}')

    return dataset

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
    model.add(Dense(np.prod(INPUT_SHAPE), activation='sigmoid'))
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

def train_epoch(generator, discriminator, dataset, batch_size):
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    gen_notes = generator.predict(noise)

    real_notes = None
    for batch in dataset:
        real_notes = batch
        break

    if real_notes is None:
        print("No batches found in the dataset.")
        return None, None

    real_notes = real_notes.numpy()[:batch_size]

    notes = np.concatenate([gen_notes, real_notes], axis=0)
    labels = np.concatenate([np.zeros((batch_size, 1)), np.ones((batch_size, 1))], axis=0)

    d_loss = discriminator.train_on_batch(notes, labels)

    misleading_targets = np.ones((batch_size, 1))
    noise = np.random.normal(0, 1, (batch_size, latent_dim))

    g_loss = gan.train_on_batch(noise, misleading_targets)

    return d_loss, g_loss

def generate_midi(filename, length):
    min_note = 0
    max_note = 127
    noise = np.random.normal(0, 1, (length, latent_dim))
    print(noise)

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
        try:
            pitch = int(note[2])
            program = int(note[4])
            pm_note = pretty_midi.Note(
                velocity=int(note[3]),
                pitch=pitch,
                start=note[0],
                end=note[1] + note[0]
            )

            pm_instrument.notes.append(pm_note)
            pm_instrument.program = program
            pm_instrument.append(pm_instrument)
        except Exception as e:
            print(f"Error al procesar la nota: {note}, {e}")

    pm.instruments.append(pm_instrument)

    pm.write(filename)


wandb.login(key="26ab38e8f6e471ce6662ff95ea15c50993b6d4a1")
wandb.init(project="MusikAI_V4")

latent_dim = 100
epochs = 100
num_samples = 10
input_shape = (10000, 5)
learning_rate = 0.001

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

    if epoch % 50 == 0:
        noise = np.random.normal(0, 1, (1, latent_dim))
        generated_data = generator.predict(noise)
        generated_data = generated_data * max_values
        generated_data = generated_data.astype(int)

    wandb.log({'epoch': epoch, 'd_loss': d_loss[0], 'g_loss': g_loss})
    print(f"Epoch {epoch}: Discriminator loss = {d_loss[0]}, Generator loss = {g_loss}")

generator.save(f"{wandb.run.dir}/generator.h5")

print('generando')

generate_midi(r"C:\Users\Pablo\Documents\canción_de_prueba.mid", length=64)

print('Hecho')
