import tensorflow as tf
import torch.nn as nn
from keras.layers import Input, LSTM, Attention, Concatenate, Reshape, Dense, Add
import numpy as np
from numpy import float32
from torch.utils.data import DataLoader, TensorDataset
import torch
import wandb

# Cargar los datos desde el archivo .npy
datos = np.load("midi_matrices.npy")

x_train = datos[:-1]
y_train = datos[1:]

# Parámetros del modelo
w_start = 1.0
w_duration = 1.0
w_pitch = 2.0
w_velocity = 0.5
w_instrument = 0.5

sequence_length = 10000
num_features = 5
hidden_units = 64
output_units = num_features
epochs = 1
batch_size = 32
learning_rate = 0.01
num_features = 5

num_segments = datos.shape[0] // sequence_length
segments = np.array_split(datos, num_segments)

wandb.login(key="26ab38e8f6e471ce6662ff95ea15c50993b6d4a1")
wandb.init(project='RNNv1')

def load_data(data, batch_size):
    dataset = torch.tensor(data).float()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

data_loader = load_data(datos, batch_size=batch_size)

class CustomLoss(nn.Module):
    def __init__(self, weights):
        super(CustomLoss, self).__init__()
        self.weights = weights

    def forward(self, y_true, y_pred):
        # Convertir targets a float32 si es necesario
        y_true = tf.cast(y_true, tf.float32)

        loss_per_feature = tf.reduce_mean(tf.square(y_true - y_pred))

        # Aplicar los pesos a cada característica
        weighted_loss_per_feature = loss_per_feature * self.weights

        # Sumar las pérdidas ponderadas a lo largo de las características
        total_loss = tf.reduce_sum(weighted_loss_per_feature)

        return total_loss

def make_custom_model(sequence_length, num_features):
    input_layer = tf.keras.layers.Input(shape=(5,))

    reshaped_input = tf.keras.layers.Reshape((1, 5))(input_layer)

    # Capa LSTM compartida
    shared_lstm = tf.keras.layers.LSTM(64, return_sequences=True)(reshaped_input)

    # Listas para almacenar capas individuales
    lstm_layers = []
    attention_layers = []

    # Capas LSTM individuales y capas de atención individuales
    for i in range(5):
        lstm = tf.keras.layers.LSTM(64, return_sequences=True)(shared_lstm)
        lstm_layers.append(lstm)

        attention = tf.keras.layers.Attention()([lstm, shared_lstm])
        attention_layers.append(attention)

    # Concatenación de los resultados de atención
    concatenated_attention = tf.keras.layers.Concatenate(axis=-1)(attention_layers)

    concatenated_lstm = tf.keras.layers.Concatenate(axis=-1)(lstm_layers)

    # Capa residual (Suma)
    residual_layer = tf.keras.layers.Add()([concatenated_attention, concatenated_lstm])

    # Capas ocultas
    hidden_layer = tf.keras.layers.Dense(128, activation='relu')(residual_layer)

    # LSTM final
    final_lstm = tf.keras.layers.LSTM(64, return_sequences=True)(hidden_layer)

    # Atención final
    final_attention = tf.keras.layers.Attention()([final_lstm, shared_lstm])

    # Capa de feedback
    feedback_layer = tf.keras.layers.Dense(64, activation='relu')(final_attention)

    # Capa de salida (matriz de 1x5)
    output_layer = tf.keras.layers.Dense(5, activation='linear')(feedback_layer)

    # Crear el modelo
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    # Compilar el modelo
    model.compile(optimizer='adam', loss=criterion)

    return model

# Optimizador
optimizer = tf.keras.optimizers.Adam()
weights = torch.tensor([w_start, w_duration, w_pitch, w_velocity, w_instrument], dtype=torch.float32)
criterion = CustomLoss(weights)
wandb_callback = wandb.keras.WandbCallback()
rnn = make_custom_model(sequence_length, num_features)
rnn.summary()

rnn.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[wandb_callback])
