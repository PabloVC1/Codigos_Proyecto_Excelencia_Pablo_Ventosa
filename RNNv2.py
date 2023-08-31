import tensorflow as tf
from keras.layers import Input, LSTM, Attention, Concatenate, Reshape, Dense, Add
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch

# Cargar los datos desde el archivo .npy
data = np.load("midi_matrices.npy")

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
epochs = 10
batch_size = 32
sequence_length = 10000
num_features = 5

num_segments = data.shape[0] // sequence_length
segments = np.array_split(data, num_segments)

def CustomLoss(predictions, targets, weights):
    loss_per_feature = torch.abs(predictions - targets)
    weighted_loss_per_feature = loss_per_feature * weights
    total_loss = torch.sum(weighted_loss_per_feature)

    return total_loss

def make_custom_model(sequence_length, num_features):
    input_layer = Input(shape=(sequence_length, num_features))

    # Shared LSTM layer
    shared_lstm = LSTM(units=hidden_units, return_sequences=True, return_state=True)

    # Apply shared LSTM layer to input
    shared_lstm_outputs, _, _ = shared_lstm(input_layer)

    # Individual LSTM layer for each feature
    individual_lstms = [LSTM(units=hidden_units, return_sequences=True)(shared_lstm_outputs) for _ in
                        range(num_features)]

    lstm_concatenated = Concatenate(axis=-1)(individual_lstms)

    # Temporal attention layer for each matrix
    attention_layers = [Attention()([individual_lstms[i], shared_lstm_outputs]) for i in range(num_features)]

    # Concatenate attention outputs along the last axis
    attention_concatenated = Concatenate(axis=-1)(attention_layers)

    # Residual layer
    residual_layer = Add()([attention_concatenated, lstm_concatenated])

    # Hidden layer
    hidden_layer = Dense(units=hidden_units)(residual_layer)

    # LSTM layer
    final_lstm = LSTM(units=hidden_units, return_sequences=True)(hidden_layer)

    # Attention layer
    final_attention = Attention()([final_lstm, hidden_layer])

    # Feedback layer
    feedback_layer = Dense(units=hidden_units)(final_attention)

    # Output layer
    output_layer = Dense(units=output_units)(feedback_layer)

    # Define the model
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

model = make_custom_model(sequence_length, num_features)
model.summary()

# Optimizador
optimizer = tf.keras.optimizers.Adam()
weights = torch.tensor([w_start, w_duration, w_pitch, w_velocity, w_instrument], dtype=torch.float32)

start_data = data[:, 0]
duration_data = data[:, 1]
pitch_data = data[:, 2]
velocity_data = data[:, 3]
instrument_data = data[:, 4]

dataset = TensorDataset(start_data, duration_data, pitch_data, velocity_data, instrument_data)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Ciclo de entrenamiento personalizado
epochs = 10
batch_size = 32

num_epochs = 50
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        # Obtener las características del lote
        start, duration, pitch, velocity, instrument = batch

        # Pasar los datos por el modelo
        predictions = model(start, duration, pitch, velocity, instrument)

        # Calcular la pérdida utilizando la función de pérdida personalizada
        loss = CustomLoss(predictions, batch, weights)

        # Realizar la retropropagación y la actualización de parámetros
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")