import torch
import torch.nn as nn
import numpy as np
import wandb
from torch.utils.data import DataLoader, TensorDataset
import pretty_midi as pm

wandb.login(key="26ab38e8f6e471ce6662ff95ea15c50993b6d4a1")
wandb.init(project='RNNv1')

w_start = 1.0
w_duration = 1.0
w_pitch = 2.0
w_velocity = 0.5
w_instrument = 0.5

input_size = 1
hidden_size = 64
batch_size = 32
output_size = 5
seq_length = 200
num_layers = 4
learning_rate = 0.01
num_epochs = 1
resolution = 480
num_features = 5


class CustomLoss(nn.Module):
    def __init__(self, weights):
        super(CustomLoss, self).__init__()
        self.weights = weights

    def forward(self, predictions, targets):
        loss_per_feature = torch.abs(predictions - targets)

        # Aplicar los pesos a cada característica
        weighted_loss_per_feature = loss_per_feature * weights

        # Sumar las pérdidas ponderadas a lo largo de las características
        total_loss = torch.sum(weighted_loss_per_feature)

        return total_loss

class FeatureDevelopment(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(FeatureDevelopment, self).__init__()
        self.start_layer = nn.Linear(hidden_size, input_size)
        self.duration_layer = nn.Linear(hidden_size, input_size)
        self.pitch_layer = nn.Linear(hidden_size, input_size)
        self.velocity_layer = nn.Linear(hidden_size, input_size)
        self.instrument_layer = nn.Linear(hidden_size, input_size)

    def forward(self, start, duration, pitch, velocity, instrument):
        start_embed = self.start_layer(start)
        duration_embed = self.duration_layer(duration)
        pitch_embed = self.pitch_layer(pitch)
        velocity_embed = self.velocity_layer(velocity)
        instrument_embed = self.instrument_layer(instrument)
        return start_embed, duration_embed, pitch_embed, velocity_embed, instrument_embed

class BidirectionalLSTMStack(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_directions):
        super(BidirectionalLSTMStack, self).__init__()
        self.lstm_stack = nn.ModuleList([
            nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
            for _ in range(num_directions)
        ])

    def forward(self, x):
        outputs = []
        for lstm_layer in self.lstm_stack:
            lstm_out, _ = lstm_layer(x)
            outputs.append(lstm_out)
        return torch.cat(outputs, dim=-1)

class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super(TemporalAttention, self).__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4)

    def forward(self, x):
        attended, _ = self.attention(x, x, x)
        return attended

class SharedLayersWithResiduals(nn.Module):
    def __init__(self, hidden_size, num_layers, num_shared_layers):
        super(SharedLayersWithResiduals, self).__init__()
        self.shared_layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_shared_layers)
        ])

    def forward(self, x):
        for shared_layer in self.shared_layers:
            x = x + shared_layer(x)
        return x

class SharedAttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(SharedAttentionLayer, self).__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4)

    def forward(self, x):
        attended, _ = self.attention(x, x, x)
        return attended

class FeedbackLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(FeedbackLayer, self).__init__()
        self.feedback_layer = nn.Linear(input_size, hidden_size)

    def forward(self, x, feedback):
        feedback_embed = self.feedback_layer(feedback)
        combined = x + feedback_embed
        return combined

class MusicGenerator(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_features, output_size):
        super(MusicGenerator, self).__init__()
        self.feature_development = FeatureDevelopment(input_size, hidden_size)
        self.shared_lstm = nn.LSTM(hidden_size * num_features, hidden_size, num_layers, batch_first=True)
        self.temporal_attention = TemporalAttention(hidden_size * 2)
        self.shared_layers_with_residuals = SharedLayersWithResiduals(hidden_size * 2, num_layers=3,
                                                                      num_shared_layers=2)
        self.shared_attention_layer = SharedAttentionLayer(hidden_size * 2)
        self.feedback_layer = FeedbackLayer(hidden_size * 2, hidden_size * 2)

        # Capas de salida para cada característica
        self.output_layers = nn.ModuleList([
            nn.Linear(hidden_size * 2, output_size) for _ in range(num_features)
        ])

    def forward(self, start, duration, pitch, velocity, instrument, feedback=None):
        start_embed, duration_embed, pitch_embed, velocity_embed, instrument_embed = self.feature_development(start,
                                                                                                              duration,
                                                                                                              pitch,
                                                                                                              velocity,
                                                                                                              instrument)

        # Desarrollo de características individuales
        developed_features = start_embed, duration_embed, pitch_embed, velocity_embed, instrument_embed

        # Concatenar y pasar a través de shared_lstm
        combined_features = torch.cat(developed_features, dim=-1)
        shared_output, _ = self.shared_lstm(combined_features)
        attended_output = self.temporal_attention(shared_output)
        residual_output = self.shared_layers_with_residuals(attended_output)

        # Aplicar atención compartida si es necesario
        if self.shared_attention_layer is not None:
            shared_attention_output = self.shared_attention_layer(residual_output)
            residual_output = residual_output + shared_attention_output

        # Aplicar capa de retroalimentación si es necesario
        if feedback is not None:
            feedback_output = self.feedback_layer(residual_output, feedback)
            residual_output = residual_output + feedback_output

        # Capas de salida para cada característica
        outputs = [output_layer(residual_output) for output_layer in self.output_layers]
        return outputs

weights = torch.tensor([w_start, w_duration, w_pitch, w_velocity, w_instrument], dtype=torch.float32)
data = np.load("midi_matrices.npy") #Matriz de matrices de 5x10000 concatenadas

# Dividir los datos en características individuales (inicio, duración, pitch, velocidad, instrumento)
start_data = data[:, 0]
duration_data = data[:, 1]
pitch_data = data[:, 2]
velocity_data = data[:, 3]
instrument_data = data[:, 4]

# Convertir los datos a tensores de PyTorch
start_tensor = torch.tensor(start_data, dtype=torch.float32)
duration_tensor = torch.tensor(duration_data, dtype=torch.float32)
pitch_tensor = torch.tensor(pitch_data, dtype=torch.float32)
velocity_tensor = torch.tensor(velocity_data, dtype=torch.float32)
instrument_tensor = torch.tensor(instrument_data, dtype=torch.float32)

# Crear un conjunto de datos
dataset = TensorDataset(start_tensor, duration_tensor, pitch_tensor, velocity_tensor, instrument_tensor)

# Crear un DataLoader para manejar el acceso a los datos en lotes durante el entrenamiento
batch_size = 64
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
rnn = MusicGenerator(input_size, hidden_size, num_layers, num_features, output_size)
criterion = CustomLoss(weights)  # Define tu función de pérdida
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)

wandb.watch(rnn)

# Entrenamiento
num_epochs = 50
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        # Obtener las características del lote
        start, duration, pitch, velocity, instrument = batch

        # Pasar los datos por el modelo
        predictions = rnn(start, duration, pitch, velocity, instrument)

        # Calcular la pérdida utilizando la función de pérdida personalizada
        loss = criterion(predictions, batch)

        # Realizar la retropropagación y la actualización de parámetros
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")

future_seq_length = 10000
future_input = torch.tensor([-1, 1, 64, 64, 0], dtype=torch.float32).view(1, 1, input_size)
# Inicializar el tensor de entrada
future_output = []
hidden = None  # Reinicializar el tensor hidden

for _ in range(future_seq_length):
    output, hidden = rnn(future_input, hidden)
    future_input = output
    rounded_output = np.round(output.detach().numpy())  # Redondear todos los valores de salida
    future_output.append(rounded_output[0, -1, :])

generated_sequence = np.array(future_output)
print("Generated Sequence:")
print(generated_sequence)

output_midi = pm.PrettyMIDI()
instrument_programs = [pm.Instrument(program=0) for _ in range(128)]

for note_data in generated_sequence:
    start_time, duration, note, velocity, instrument = note_data

    velocity = np.round(velocity)
    note = np.round(note)

    velocity = max(0, min(127, velocity))
    note = max(0, min(127, note))

    note_event = pm.Note(
        velocity=int(velocity),
        pitch=int(note),
        start=float(start_time),
        end=float(start_time) + float(duration)
    )

    program = int(max(0, min(127, instrument)))

    instrument_programs[program].notes.append(note_event)

for instrument in instrument_programs:
    output_midi.instruments.append(instrument)

output_midi.write('generated_music.mid')
print("Generated MIDI saved as 'generated_music.mid'")

wandb.finish()