import torch
import torch.nn as nn
import numpy as np
import pickle
import wandb
import pretty_midi

cache_file = r"C:\Users\Pablo\Documents\MusikIA\Numpy\midi_matrices.pickle"
with open(cache_file, 'rb') as f:
    datos = np.array(pickle.load(f))


wandb.login(key="26ab38e8f6e471ce6662ff95ea15c50993b6d4a1")
wandb.init(project='RNNv1')

def load_data(data, batch_size):
    dataset = torch.tensor(data).float()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

input_size = 5
hidden_size = 64
batch_size = 32
output_size = 5
seq_length = 200
num_layers = 1
learning_rate = 0.002
num_epochs = 15

data_loader = load_data(datos, batch_size=batch_size)

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)
        return out, hidden

rnn = SimpleRNN(input_size, hidden_size, output_size)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)

wandb.watch(rnn)

for epoch in range(num_epochs):
    for batch_data in data_loader:
        optimizer.zero_grad()
        hidden = None
        output, hidden = rnn(batch_data, hidden)
        loss = criterion(output, batch_data)
        loss.backward()
        optimizer.step()

        # Registro del loss con W&B
        wandb.log({"loss": loss.item()})

    if (epoch + 1) % 1 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

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

output_midi = pretty_midi.PrettyMIDI()
instrument_programs = [pretty_midi.Instrument(program=0) for _ in range(128)]

for note_data in generated_sequence:
    start_time, duration, note, velocity, instrument = note_data

    velocity = np.round(velocity)
    note = np.round(note)

    velocity = max(0, min(127, velocity))
    note = max(0, min(127, note))

    note_event = pretty_midi.Note(
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
