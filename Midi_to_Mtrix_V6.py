import os
import pretty_midi as pm
import numpy as np

max_notes = 10000
max_values = np.array([5306767, 561728, 127])
resolution = 480
notes_per_song = []


def process_single_midi_file(file_path):
    try:
        midi_data = pm.PrettyMIDI(file_path)

        song = []
        # Process each keyboard instrument
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                # Cada fila de la matriz representa una nota MIDI y contiene cinco valores:
                # el inicio del evento, el final del evento, el número de nota MIDI, la velocidad y el número de instrumento
                song.append(
                    [int(note.start * resolution), int(note.end * resolution), int(note.pitch), int(note.velocity), int(instrument.program)])

        song = np.array(song)
        num_rows = song.shape[0]

        if num_rows < max_notes:
            notes_per_song.append(num_rows)
            zeros = np.zeros((max_notes - num_rows, song.shape[1]), dtype=int)
            song = np.vstack((song, zeros))
            print(f'{file_path} completado')
            return song

    except Exception as e:
        print(f"Error al procesar el archivo {file_path}: {str(e)}")
        return None

def process_midi_files(root_folder, batch_size, max_notes):
    midi_files = []
    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith('.mid') or file.lower().endswith('.midi'):
                midi_files.append(os.path.join(root, file))

    num_files = len(midi_files)
    num_batches = (num_files + batch_size - 1) // batch_size

    all_matrices = []

    for batch_index in range(num_batches):
        start_index = batch_index * batch_size
        end_index = min((batch_index + 1) * batch_size, num_files)

        batch_files = midi_files[start_index:end_index]
        batch_matrices = []

        for file_path in batch_files:
            matrix = process_single_midi_file(file_path)
            if matrix is not None:
                batch_matrices.append(matrix)

        if batch_matrices:
            batch_matrices = np.concatenate(batch_matrices, axis=0)
            all_matrices.append(batch_matrices)
            print(f"Lote {batch_index + 1}/{num_batches} procesado. Archivos procesados: {len(batch_files)}")

    all_matrices_tensor = np.concatenate(all_matrices, axis=0)

    output_data_file = "midi_matrices.npy"
    np.save(output_data_file, all_matrices_tensor)

    print(f"Proceso completado. Matrices guardadas en {output_data_file}.")


# Ejemplo de uso
root_folder = r"C:\Users\Pablo\Desktop\LAKH"
batch_size = 1000
process_midi_files(root_folder, batch_size, max_notes)
