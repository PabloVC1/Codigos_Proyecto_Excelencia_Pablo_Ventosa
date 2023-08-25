import os
import pretty_midi as pm
import numpy as np
import pickle
import matplotlib.pyplot as plt

max_notes = 200
max_values = np.array([5306767, 561728, 127])
resolution = 480
interval_1 = [0, 30]
interval_2 = [40, 87]
notes_per_song = []


def pad_and_stack_matrices(matrices, max_notes):
    padded_matrices = []

    for matrix in matrices:
        num_rows = matrix.shape[0]
        if num_rows < max_notes:
            zeros = np.zeros((max_notes - num_rows, matrix.shape[1]), dtype=int)
            padded_matrix = np.vstack((matrix, zeros))
            padded_matrices.append(padded_matrix)
        else:
            padded_matrices.append(matrix)

    stacked_matrices = np.stack(padded_matrices)
    return stacked_matrices

def process_single_midi_file(file_path):
    try:
        midi_data = pm.PrettyMIDI(file_path)
        keyboard_instruments = []
        Vinstruments = []
        Finstruments = []

        for instrument in midi_data.instruments:
            if (interval_1[0] <= instrument.program <= interval_1[1]) or (
                    interval_2[0] <= instrument.program <= interval_2[1]):
                keyboard_instruments.append(instrument)

            else:
                Finstruments.append(instrument.program)

        if not keyboard_instruments:
            print(f"Los instrumentos de {file_path} no se encuentran en el rango ({interval_1}, {interval_2})")

        song = []
        # Process each keyboard instrument
        for instrument in keyboard_instruments:
            cancion = []
            for note in instrument.notes:
                start = int(note.start * resolution)
                duration = int((note.end - note.start) * resolution)
                pitch = int(note.pitch)

                cancion.append([start, duration, pitch])

            cancion = np.array(cancion)
            num_rows = cancion.shape[0]

            for row in cancion:
                for col_index, value in enumerate(row):
                    if value > max_values[col_index]:
                        max_values[col_index] = value

            if num_rows <= max_notes:
                notes_per_song.append(num_rows)
                Vinstruments.append(instrument.program)
                '''
                cancion = cancion / max_values[:3]
                cancion = (cancion * 2) - 1
                '''
                zeros = np.zeros((max_notes - num_rows, cancion.shape[1]), dtype=int)
                extended_matrix = np.vstack((cancion, zeros))
                song.append(extended_matrix)
            else:
                Finstruments.append(instrument.program)

        print(f"{file_path} {np.shape(song)}: {Vinstruments}//{Finstruments}")
        return song


    except Exception as e:
        print(f"Error al procesar el archivo {file_path}: {str(e)}")
        return None

def process_midi_files(root_folder, batch_size, max_notes):
    # Obtener la lista de archivos MIDI en la carpeta principal y las subcarpetas
    midi_files = []
    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.mid') or file.endswith('.midi'):
                midi_files.append(os.path.join(root, file))

    num_files = len(midi_files)
    num_batches = (num_files // batch_size) + 1

    all_matrices = []
    num_samples = 0

    for batch_index in range(num_batches):
        start_index = batch_index * batch_size
        end_index = (batch_index + 1) * batch_size

        batch_files = midi_files[start_index:end_index]
        batch_matrices = []

        for file_path in batch_files:
            batch_matrix = process_single_midi_file(file_path)
            if batch_matrix is not None:
                batch_matrices.extend([matrix for matrix in batch_matrix if matrix.shape[0] > 0])

        shapes = [matrix.shape for matrices in batch_matrices for matrix in matrices]
        if len(set(shapes)) != 1:
            print(f"Formas de las matrices en el lote {batch_index + 1} no son iguales:", shapes)

        stacked_batch = np.stack(batch_matrices)
        all_matrices.extend(stacked_batch)

        print(f"Lote {batch_index + 1}/{num_batches} procesado. Archivos procesados: {len(batch_files)}")

    padded_stacked_matrices = np.stack(all_matrices)

    # Guardar las matrices en un archivo pickle
    output_file = "nuevas_matrices.pickle"
    print(len(all_matrices))
    print(num_samples)
    with open(output_file, "wb") as file:
        pickle.dump(padded_stacked_matrices, file)

    print(padded_stacked_matrices)
    print(f"Proceso completado. Matrices guardadas en {output_file}.")


# Ejemplo de uso
root_folder = r"C:\Users\Pablo\Desktop\clean_midi"
process_midi_files(root_folder, batch_size=1000, max_notes=max_notes)
print(max_values)
