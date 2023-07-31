import os
import pretty_midi as pm
import numpy as np
import pickle
import matplotlib.pyplot as plt
import time

max_notes = 500
max_values = np.array([0, 0, 127])
resolution = 480
interval_1 = [0, 30]
interval_2 = [40, 87]
notes_per_song = []

def process_midi_files(root_folder, batch_size):
    midi_files = []
    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.mid') or file.endswith('.midi'):
                midi_files.append(os.path.join(root, file))

    num_files = len(midi_files)
    num_batches = (num_files // batch_size) + 1

    all_matrices = []
    all_Vinstruments = []
    all_Finstruments = []
    num_samples = 0

    for batch_index in range(num_batches):
        start_index = batch_index * batch_size
        end_index = (batch_index + 1) * batch_size

        batch_files = midi_files[start_index:end_index]
        batch_matrices = []

        for file_path in batch_files:
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

                # Process each keyboard instrument
                for instrument in keyboard_instruments:
                    cancion = []
                    for note in instrument.notes:
                        start = int(note.start * resolution)
                        duration = int(note.end - note.start) * resolution

                        # Limit the duration to a whole note (redonda)
                        duration = min(duration, resolution * 4)

                        pitch = int(note.pitch)
                        cancion.append([start, duration, pitch])

                    cancion = np.array(cancion)
                    num_rows = cancion.shape[0]
                    notes_per_song.append(num_rows)

                    if num_rows == 0:
                        print(f"No se encontraron notas en el archivo {file_path}")
                        Finstruments.append(instrument.program)
                    else:
                        # Divide into groups of 16 compases and adjust the first column of each group
                        time_signature_changes = midi_data.time_signature_changes
                        time_signature_map = {ts.time: ts.numerator for ts in time_signature_changes}

                        # Find the first time signature change before the first note
                        first_time_signature = 4  # Default to 4/4 time signature if not found
                        for ts_time, ts_numerator in sorted(time_signature_map.items()):
                            if ts_time <= cancion[0, 0] / resolution:
                                first_time_signature = ts_numerator

                        # Calculate the number of ticks for 4 compases
                        ticks_per_beat = midi_data.resolution
                        ticks_per_compas = ticks_per_beat * 4 * first_time_signature

                        # Create a list to store the groups of notes
                        grouped_notes = []

                        # Divide the notes into groups based on the ticks for 4 compases
                        current_group = []
                        current_group_ticks = 0
                        for note in cancion:
                            start_tick = note[0]
                            note_ticks = note[1]
                            if current_group_ticks + note_ticks <= ticks_per_compas:
                                current_group.append(note)
                                current_group_ticks += note_ticks
                            else:
                                grouped_notes.append(current_group)
                                current_group = [note]
                                current_group_ticks = note_ticks

                        # Add the last group
                        if current_group:
                            grouped_notes.append(current_group)

                        if len(grouped_notes) == 0:
                            print(f"{file_path}: Todos los grupos están vacíos.")
                        else:
                            for i, group_notes in enumerate(grouped_notes):

                                # Normalize the start times within the group
                                first_note_start = group_notes[0][0]  # Accedemos al primer elemento de la primera fila
                                for note in group_notes:
                                    note[0] = note[0] - first_note_start
                                print(f"Start times of notes in group {i}: {[note[0] for note in group_notes]}")

                                # Calculate the max values for this group and update max_values
                                for row in group_notes:
                                    for col_index, value in enumerate(row):
                                        if value > max_values[col_index]:
                                            max_values[col_index] = value

                                if num_rows <= max_notes:
                                    zeros = np.zeros((max_notes - num_rows, cancion.shape[1]), dtype=int)
                                    group_notes = np.vstack((group_notes, zeros))
                                    Vinstruments.append(instrument.program)
                                else:
                                    Finstruments.append(instrument.program)

                            print(f"{file_path}: {Vinstruments}//{Finstruments}")
                            batch_matrices.extend(grouped_notes)

            except Exception as e:
                print(f"Error al procesar el archivo {file_path}: {str(e)}")

        all_matrices.extend(batch_matrices)
        all_Vinstruments.extend(all_Vinstruments)
        all_Finstruments.extend(all_Finstruments)

        print(f"Lote {batch_index + 1}/{num_batches} procesado. Archivos procesados: {len(batch_files)}"
              f"{all_Finstruments}//{all_Vinstruments}")

    output_file = "datos_0.pickle"
    with open(output_file, "wb") as file:
        pickle.dump((all_matrices), file)

    print(f"Proceso completado. Matrices guardadas en {output_file}.")

    plt.figure(figsize=(10, 6))
    plt.hist(notes_per_song, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Número de notas por canción')
    plt.ylabel('Frecuencia')
    plt.title('Distribución del número de notas por canción')
    plt.grid(True)
    plt.show()

# Ejemplo de uso
root_folder = r"C:\Users\Pablo\Desktop\LAKH"
process_midi_files(root_folder, batch_size=1000)
print(max_values)
