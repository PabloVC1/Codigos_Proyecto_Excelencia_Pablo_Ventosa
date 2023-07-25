import os
import pretty_midi
import numpy as np
import pickle

max_values = np.array([11055,1170,127,127,127])

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
            try:
                midi_data = pretty_midi.PrettyMIDI(file_path)
                cancion = []

                for instrument in midi_data.instruments:
                    for note in instrument.notes:
                        duracion = int(note.end - note.start)
                        if not duracion <= 0:
                            cancion.append(
                                [int(note.start),
                                int(duracion),
                                int(note.pitch),
                                int(note.velocity),
                                int(instrument.program)]
                            )

                """
                for row in cancion:
                    for col_index, value in enumerate(row):
                        if value > max_values[col_index]:
                            max_values[col_index] = value
                            """
                cancion = cancion / max_values[:5]
                cancion = np.array(cancion)
                num_rows = cancion.shape[0]
                if num_rows <= max_notes:
                    zeros = np.zeros((max_notes - num_rows, cancion.shape[1]), dtype=int)
                    cancion = np.vstack((cancion, zeros))
                    cancion = np.array(cancion)
                    batch_matrices.append(cancion)
                    print(f'CONGRATS: {file_path}!! {cancion}')

                else:
                    print(f'sry: {file_path}')

            except Exception as e:
                print(f"Error al procesar el archivo {file_path}: {str(e)}")

        all_matrices.extend(batch_matrices)

        print(f"Lote {batch_index + 1}/{num_batches} procesado. Archivos procesados: {len(batch_files)}")

    # Guardar las matrices en un archivo pickle
    output_file = "matrices10000.pickle"
    print(len(all_matrices))
    print(num_samples)
    with open(output_file, "wb") as file:
        pickle.dump(all_matrices, file)

    print(all_matrices)
    print(f"Proceso completado. Matrices guardadas en {output_file}.")


# Ejemplo de uso
root_folder = r"C:\Users\Pablo\Desktop\clean_midi"
process_midi_files(root_folder, batch_size=1000, max_notes=10000)
print(max_values)