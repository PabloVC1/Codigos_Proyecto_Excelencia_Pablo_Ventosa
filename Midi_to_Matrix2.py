import os
import pretty_midi
import numpy as np
import pickle


def process_midi_files(root_folder, batch_size=1000, max_notes=10000):
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

                if len(midi_data.instruments) > 0 and len(midi_data.instruments[0].notes) < max_notes:
                    matrix = np.zeros((max_notes, 5))

                    for i, note in enumerate(midi_data.instruments[0].notes):
                        try:
                            matrix[i, 0] = note.start
                            matrix[i, 1] = note.end
                            matrix[i, 2] = note.pitch
                            matrix[i, 3] = note.velocity
                            matrix[i, 4] = midi_data.instruments[0].program
                        except ValueError as ve:
                            print(f"Error al procesar la nota en el archivo {file_path}: {str(ve)}")
                            break

                    num_samples = num_samples + 1
                    batch_matrices.append(matrix)

            except Exception as e:
                print(f"Error al procesar el archivo {file_path}: {str(e)}")

        all_matrices.extend(batch_matrices)

        print(f"Lote {batch_index + 1}/{num_batches} procesado. Archivos procesados: {len(batch_files)}")

    # Guardar las matrices en un archivo pickle
    output_file = "nuevas_matrices2.pickle"
    print(len(all_matrices))
    print(num_samples)
    with open(output_file, "wb") as file:
        pickle.dump(all_matrices, file)

    print(f"Proceso completado. Matrices guardadas en {output_file}.")


# Ejemplo de uso
root_folder = r"C:\Users\Pablo\Desktop\clean_midi"
process_midi_files(root_folder, batch_size=1000, max_notes=10000)