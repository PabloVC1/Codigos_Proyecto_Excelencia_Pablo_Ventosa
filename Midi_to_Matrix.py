import numpy as np
import sys
import os
import pretty_midi
import pickle
# Ruta de la carpeta principal que contiene las subcarpetas con los archivos MIDI
midi_folder = r'C:\Users\Pablo\Desktop\clean_midi'
# Inicializa una lista para almacenar las matrices resultantes de cada archivo MIDI
midi_arrays = []
failed_files = []

# Ruta de la carpeta donde se guardarán los archivos de caché
cache_folder = r'C:\Users\Pablo\Desktop\Falso_caché'

# Si la carpeta de caché no existe, créela
if not os.path.exists(cache_folder):
    os.makedirs(cache_folder)

# Ruta del archivo de caché donde se guardarán las matrices resultantes
cache_file = os.path.join(cache_folder, 'cache.pickle')

# Crea el archivo de caché si no existe
if not os.path.exists(cache_file):
    with open(cache_file, 'wb') as f:
        pickle.dump([], f)

# Itera a través de las subcarpetas en la carpeta principal
for root, dirs, files in os.walk(midi_folder):
    for file in files:
        if file.endswith('.mid'):
            try:
                # Carga el archivo MIDI
                midi_data = pretty_midi.PrettyMIDI(os.path.join(root, file))

                # Inicializa una matriz para almacenar los eventos del archivo MIDI
                midi_array = []

                # Recorre todas las notas de todos los instrumentos en el archivo MIDI y los agrega a la matriz
                for instrument in midi_data.instruments:
                    for note in instrument.notes:
                        # Cada fila de la matriz representa una nota MIDI y contiene cinco valores:
                        # el inicio del evento, el final del evento, el número de nota MIDI, la velocidad y el número de instrumento
                        midi_array.append([int(note.start), int(note.end), int(note.pitch), int(note.velocity), int(instrument.program)])

                # Ordena la matriz por tiempo de inicio del evento
                midi_array.sort(key=lambda x: x[0])
                print(f'{file} completado')
                midi_array = np.matrix(np.array(midi_array))
                #Añadir: 'np.set_printoptions(threshold=sys.maxsize) ; print(midi_array)' en caso de querer ver las matrices individuales

                # Agrega la matriz resultante a la lista de matrices
                midi_arrays.append(midi_array)
            except Exception as e:
                print(f'Error al procesar el archivo {file}: {e}')
                failed_files.append(file)
                continue

print(failed_files)
with open(cache_file, 'wb') as f:
    pickle.dump(midi_arrays, f)