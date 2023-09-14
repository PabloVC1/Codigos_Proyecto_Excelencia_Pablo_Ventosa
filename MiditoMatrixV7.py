import os
import pretty_midi as pm
import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Configurar las credenciales de la API de Spotify
client_credentials_manager = SpotifyClientCredentials(client_id='238761ba5daa461fa6c9d5c5ad00140e',
                                                      client_secret='85d76ae10ffc43bd87942528edcd522e')
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

max_notes = 10000
max_values = np.array([5306767, 5429390, 127, 127, 127])
resolution = 480
notes_per_song = []
artist_genres_dict = {}
desired_genre = "pop"


def get_artist_genres(artist_name):
    if artist_name in artist_genres_dict:
        return artist_genres_dict[artist_name]

    artist_info = sp.search(q='artist:' + artist_name, type='artist', limit=1)
    if artist_info['artists']['items']:
        artist = artist_info['artists']['items'][0]
        artist_genres = [genre.lower() for genre in artist['genres']]
        artist_genres_dict[artist_name] = artist_genres
        return artist_genres

    return []

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
            '''for row in song:
                for col_index, value in enumerate(row):
                    if value > max_values[col_index]:
                        max_values[col_index] = value'''
            song = song / max_values[:5]
            song = (song * 2) - 1
            notes_per_song.append(num_rows)
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
            artist_name = os.path.basename(os.path.dirname(file_path))  # Obtén el nombre del artista
            artist_genre = get_artist_genres(artist_name)

            if desired_genre in artist_genre:
                matrix = process_single_midi_file(file_path)
                if matrix is not None:
                    batch_matrices.append(matrix)

        if batch_matrices:
            batch_matrices = np.concatenate(batch_matrices, axis=0)
            all_matrices.append(batch_matrices)
            print(f"Lote {batch_index + 1}/{num_batches} procesado. Archivos procesados: {len(batch_matrices)}")

    all_matrices_tensor = np.concatenate(all_matrices, axis=0)

    output_data_file = "midi_matrices.npy"
    np.save(output_data_file, all_matrices_tensor)

    print(f"Proceso completado. Matrices guardadas en {output_data_file}.")


# Ejemplo de uso
root_folder = r"C:\Users\Pablo\Desktop\LAKH"
batch_size = 1000
process_midi_files(root_folder, batch_size, max_notes)
print(max_values)
