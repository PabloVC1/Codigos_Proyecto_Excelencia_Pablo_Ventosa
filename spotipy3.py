import os
import re
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Configurar las credenciales de la API de Spotify
client_credentials_manager = SpotifyClientCredentials(client_id='238761ba5daa461fa6c9d5c5ad00140e',
                                                      client_secret='85d76ae10ffc43bd87942528edcd522e')
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Carpeta principal donde se encuentran las carpetas de artistas
main_folder = r"C:\Users\Pablo\Desktop\LAKH"

# Géneros y sus índices en la matriz
genre_indices = {
    'rock': 0,
    'country': 1,
    'metal': 2,
    'pop': 3,
    'hip': 4,
    'hop': 4,
    'house': 5,
    'opera': 6,
    'blues': 7,
    'punk': 8,
    'soul': 9,
    'reggae': 10,
    'jazz': 11
}

# Diccionario para contar géneros
genres_counts = {genre: 0 for genre in genre_indices}

# Recorrer cada carpeta de artista
for artist_folder in os.listdir(main_folder):
    # Ruta a la carpeta actual de artista
    artist_path = os.path.join(main_folder, artist_folder)

    # Recorrer cada archivo MIDI en la carpeta del artista
    for midi_file in os.listdir(artist_path):
        # Verificar si el archivo es un archivo MIDI
        if midi_file.endswith('.mid'):
            # Obtener el nombre de la canción a partir del nombre del archivo
            artist_name = artist_folder
            genre_matrix = [[0] * 4 for _ in range(3)]

            artist_info = sp.search(q='artist:' + artist_name, type='artist', limit=1)
            if artist_info['artists']['items']:
                artist = artist_info['artists']['items'][0]
                artist_genres = [genre.lower() for genre in artist['genres']]

                for genre in artist_genres:
                    for key, index in genre_indices.items():
                        if key in genre:
                            if 0 <= index < 12:  # Asegurarse de que el índice esté dentro del rango de la matriz
                                genre_matrix[index // 4][index % 4] = 1
                                genres_counts[key] += 1

                print(f'{genre_matrix}: {artist_genres}')

for genre, count in genres_counts.items():
    print(f"{genre}: {count} canciones")
