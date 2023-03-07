import os
import re
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Configurar las credenciales de la API de Spotify
client_credentials_manager = SpotifyClientCredentials(client_id='238761ba5daa461fa6c9d5c5ad00140e',
                                                      client_secret='00af06e06e6e4dc4b63bdb37a2f35efa')
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Carpeta principal donde se encuentran las carpetas de artistas
main_folder = r'C:\Users\Pablo\Desktop\clean_midi'

genres_counts = {}
# Recorrer cada carpeta de artista
for artist_folder in os.listdir(main_folder):
    # Ruta a la carpeta actual de artista
    artist_path = os.path.join(main_folder, artist_folder)

    # Recorrer cada archivo MIDI en la carpeta del artista
    for midi_file in os.listdir(artist_path):
        # Verificar si el archivo es un archivo MIDI
        if midi_file.endswith('.mid'):
            # Obtener el nombre de la canción a partir del nombre del archivo
            song_name = midi_file[:-4]  # Remover la extensión del archivo

            # Eliminar el número al final del nombre de la canción, si existe
            song_name = re.sub('\.\d+$', '', song_name)

            # Buscar información de la canción en Spotify
            search_results = sp.search(q=song_name, type='track', limit=1)

            # Verificar si se encontró una canción
            if search_results['tracks']['total'] > 0:
                # Obtener el ID de la primera canción encontrada
                track_id = search_results['tracks']['items'][0]['id']
                track_data = sp.track(track_id)

                # Obtener nombre de los artistas a partir del track id
                artist_ids = []
                for artist in track_data["artists"]:
                    artist_ids.append(artist["id"])
                artists_data = sp.artists(artist_ids)

                # Obtener géneros a partir de los artistas
                genres = []
                for artist in artists_data["artists"]:
                    genres += artist["genres"]
                genres = set(genres)

                # Contar los géneros
                for genre in genres:
                    if genre in genres_counts:
                        genres_counts[genre] += 1
                    else:
                        genres_counts[genre] = 1

                print(f"Géneros de '{song_name}' encontrado: {genres}")

            else:
                print(f"No se encontró ninguna canción con el nombre '{song_name}'")

for genre, count in genres_counts.items():
    print(f"{genre}: {count} canciones")