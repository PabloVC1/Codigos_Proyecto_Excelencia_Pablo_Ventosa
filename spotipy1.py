import os
import re
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

client_credentials_manager = SpotifyClientCredentials(client_id=,
                                                      client_secret=)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

main_folder = r'C:\Users\Pablo\Desktop\clean_midi'

genres_counts = {}
for artist_folder in os.listdir(main_folder):
    artist_path = os.path.join(main_folder, artist_folder)
    for midi_file in os.listdir(artist_path):
        if midi_file.endswith('.mid'):
            song_name = midi_file[:-4]

            song_name = re.sub('\.\d+$', '', song_name)

            search_results = sp.search(q=song_name, type='track', limit=1)

            if search_results['tracks']['total'] > 0:
                track_id = search_results['tracks']['items'][0]['id']
                track_data = sp.track(track_id)

                artist_ids = []
                for artist in track_data["artists"]:
                    artist_ids.append(artist["id"])
                artists_data = sp.artists(artist_ids)

                genres = []
                for artist in artists_data["artists"]:
                    genres += artist["genres"]
                genres = set(genres)

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
