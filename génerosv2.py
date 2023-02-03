import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os
from collections import Counter

CLIENT_ID = "238761ba5daa461fa6c9d5c5ad00140e"
CLIENT_SECRET = "00af06e06e6e4dc4b63bdb37a2f35efa"
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET))

all_genres = []


def get_track_id(root_dir):
    for subdir, dirs, files in os.walk(root_dir):
        for dir in dirs:
            artist = dir
            for file in os.listdir(os.path.join(subdir, dir)):
                if file.endswith('.mid'):
                    artist_ids = []
                    artists_data = sp.artists(artist_ids)
                    genres = []
                    for artist in artists_data["artists"]:
                        genres += artist["genres"]
                        print(artist)
                    genres = set(genres)
                    all_genres.append(genre)





get_track_id(r'C:\Users\Pablo\Desktop\clean_midi\A-I')

genre_count = Counter(all_genres)
for genre, count in genre_count.items():
    print(f'{genre}: {count}')