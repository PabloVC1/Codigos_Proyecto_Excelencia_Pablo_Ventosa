import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Establece tus credenciales de Spotify
client_id = '238761ba5daa461fa6c9d5c5ad00140e'
client_secret = '85d76ae10ffc43bd87942528edcd522e'
redirect_uri = 'https://example.com/callback'  # Puede ser cualquier URL válida, se usa para autenticación

# Ruta de la carpeta principal
carpeta_principal = r'C:\Users\Pablo\Desktop\clean_midi'

archivo_salida = r'C:\Users\Pablo\Desktop\generos.txt'

# Configuración de autenticación de Spotipy
sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(client_id, client_secret))

# Diccionario para almacenar los géneros y la cantidad de canciones
generos_canciones = {}

# Recorre las carpetas y archivos
for artista in os.listdir(carpeta_principal):
    carpeta_artista = os.path.join(carpeta_principal, artista)
    if os.path.isdir(carpeta_artista):
        # Verifica si la carpeta tiene más de 3 canciones
        cantidad_canciones = len(os.listdir(carpeta_artista))
        nombre_artista = os.path.splitext(artista)[0]

        resultados = sp.search(q='artist:' + nombre_artista, type='artist', limit=1)
        if resultados['artists']['items']:
            artist_info = resultados['artists']['items'][0]
            for genero in artist_info['genres']:
                print(f'{artista}, genero: {genero}')
                generos_canciones[genero] = generos_canciones.get(genero, 0) + cantidad_canciones

# Filtra los géneros con más de 3 canciones
generos_canciones_filtrados = {genero: cantidad_canciones for genero, cantidad_canciones in generos_canciones.items() if cantidad_canciones > 3}

# Imprime los géneros y la cantidad de canciones por género
for genero, cantidad_canciones in generos_canciones_filtrados.items():
    print(f'{genero}: {cantidad_canciones} canciones')

# Imprime el total de géneros diferentes
total_generos = len(generos_canciones_filtrados)
print(f'Total de géneros diferentes con más de 3 canciones: {total_generos}')

with open(archivo_salida, 'w') as file:
    for genero, cantidad_canciones in generos_canciones_filtrados.items():
        file.write(f'{genero}: {cantidad_canciones} canciones\n')

# Imprime un mensaje de confirmación
print(f'El conteo se ha guardado en el archivo: {archivo_salida}')
