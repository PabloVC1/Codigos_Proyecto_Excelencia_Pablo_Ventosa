import random
import numpy as np
import pretty_midi as pm
import pygame
import base64

lista_duraciones = [1.5, 1.0, 0.75, 0.5, 0.375, 0.25, 0.1875, 0.125]
n = 2.0
num_compases = 4
escalas_circulo_quintas = {
    "C Mayor": ["C", "D", "E", "F", "G", "A", "B"],
    "G Mayor": ["G", "A", "B", "C", "D", "E", "F#"],
    "D Mayor": ["D", "E", "F#", "G", "A", "B", "C#"],
    "A Mayor": ["A", "B", "C#", "D", "E", "F#", "G#"],
    "E Mayor": ["E", "F#", "G#", "A", "B", "C#", "D#"],
    "B Mayor": ["B", "C#", "D#", "E", "F#", "G#", "A#"],
    "F Mayor": ["F", "G", "A", "Bb", "C", "D", "E"],
    "Bb Mayor": ["Bb", "C", "D", "Eb", "F", "G", "A"],
    "Eb Mayor": ["Eb", "F", "G", "Ab", "Bb", "C", "D"],
    "Ab Mayor": ["Ab", "Bb", "C", "Db", "Eb", "F", "G"],
    "Db Mayor": ["Db", "Eb", "F", "Gb", "Ab", "Bb", "C"]
}

tipos_acordes = {
    1: [0, 4, 7],
    2: [0, 3, 7],
    3: [0, 3, 7],
    4: [0, 4, 7],
    5: [0, 4, 7],
    6: [0, 3, 7],
    7: [0, 3, 6]
}

progresiones_mayor = {
    1: [1, 6, 3, 4],
    2: [1, 6, 4, 5],
    3: [2, 5, 1, 6],
    4: [1, 6, 2, 5],
    5: [1, 4, 6, 5],
    6: [1, 5, 2, 4],
    7: [1, 3, 4, 5],
    8: [6, 2, 5, 1],
    9: [1, 4, 2, 5]
}

progresion_original = [1, 5, 6, 4]
midi_data = pm.PrettyMIDI()
instrument = pm.Instrument(0)

def cambiar_disposicion(porgresiones_mayor):
    progresion_aleatoria = random.choice(list(progresiones_mayor.values()))
    indice_aleatorio = random.randint(0, 3)
    progresion_cambiada = progresion_aleatoria[indice_aleatorio:] + progresion_aleatoria[:indice_aleatorio]
    return progresion_cambiada, progresion_aleatoria

progresion_acordes, progresion_aleatoria = cambiar_disposicion(progresiones_mayor)

def generar_escala_aleatoria(diccionario_escalas, progresion_acordes):
    nombre_escala, notas_escala = random.choice(list(diccionario_escalas.items()))
    notas_progresion = [notas_escala[chord - 1] for chord in progresion_acordes]

    notas_acorde = [tipos_acordes[chord] for chord in progresion_acordes]

    progresion_final = dict(zip(notas_progresion, notas_acorde))

    return nombre_escala, progresion_final

def notas_piano_a_midi():
    notas_midi = {}
    notas = ["C", "C#", "Db", "D", "D#", "Eb", "E", "F", "F#", "Gb", "G", "G#", "Ab", "A", "A#", "Bb", "B"]
    octavas_anteriores = 4 * 12
    for octava in range(2):
        for nota in notas:
            valor_midi = octava * 12 + notas.index(nota) + octavas_anteriores

            if nota in notas_midi:
                notas_midi[nota].append(valor_midi)
            else:
                notas_midi[nota] = [valor_midi]

    return notas_midi

notas_midi = {"C": 60, "C#": 61, "Db": 61, "D": 62, "D#": 63, "Eb": 63, "E": 64, "F": 65, "F#": 66, "Gb": 66, "G": 67, "G#": 68, "Ab": 68, "A": 69, "A#": 70, "Bb": 70, "B": 71}

def notas_de_escala_para_acorde(acorde, notas_midi_dict, notas_progresion):
    notas_acorde = notas_progresion[acorde]
    raiz = notas_midi_dict[acorde]
    notas_escala = []
    for semitono in notas_acorde:
            nota_midi = (raiz + semitono)
            print(nota_midi)
            notas_escala.append(nota_midi)

    return notas_escala

def get_notes(progresion_acordes):
    nombre_escala, notas_progresion = generar_escala_aleatoria(escalas_circulo_quintas, progresion_acordes)
    return notas_progresion, nombre_escala

def seleccionar_duraciones_exacto(duraciones_disponibles, n):
    duraciones_seleccionadas = []
    suma_actual = 0

    while suma_actual != n:
        print(suma_actual)
        duracion = random.choice(duraciones_disponibles)
        if suma_actual + duracion <= n:
            duraciones_seleccionadas.append(duracion)
            suma_actual += duracion
        elif suma_actual + duracion > n:
            duraciones_seleccionadas = []
            suma_actual = 0

    return duraciones_seleccionadas


def generate_matrix(progresion_acordes, notas_midi_dict):
    notas_progresion, nombre_escala = get_notes(progresion_acordes)

    start = 0

    for acorde in notas_progresion:
        notas_escala = notas_de_escala_para_acorde(acorde, notas_midi, notas_progresion)
        print(notas_escala)
        duraciones_elegidas = seleccionar_duraciones_exacto(lista_duraciones, n)
        notas_elegidas = []
        for _ in duraciones_elegidas:
            notas_elegidas.append(random.choice(notas_escala))
        for duration, pitch in zip(duraciones_elegidas, notas_elegidas):
            print(f'{duration}:{pitch}')
            end = start + duration
            note = pm.Note(velocity=100, pitch=pitch, start=start, end=end)
            instrument.notes.append(note)
            start = end
    crear_acordes_midi(notas_progresion)
    return nombre_escala, notas_progresion


def crear_acordes_midi(diccionario_acordes):
    tiempo_inicio = 0
    for acorde, intervalos in diccionario_acordes.items():
        for intervalo in intervalos:
            nota_acorde = pm.Note(velocity=100, pitch=notas_midi[acorde] + intervalo - 12, start=tiempo_inicio, end=tiempo_inicio + 2)
            instrument.notes.append(nota_acorde)

        tiempo_inicio += 2.0

    midi_data.instruments.append(instrument)

notas_progresion, nombre_escala =generate_matrix(progresion_acordes, notas_midi_dict=notas_midi)
midi_data.write('nombre_de_tu_archivo.mid')

print(f"Nombre de la escala: {nombre_escala}")
print(f"Nombre de la progresion: {progresion_aleatoria}")
print(f"Notas de la progresión de acordes: {notas_progresion}")
def play_music(music_file):
    clock = pygame.time.Clock()
    try:
        pygame.mixer.music.load(music_file)
    except pygame.error:
        return
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        clock.tick(30)

music_file = "nombre_de_tu_archivo.mid"

freq = 44100
bitsize = -16
channels = 2
buffer = 1024
pygame.mixer.init(freq, bitsize, channels, buffer)
pygame.mixer.music.set_volume(0.8)

try:
    play_music(music_file)
except KeyboardInterrupt:
    pygame.mixer.music.fadeout(1000)
    pygame.mixer.music.stop()
    raise SystemExit
