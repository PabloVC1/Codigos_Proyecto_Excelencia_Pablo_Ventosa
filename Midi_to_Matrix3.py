from itertools import chain

import mido
import numpy as np
import glob

SAMPLES_PER_MEASURE = 8
MEASURES_IN_OUTPUT = 1250
TOTAL_OUTPUT_TIME_LENGTH = SAMPLES_PER_MEASURE * MEASURES_IN_OUTPUT

def files_in_folders(carpeta_principal):
    archivos_midi = glob.glob(carpeta_principal + '/**/*.mid', recursive=True)

    for archivo_midi in archivos_midi:
        there_and_back_again(archivo_midi)

def detect_time_signature(mid):
    has_time_sig = False
    ticks_per_beat = mid.ticks_per_beat
    ticks_per_measure = 4 * ticks_per_beat
    flag_warning = False
    for i, track in enumerate(mid.tracks):
        for msg in track:
            if msg.type == 'time_signature':
                new_tpm = msg.numerator * ticks_per_beat * 4 / msg.denominator
                if has_time_sig and new_tpm != ticks_per_measure:
                    flag_warning = True
                ticks_per_measure = new_tpm
                has_time_sig = True
    if flag_warning:
        raise ValueError('multiple distinct time signatures')
    return ticks_per_measure


def numpy_from_midi(midi):
    ticks_per_measure = detect_time_signature(midi)
    note_start_times = {}
    note_start_velocities = {}
    channels = {}
    finished_channels = []
    absolute_time = 0
    samples_per_tick = SAMPLES_PER_MEASURE / ticks_per_measure

    def curr_array_time():
        return int(absolute_time * samples_per_tick)

    def finish_note(note, stop_velocity, channel_num):
        start_velocity = note_start_velocities[channel_num][note]
        start_time = note_start_times[channel_num][note]
        stop_time = curr_array_time()
        assert start_velocity <= 1 and stop_velocity <= 1
        if stop_time - start_time == 0:
            stop_time += 1
        on_strip = np.linspace(
            start_velocity,
            stop_velocity,
            num=stop_time - start_time
        )
        channels[channel_num].array[note][start_time:stop_time] = on_strip
        note_start_times[channel_num][note] = -1  # mark note as finished

    def create_channel(channel_num, program):
        channels[channel_num] = ChannelArray(msg.channel, program=program)
        note_start_times[channel_num] = np.full((128,), -1)
        note_start_velocities[channel_num] = np.zeros((128,), dtype=np.float32)

    for track in midi.tracks:
        pass
        for msg in track:
            if msg.type == 'control_change':
                if msg.channel not in channels:  # create new channel
                    channels[msg.channel] = ChannelArray(msg.channel, program=None)
                    note_start_times[msg.channel] = np.full((128,), -1)
                    note_start_velocities[msg.channel] = np.zeros((128,), dtype=np.float32)
                if msg.control == 0:
                    absolute_time = 0
                    continue
            elif msg.type == 'end_of_track':
                absolute_time = 0
                continue
            elif msg.type == 'marker':
                continue
            absolute_time += msg.time

            if msg.type == 'program_change':
                if msg.channel in channels:
                    if channels[msg.channel].program is None:
                        channels[msg.channel].program = msg.program
                    else:
                        finished_channels.append(channels[msg.channel])  # archive channel
                        create_channel(msg.channel, msg.program)
                else:
                    create_channel(msg.channel, msg.program)
            elif msg.type == 'note_on':
                if curr_array_time() >= TOTAL_OUTPUT_TIME_LENGTH:
                    continue
                if msg.channel not in channels:
                    create_channel(msg.channel, None)
                if msg.velocity == 0:  # same as note_off
                    if note_start_times[msg.channel][msg.note] != -1:  # note started, lets finish
                        finish_note(msg.note, msg.velocity / 127, msg.channel)
                    continue

                if note_start_times[msg.channel][msg.note] != -1:
                    finish_note(msg.note, msg.velocity / (127 * 2), msg.channel)

                note_start_velocities[msg.channel][msg.note] = msg.velocity / 127
                note_start_times[msg.channel][msg.note] = curr_array_time()

            elif msg.type == 'note_off':
                if curr_array_time() >= TOTAL_OUTPUT_TIME_LENGTH:
                    continue
                if note_start_times[msg.channel][msg.note] == -1:
                    continue
                finish_note(msg.note, msg.velocity / 127, msg.channel)
    return chain(finished_channels, channels.values())


def shift_left_array(array):
    res = np.roll(array, -1)
    res[-1] = 0
    return res


def shift_right_array(array):
    res = np.roll(array, 1)
    res[0] = 0
    return res


def numpy_to_midi_track(chan_array, ticks_per_measure):
    ticks_per_sample = ticks_per_measure / SAMPLES_PER_MEASURE

    chan_num = chan_array.channel_num
    array = chan_array.array
    program = chan_array.program or 0

    shift_right = shift_right_array(array)
    note_starts = (shift_right == 0) & (array != 0)
    note_stops = (shift_left_array(array) == 0) & (array != 0)
    note_stops = shift_right_array(note_stops)
    array = array.T

    note_starts = note_starts.T
    note_stops = note_stops.T
    did_note_start = np.full((128,), False)
    delta_time = 0.0
    track = mido.MidiTrack()
    track.extend([
        mido.MetaMessage('track_name', name=chan_array.name),
        mido.Message('control_change', control=0,
                     channel=chan_num),
        mido.Message('program_change', channel=chan_num,
                     program=program)
    ])
    if chan_array.messages:
        track.extend(chan_array.messages)

    for vel_arr, is_starting_arr, is_stopping_arr in zip(array,
                                                         note_starts,
                                                         note_stops):
        for note_num, (vel, is_starting, is_stopping) in enumerate(zip(vel_arr,
                                                                       is_starting_arr,
                                                                       is_stopping_arr)):
            if is_starting:
                if did_note_start[note_num]:
                    track.append(mido.Message('note_off', note=note_num,
                                              velocity=int(127 * vel),
                                              time=int(delta_time), channel=chan_num))
                    delta_time -= int(delta_time)
                track.append(mido.Message('note_on', note=note_num,
                                          velocity=int(127 * vel),
                                          time=int(delta_time), channel=chan_num))
                did_note_start[note_num] = True
                delta_time -= int(delta_time)
            if is_stopping:
                if did_note_start[note_num]:
                    track.append(mido.Message('note_off', note=note_num,
                                              velocity=int(127 * vel),
                                              time=int(delta_time), channel=chan_num))
                    delta_time -= int(delta_time)
                    did_note_start[note_num] = False
                else:
                    pass

        delta_time += ticks_per_sample

    track.append(mido.MetaMessage('end_of_track'))
    return track


class ChannelArray:
    def __init__(self, channel_num, program=None, name='',
                 messages=None):
        self.program = program
        self.name = name
        self.channel_num = channel_num
        self.array = np.zeros((128, TOTAL_OUTPUT_TIME_LENGTH), dtype=np.float32)
        self.messages = messages

    def _arr_info(self):
        playing = (self.array > 0).T
        if not np.any(playing):
            return 'empty'
        start_ind = np.argmax(playing) // playing.shape[1]
        stop_ind = (len(playing.flatten()) - np.argmax(playing[::-1])) // playing.shape[1]
        return ('starting %4d stopping %4d'
                % (start_ind, stop_ind))

    def __repr__(self):
        return 'channelArray {}#{} with prog {:3d}. {}' \
            .format(self.name + ' ' if self.name else '',
                    self.channel_num,
                    self.program if self.program is not None else -1,
                    self._arr_info())


def map_to_one_channel(channels):
    """
    maps multiple channels to only one channels, intended for reduction the number of channels for an AI to process
    :param channels: an iterable of ChannelArrays
    :return: one ChannelArray
    """

    piano = ChannelArray(2, name='piano', program=4)
    for channel in channels:
        channel.program = channel.program or 0
        if not (channel.program == 0 or channel.program >= 111):
            piano.array += channel.array
    piano.array = np.minimum(piano.array, 1.0)
    return piano


def write_msgs_to_file(filename, midi):
    f = open(filename, 'w')
    for track in midi.tracks:
        f.write(str(track) + '\n')
        for msg in track:
            f.write(' ' + str(msg) + '\n')


def there_and_back_again(filename):
    print('testing on', filename)
    try:
        src_midi = mido.MidiFile(filename)
    except EOFError:
        print('error with midi file, exiting')
        return
    ticks_per_measure = detect_time_signature(src_midi)
    print(src_midi)
    print(src_midi.ticks_per_beat)
    channels = numpy_from_midi(src_midi)
    channel = map_to_one_channel(channels)
    midi = mido.MidiFile(ticks_per_beat=src_midi.ticks_per_beat)

    back_track = numpy_to_midi_track(channel, ticks_per_measure)
    midi.tracks.append(back_track)
    midi.save('result.mid')
    write_msgs_to_file(filename.split('.')[0] + '_result_messages.txt', midi)
    write_msgs_to_file(filename.split('.')[0] + '_src_messages.txt', src_midi)


carpeta_principal = r'C:\Users\Pablo\Desktop\clean_midi'

files_in_folders(carpeta_principal)