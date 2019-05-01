#!/usr/bin/env python3

import os
import sys
import mido
import music21
import numpy as np
from fractions import Fraction

fname = sys.argv[1]
new_path = '%s.npy' % os.path.basename(fname)
if os.path.isfile(new_path):
    print('%s already exists' % fname)
    sys.exit(1)

#import mido
#file = mido.MidiFile(fname)

#keywords = ['drum', 'tromme', 'percussion', 'snare', 'sticks', 'kit']

#for track in file.tracks:
#    if any(word in track.name.lower() for word in keywords):
#        file.tracks.remove(track)

for t, track in enumerate(file.tracks):
    newtrack = mido.midifiles.tracks.MidiTrack()
    has_time = False
    has_tempo = False
    for msg in track:
        m = msg.dict()
        if m['type'] == 'note_on' or m['type'] == 'note_off':
            m['channel'] = 0
            #m['time'] = 0
            newtrack.append(mido.messages.Message.from_dict(m))
        #elif m['type'] == 'key_signature' or m['type'] == 'time_signature':
        elif m['type'] == 'time_signature' and not has_time:
            has_time = True
            newtrack.append(msg)
        elif m['type'] == 'set_tempo' and not has_tempo:
            has_tempo = True
            newtrack.append(msg)
        elif m['type'] == 'end_of_track':
            newtrack.append(msg)
        else:
            pass
    
    file.tracks[t] = newtrack

fname = os.path.basename(fname)
file.save(fname)

samples_per_measure = 96
quarters_per_measure = 24

lowest_pitch = 12
highest_pitch = 107
total_notes = highest_pitch - lowest_pitch + 1

def pitch_in_range(pitch):
    if pitch >= lowest_pitch or pitch <= highest_pitch:
        return True
    else:
        return False

def sampled_time(offset):
    return int(Fraction(offset) * quarters_per_measure)

def quarter_time(time):
    return time / quarters_per_measure

def pitch_index(pitch):
    return int(pitch - lowest_pitch)

def midi_index(index):
    return int(index + lowest_pitch)

def midi_to_samples(filename, end_measure=16, start_measure=1):
    X = np.zeros((end_measure, samples_per_measure, total_notes))
    piece = music21.converter.parse(filename)
    for m in range(start_measure, end_measure+1):
        measure = piece.measures(m, m)
        for notes in measure.flat.notes:
            note_start = sampled_time(notes.offset)
            note_length = sampled_time(notes.quarterLength)

            pitches = []
            if type(notes) == music21.note.Note:
                pitches = [notes.pitch.midi]
            elif type(notes) == music21.chord.Chord:
                pitches = [p.midi for p in notes.pitches]

            assert all(pitch_in_range(p) for p in pitches)
            pitches = map(pitch_index, pitches)

            current_measure = m-1
            while note_length > 0:
                length = min(note_length, samples_per_measure)
                note_end = note_start + length

                for p in pitches:
                    X[current_measure, note_start:note_end, p] = 1
                    if note_start > 0:
                        if X[current_measure, note_start-1, p] == 1:
                            X[current_measure, note_start, p] = 2
                    elif note_start == 0:
                        if m > 0 and X[current_measure-1, samples_per_measure-1, p] == 1:
                            X[current_measure, note_start, p] = 2

                note_start = 0
                note_length -= length
                current_measure += 1
    return X, piece

X, _ = midi_to_samples(fname)
np.save(new_path, X)
print(fname)

fname = sys.argv[1]
f = mido.MidiFile(fname)

f.save(os.path.basename(fname))

