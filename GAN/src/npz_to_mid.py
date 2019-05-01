import sys
from pypianoroll import Track, Multitrack 
import numpy as np

piano_only = True
num_beats = 4
num_bars = 4
num_tracks = 5
piano_track_id = 1

src_filename = sys.argv[1]
dst_filename = sys.argv[2]
tempo = int(sys.argv[3])

mt = Multitrack(src_filename)
beat_resolution = mt.beat_resolution

song_len = mt.get_maximal_length()
num_timesteps = beat_resolution * num_beats
phrase_len = num_timesteps * (num_bars + 1)
endpoints = np.arange(0, song_len, phrase_len)
endpoints = endpoints.reshape([-1, 1])
goodpoints = endpoints + np.arange(num_timesteps * num_bars).reshape([1, -1])
goodpoints = goodpoints.flatten()
for i in range(num_tracks):
    mt.tracks[i].pianoroll = mt.tracks[i].pianoroll[goodpoints]

if piano_only:
    piano = mt.tracks[piano_track_id].pianoroll
    piano = piano[:(12*4*12):3]
    piano = piano > 0
    piano_track = Track(pianoroll=piano, program=1, is_drum=False)
    song = Multitrack(tracks=[piano_track], tempo=tempo, beat_resolution=beat_resolution // 3)
else:
    song = mt
    song.tracks[1].program = 1

song.tempo = np.array([tempo])
song.write(dst_filename)