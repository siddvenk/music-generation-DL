import numpy as np
import pypianoroll
from pypianoroll import Multitrack, Track
import os

NUM_SONGS = 3
NUM_PHASES = 4
NUM_BARS = 4
NUM_TIMESTEPS = 48
NUM_TRACKS = 5
NUM_PITCHES = 84
# song_dirname = '/home/david/data/vgmusic/pokemon'
# song_dirname = '/home/david/data/gba_pianoroll/merge_5'
song_dirname = '/home/david/data/battle_pianorolls/merge_5'
out_filename = 'train_battle.npz'

nonzero = []
cut = NUM_PHASES * NUM_BARS * NUM_TIMESTEPS

# for song_id, song_basename in enumerate(os.listdir(song_dirname)):
#     mt = Multitrack(os.path.join(song_dirname, song_basename))
#     if song_id < NUM_SONGS and mt.get_maximal_length() >= cut:
#         mt.binarize()
#         songphase_nonzero = []
#         for track_id,track in enumerate(mt.tracks):
#             if track_id not in mt.get_empty_tracks():
#                 roll = track.pianoroll[:cut, 36:120]
#                 phases = np.vsplit(roll, NUM_PHASES)
#                 for phase_id,phase in enumerate(phases):
#                     bars = np.vsplit(phase, NUM_BARS)
#                     for bar_id,bar in enumerate(bars):
#                         timestep_i, note_i = np.nonzero(bar)
#                         songphase_i = np.repeat(song_id*NUM_PHASES+phase_id, timestep_i.size)
#                         bar_i = np.repeat(bar_id, timestep_i.size)
#                         track_i = np.repeat(track_id, timestep_i.size)
#                         indexes = np.vstack([
#                             songphase_i,
#                             bar_i,
#                             timestep_i,
#                             note_i,
#                             track_i
#                         ])
#                         songphase_nonzero.append(indexes)
#         songphase_nonzero = np.hstack(songphase_nonzero)
#         nonzero.append(songphase_nonzero)
# nonzero = np.hstack(nonzero)
# shape = [NUM_SONGS*NUM_PHASES, NUM_BARS, NUM_TIMESTEPS, NUM_PITCHES, NUM_TRACKS]
# np.savez(out_filename, nonzero=nonzero, shape=shape)
# print(nonzero)
# print(shape)

loaded = np.load(out_filename)
loaded_nonzero = loaded['nonzero']
loaded_shape = loaded['shape']
x = np.zeros(loaded_shape)
x[[e for e in loaded_nonzero]] = 1
