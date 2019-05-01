import numpy as np
from pypianoroll import Multitrack

in_filename = 'battle_1300.npz'
time_begin = 240 * 2
time_end = 240 * 4
NUM_TRACKS = 5

x = Multitrack(in_filename)
for i in range(NUM_TRACKS):
    x.tracks[i].pianoroll = x.tracks[i].pianoroll[time_begin:time_end]

print(x.get_maximal_length())
# x.write('out.mid')
x.save('out.npz')