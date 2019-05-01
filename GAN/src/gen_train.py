import numpy as np
from pypianoroll import Track, Multitrack
import matplotlib.pyplot as plt 

# in_filename = '~/data/pokemon/midi/Pokemon RubySapphireEmerald - Wild Pokemon Battle.mid'
# out_filename = '~/data/musicGeneration/wildbattle.npz'
# bar_start = 2

num_bars = 4
beat_resolution = 12
pitch_min = 36
pitch_max = 120
num_tracks = 5
piano_track_id = 1

def get_nonzero(in_filename, bar_start=0, bar_end=None):
        mt = Multitrack(in_filename)
        mt.binarize()

        # beat_shrink = mt.beat_resolution // beat_resolution
        # mt.tracks[0].pianoroll = mt.tracks[0].pianoroll[::beat_shrink]
        pianoroll = mt.tracks[0].pianoroll
        # pianoroll = np.hstack([
        #         pianoroll[0::6],
        #         pianoroll[1::6],
        #         pianoroll[2::6]]).reshape(-1, pianoroll.shape[1])
        # pianoroll = pianoroll[::3] # 32th note as timestep
        # pianoroll = pianoroll[::beat_shrink] # 16th note as timestep
        # pianoroll = np.repeat(pianoroll, 3, axis=0)  # 48th note as timestep (12 timesteps per beat)
        pianoroll = np.sum([
                np.repeat(pianoroll[0::6], 3, axis=0),
                np.repeat(pianoroll[1::6], 3, axis=0),
                np.repeat(pianoroll[2::6], 3, axis=0)], axis=0) > 0
        mt.tracks[0].pianoroll = pianoroll

        mt.beat_resolution = beat_resolution
        # mt.write('data/test.mid')

        num_timesteps = num_bars * mt.beat_resolution
        time_start = num_timesteps * bar_start
        if bar_end is None:
                time_end = mt.get_maximal_length()
        else:
                time_end = num_timesteps * bar_end
        phrase_len = num_bars * num_timesteps
        phrase_endpoints = np.arange(time_start, time_end, phrase_len)
        phrases = np.vsplit(mt.tracks[0].pianoroll, phrase_endpoints)
        phrases = phrases[1:]
        num_phrases = len(phrases) - 1
        num_pitches = pitch_max - pitch_min
        A = np.zeros([num_phrases, num_bars, num_timesteps, num_pitches, num_tracks])
        for phrase_id, phrase in enumerate(phrases):
                if phrase_id < num_phrases:
                        mt.tracks[0].pianoroll = phrase
                        # phrase_filename = 'data/test_{}.mid'.format(phrase_id)
                        # mt.write(phrase_filename)
                        phrase = phrase[:, pitch_min:pitch_max]
                        phrase = phrase.reshape(num_bars, num_timesteps, num_pitches)
                        A[phrase_id, :, :, :, piano_track_id] = phrase
        return A


def load_nonzero(in_filename, out_filename):
        loaded = np.load(in_filename)
        loaded_nonzero = loaded['nonzero']
        loaded_shape = loaded['shape']
        B = np.zeros(loaded_shape)
        B[[e for e in loaded_nonzero]] = 1
        piano = B[:,:,:,:,piano_track_id]
        piano = piano.reshape(-1, piano.shape[-1])
        pad_lo = np.zeros([piano.shape[0], pitch_min])
        pad_hi = np.zeros([piano.shape[0], 128 - pitch_max])
        piano = np.hstack([pad_lo, piano, pad_hi])
        piano_track = Track(pianoroll=piano, program=1, is_drum=False,
                name='piano')
        song = Multitrack(tracks=[piano_track], tempo=190.0, beat_resolution=beat_resolution)
        song.binarize()
        song.write(out_filename)



out_npz_filename = '/home/david/data/musicGeneration/pokemonbattle.npz'
out_mid_filename = '/home/david/data/musicGeneration/pokemonbattle.mid'

A_list = []

in_filename = '/home/david/data/pokemon/midi/Pokemon RubySapphireEmerald - Wild Pokemon Battle.mid'
bar_start = 2
bar_end = None
B = get_nonzero(in_filename, bar_start, bar_end)
A_list.append(B)

in_filename = '/home/david/data/pokemon/midi/Pokemon RubySapphireEmerald - Rival Battle.mid'
bar_start = 2
bar_end = 34
B = get_nonzero(in_filename, bar_start, bar_end)
A_list.append(B)

in_filename = '/home/david/data/pokemon/midi/Pokemon RubySapphireEmerald - Rival Battle.mid'
bar_start = 37
bar_end = 77
B = get_nonzero(in_filename, bar_start, bar_end)
A_list.append(B)

in_filename = '/home/david/data/pokemon/midi/Pokemon RubySapphireEmerald - Rival Battle.mid'
bar_start = 80
bar_end = 96
B = get_nonzero(in_filename, bar_start, bar_end)
A_list.append(B)

in_filename = '/home/david/data/pokemon/midi/Pokemon RubySapphireEmerald - Elite 4 Battle.mid'
bar_start = 19
bar_end = None
B = get_nonzero(in_filename, bar_start, bar_end)
A_list.append(B)

in_filename = '/home/david/data/pokemon/midi/Pokemon RubySapphireEmerald - Elite 4 Battle.mid'
bar_start = 2
bar_end = 18
B = get_nonzero(in_filename, bar_start, bar_end)
A_list.append(B)

in_filename = '/home/david/data/pokemon/midi/Pokemon RubySapphireEmerald - Champion Battle.mid'
bar_start = 2
bar_end = 26
B = get_nonzero(in_filename, bar_start, bar_end)
A_list.append(B)

in_filename = '/home/david/data/pokemon/midi/Pokemon RubySapphireEmerald - Champion Battle.mid'
bar_start = 2
bar_end = None
B = get_nonzero(in_filename, bar_start, bar_end)
A_list.append(B)


A = np.concatenate(A_list, axis=0)
nonzero = np.nonzero(A)
shape = A.shape
np.savez(out_npz_filename, nonzero=nonzero, shape=shape)
load_nonzero(out_npz_filename, out_mid_filename)