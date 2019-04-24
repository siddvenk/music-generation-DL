from music21 import converter, instrument, note, chord, stream
import glob
import pickle
import numpy as np
import torch
import torch.utils.data as data_utils
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
import math
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import time
from scipy.stats import mode

def generate():
    """ Generate a piano midi file """
    #load the notes used to train the model
    with open('data/notes', 'rb') as filepath:
        notes = pickle.load(filepath)

    # Get all pitch names
    pitchnames = sorted(set(item for item in notes))
    # Get all pitch names
    n_vocab = len(set(notes))

    network_input, normalized_input = prepare_sequences(notes, pitchnames, n_vocab)
    
    model = RNN(n_vocab)
    checkpoint = torch.load('music.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    prediction_output = generate_notes(model, network_input, pitchnames, n_vocab)
    create_midi(prediction_output)

def prepare_sequences(notes, pitchnames, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    # map between notes and integers and back
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    sequence_length = 100
    network_input = []
    output = []
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    normalized_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input
    normalized_input = normalized_input / float(n_vocab)

    return (network_input, normalized_input)


class RNN(nn.Module):
    def __init__(self, n_vocab):
        super().__init__()

        self.RNN1 = nn.RNN(1, 512, 2)
        self.Dropout = nn.Dropout(0.2)
        self.RNN2 = nn.RNN(512, 512, 2)
        self.RNN3 = nn.RNN(512, 512, 2)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, n_vocab)
        self.Softmax = nn.Softmax(dim=-1)


        self.forward = nn.Sequential(
            nn.RNN(n_vocab, 512),
            nn.Dropout(0.2),
            nn.RNN(512, 512),
            nn.Dropout(0.2),
            nn.RNN(512, 512),
            nn.Linear(512, 256),
            nn.Dropout(0.1),
            nn.Linear(256, n_vocab),
            nn.Softmax()
            )
    def forward(self, x):
        h0 = torch.randn(2, 100, 512)
        y_hat, hn = self.RNN1(x, h0)
        y_hat = self.Dropout(y_hat)
        y_hat, hn = self.RNN2(y_hat, hn)
        y_hat = self.Dropout(y_hat)
        y_hat, hn = self.RNN3(y_hat, hn)
        y_hat = self.Dropout(y_hat)
        y_hat = self.fc1(y_hat)
        y_hat = self.Dropout(y_hat)
        y_hat = self.fc2(y_hat)
        y_hat = self.Softmax(y_hat)
        return y_hat


def generate_notes(model, network_input, pitchnames, n_vocab):
    """ Generate notes from the neural network based on a sequence of notes """
    # pick a random sequence from the input as a starting point for the prediction
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    start = np.random.randint(0, len(network_input)-1)

    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

    pattern = network_input[start]
    prediction_output = []

    # generate 500 notes
    for note_index in range(500):
        print('Generating note %d' % (note_index + 1))
        prediction_input = np.reshape(np.asarray(pattern), (1, len(pattern), 1))
        #print(prediction_input.shape)
        prediction_input = prediction_input / float(n_vocab)
        prediction_input = torch.tensor(prediction_input).float()
        prediction_input = prediction_input.to(device).float()
        

        prediction = model(prediction_input)
        prediction = prediction.view(-1, n_vocab)
        prediction = prediction.detach().numpy()
        #print(prediction.shape)
        index = np.argmax(prediction[-1])
        #print(index.shape)
        #index = mode(index, axis=None)[0][0]
        #prediction = prediction.view(-1, n_vocab)
        #print(prediction.shape)
        #index = int(prediction.max(1)[1][len(pattern) - 1])
        #print(index)
        #index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)

        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    return prediction_output

def create_midi(prediction_output):
    """ convert the output from the prediction to notes and create a midi file
        from the notes """
    offset = 0
    output_notes = []

    # create note and chord objects based on the values generated by the model
    for pattern in prediction_output:
        # pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # pattern is a note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # increase offset each iteration so that notes do not stack
        offset += 0.5

    midi_stream = stream.Stream(output_notes)

    midi_stream.write('midi', fp='test_output.mid')

if __name__ == '__main__':
    generate()
