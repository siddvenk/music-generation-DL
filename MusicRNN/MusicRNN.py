from music21 import converter, instrument, note, chord
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
import sys

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]


def train_network():
    """ Train a Neural Network to generate music """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sys.setrecursionlimit(10000)
    notes = get_notes()

    # get amount of pitch names
    n_vocab = len(set(notes))

    network_input, network_output = prepare_sequences(notes, n_vocab)
    print('Size of input', network_input.shape)

    X_train = torch.tensor(network_input.astype(dtype = 'float32'))
    y_train = torch.tensor(network_output.astype(dtype = 'float32'))

    model = RNN(n_vocab).to(device)
    checkpoint = torch.load(sys.argv[1] + '.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    trainset = data_utils.TensorDataset(X_train, y_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=math.floor(network_input.shape[0] / 3), shuffle=True)

    train(trainloader, model, criterion, optimizer, device)


def get_notes():
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    notes = []

    for file in glob.glob(sys.argv[1] + "/*.mid"):
        midi = converter.parse(file)

        print("Parsing %s" % file)

        notes_to_parse = None

        try: # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse() 
        except: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    with open('data/notes_' + sys.argv[1] , 'w+b') as filepath:
        pickle.dump(notes, filepath)

    return notes

def prepare_sequences(notes, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 100

    # get all pitch names
    pitchnames = sorted(set(item for item in notes))

     # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with RNN layers
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input
    network_input = network_input / float(n_vocab)

    network_output = to_categorical(network_output, n_vocab)
    print('Num notes %s' % (n_vocab))

    return (network_input, network_output)

class RNN(nn.Module):
    def __init__(self, n_vocab):
        super().__init__()

        self.RNN1 = nn.RNN(1, 512, 2)
        self.Dropout = nn.Dropout(0.2)
        self.RNN2 = nn.RNN(512, 512, 2)
        self.RNN3 = nn.RNN(512, 512, 2)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, n_vocab)
        self.Softmax = nn.Softmax(dim=1)


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


def train(trainloader, net, criterion, optimizer, device):
    """ train the neural network """

    loss_graph = []
    start = time.time()
    for epoch in range(int(sys.argv[2])):  # loop over the dataset for x number of epochs
        running_loss = 0.0
        print('epoch: %d' % (epoch + 1))
        #For each batch run through model, backprop, and optimize weights
        for i, (data, notes) in enumerate(trainloader):
            data = data.to(device).float()
            notes = notes.to(device).long()
            #print(data.shape)

            optimizer.zero_grad()
            output = net(data)
            #print(output.shape)
            loss = criterion(output, notes)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            
            if i % 10 == 0:
                print('Iteration: ', i)
                loss_graph.append(loss.item())
                running_loss = 0.0
        end = time.time()
        print('elapsed time %.3f' % (end - start))
        start = time.time()
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
            }, sys.argv[1]+'.pth')
    
    # Plot learning curve
    fig1, ax1 = plt.subplots()
    ax1.plot(loss_graph, '--')
    ax1.set_title('Learning curve.')
    ax1.set_ylabel('L1 Loss')
    ax1.set_xlabel('Optimization steps.')

    print('Finished Training')




if __name__ == '__main__':
    train_network()


















