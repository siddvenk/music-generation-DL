from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torch.utils.data as utils

import random
import glob
import os
import sys
import music21
import numpy as np
from fractions import Fraction

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
                    #print(current_measure, '\t', note_start, length, '\t', p)

                note_start = 0
                note_length -= length
                current_measure += 1
    return X, piece

def samples_to_music21(data, num_measures=16):
    song = music21.stream.Stream()

    measures, samples, pitches = data.shape
    on_pitches = {i:0 for i in range(pitches)}
    for m in range(num_measures):
        offset = m * samples_per_measure
        for s in range(samples):
            for p in range(pitches):
                pitch = data[m, s, p]
                if pitch != 1 and on_pitches[p] > 0:
                    note = music21.note.Note(midi_index(p))
                    song.append(note)
                    note.duration.quarterLength = quarter_time(on_pitches[p])
                    note.offset = quarter_time(offset + s - on_pitches[p])

                    if pitch == 2:
                        on_pitches[p] = 1
                    else:
                        on_pitches[p] = 0
                elif pitch == 1:
                    on_pitches[p] += 1

        for p in range(pitches):
            if on_pitches[p] > 0:
                if m+1 == measures or data[m+1, 0, p] == 0:
                    note = music21.note.Note(midi_index(p))
                    song.append(note)
                    note.duration.quarterLength = quarter_time(on_pitches[p])
                    note.offset = quarter_time(offset + samples_per_measure - on_pitches[p])
                    on_pitches[p] = 0
    return song

if not os.path.exists('data'):
    print('No data folder found!')
    sys.exit(1)

loaded_file_names = []
loaded_files = []
myLabels = []
data_files = glob.glob('data/**/*.npy', recursive=True)
for f in data_files:
    if '/piano/' in f or '/classical/' in f:
        label = 0
    elif '/battle/' in f:
        label = 1
    elif '/title/' in f:
        label = 2
    else:
        continue

    x = np.load(f)
    if np.any(x):
        loaded_files.append(x)
    elif X.shape != (16, 96, 96):
        continue
    else:
        os.remove(f)
        continue

    myLabels.append(label)
    loaded_file_names.append(f)

myData = list(zip(loaded_files, myLabels, loaded_file_names))
random.shuffle(myData)
loaded_files, myLabels, loaded_file_names = zip(*myData)
    
myX = np.array(loaded_files)
myLabels = np.array(myLabels)

part = int(len(myX) * 0.1)
dataTrain, dataTest = myX[part:], myX[:part]
labelsTrain, labelsTest = myLabels[part:], myLabels[:part]
filesTrain, filesTest = loaded_file_names[part:], loaded_file_names[:part]
        
with open('output/train_names.txt', 'w') as f:
    f.write('\n'.join(filesTrain))
        
with open('output/test_names.txt', 'w') as f:
    f.write('\n'.join(filesTest))

print(dataTrain.shape, dataTest.shape, labelsTrain.shape, labelsTest.shape)

class CVAE(nn.Module):
    def __init__(self, measures=16, samples=96, notes=96, latent1_size=200, latent2_size=100,
                 class_size=2, units1=2000, units2=400):
        super(CVAE, self).__init__()
        self.measures = measures
        self.samples = samples
        self.notes = notes
        
        self.input_size = samples*notes
        self.class_size = class_size
        self.latent1_size = latent1_size
        self.latent2_size = latent2_size
        self.units1 = units1
        self.units2 = units2

        self.dr = 0.1
        self.prelu_init = 0.25

        self.enc1_fc1 = nn.Linear(self.input_size+self.class_size, self.units1, bias=False)
        self.enc1_bn1 = nn.BatchNorm1d(self.units1)
        self.enc1_fc2 = nn.Linear(self.units1, self.latent1_size, bias=False)
        self.enc1_bn2 = nn.BatchNorm1d(self.latent1_size)
        self.enc2_fc1 = nn.Linear(self.measures*self.latent1_size, self.units2, bias=False)
        self.enc2_bn1 = nn.BatchNorm1d(self.units2)
        self.enc2_fc2 = nn.Linear(self.units2, self.latent2_size, bias=False)
        self.enc2_bn2 = nn.BatchNorm1d(self.latent2_size)
        self.enc_fc_mu = nn.Linear(self.units2, self.latent2_size, bias=False)
        self.enc_fc_logvar = nn.Linear(self.units2, self.latent2_size, bias=False)
        
        self.dec2_fc1 = nn.Linear(self.latent2_size+self.class_size, self.units2, bias=False)
        self.dec2_bn1 = nn.BatchNorm1d(self.units2)
        self.dec2_fc2 = nn.Linear(self.units2, self.units2, bias=False)
        self.dec2_bn2 = nn.BatchNorm1d(self.units2)
        self.dec2_fc3 = nn.Linear(self.units2, self.measures*self.latent1_size, bias=False)
        self.dec2_bn3 = nn.BatchNorm1d(self.measures*self.latent1_size)
        self.dec1_fc1 = nn.Linear(self.latent1_size, self.units1, bias=False)
        self.dec1_bn1 = nn.BatchNorm1d(self.units1)
        self.dec1_fc2 = nn.Linear(self.units1, self.input_size, bias=False)
        
        self.enc1_prelu1 = Parameter(torch.Tensor(self.units1).fill_(self.prelu_init))
        self.enc1_prelu2 = Parameter(torch.Tensor(self.latent1_size).fill_(self.prelu_init))
        self.enc2_prelu1 = Parameter(torch.Tensor(self.units2).fill_(self.prelu_init))
        self.enc2_prelu2 = Parameter(torch.Tensor(self.units2).fill_(self.prelu_init))
        
        self.dec2_prelu1 = Parameter(torch.Tensor(self.units2).fill_(self.prelu_init))
        self.dec2_prelu2 = Parameter(torch.Tensor(self.units2).fill_(self.prelu_init))
        self.dec2_prelu3 = Parameter(torch.Tensor(self.measures*self.latent1_size).fill_(self.prelu_init))
        self.dec1_prelu1 = Parameter(torch.Tensor(self.units1).fill_(self.prelu_init))

    def encoder(self, x, c):
        X = x.view(-1, self.measures, self.input_size)
        
        enc1_outputs = []
        for m in range(self.measures):
            measure = X[:, m, :]
            concat = torch.cat([measure, c], 1)
            hidden1 = F.prelu(self.enc1_bn1(self.enc1_fc1(concat)), self.enc1_prelu1)
            hidden2 = F.prelu(self.enc1_bn2(self.enc1_fc2(hidden1)), self.enc1_prelu2)
            enc1_outputs.append(hidden2)
        
        concat = torch.cat(enc1_outputs, 1)
        
        hidden1 = F.prelu(self.enc2_bn1(self.enc2_fc1(concat)), self.enc2_prelu1)

        return self.enc2_bn2(self.enc2_fc2(hidden1))
    
    def decoder(self, z, c):
        concat = torch.cat([z, c], 1)
        hidden1 = F.prelu(self.dec2_bn1(self.dec2_fc1(concat)), self.dec2_prelu1)
        dropout = F.dropout(hidden1, p=self.dr)
        hidden2 = F.prelu(self.dec2_bn2(self.dec2_fc2(dropout)), self.dec2_prelu2)
        dec1_input = F.prelu(self.dec2_bn3(self.dec2_fc3(hidden2)), self.dec2_prelu3)
        dec1_inputs = dec1_input.view(-1, self.measures, self.latent1_size)
        
        dec1_outputs = []
        for m in range(self.measures):
            measure = dec1_inputs[:, m, :]
            hidden1 = F.prelu(self.dec1_bn1(self.dec1_fc1(measure)), self.dec1_prelu1)
            dropout = F.dropout(hidden1, p=self.dr)
            hidden2 = self.dec1_fc2(dropout)
            dec1_outputs.append(hidden2)
        
        x_hat = torch.stack(dec1_outputs, 1).view(-1, self.measures, self.samples, self.notes)
        
        return x_hat
    
    def forward(self, x, c):
        z = self.encoder(x, c)
        
        x_hat = self.decoder(z, c)
        
        return x_hat, 0, 0


def to_var(x, use_cuda):
    x = Variable(x).float()
    if use_cuda:
        x = x.cuda()
    return x


def one_hot(labels, class_size, use_cuda):
    targets = torch.zeros(labels.size(0), class_size)
    for i, label in enumerate(labels):
        targets[i, label] = 1
    return to_var(targets, use_cuda)


def loss_function(x_hat, x):
    x[x == 2] = 1 # values of 2 mean the note was stuck again while being played already
    loss = F.binary_cross_entropy_with_logits(x_hat, x)
    return loss

use_cuda = True
use_noise = True
batch_size = 1024
num_epochs = 2000
num_classes = 3
latent1_size = 256
latent2_size = 128
hidden_units1 = 2048
hidden_units2 = 2048

model = CVAE(measures=16, samples=96, notes=96,
                latent1_size=latent1_size,
                latent2_size=latent2_size,
                class_size=num_classes,
                units1=hidden_units1,
                units2=hidden_units2)

if len(sys.argv) == 1:
    for m in model.modules():
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight.data)
else:
    params = torch.load(sys.argv[1])
    model.load_state_dict(params)
    

if use_cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=1e-5)

train_data = to_var(torch.from_numpy(dataTrain), use_cuda)
train_labels = one_hot(torch.from_numpy(labelsTrain), num_classes, use_cuda)
train_loader = utils.DataLoader(utils.TensorDataset(train_data, train_labels), batch_size=batch_size, shuffle=True)

test_data = to_var(torch.from_numpy(dataTest), use_cuda)
test_labels = one_hot(torch.from_numpy(labelsTest), num_classes, use_cuda)
test_loader = utils.DataLoader(utils.TensorDataset(test_data, test_labels), batch_size=batch_size)

if not os.path.exists('output'):
    os.makedirs('output')

try:
    best_epoch = -1
    best_loss = 1e6
    best_params = None
    for epoch in range(1, num_epochs):
        model.train()
        epoch_loss = 0
        batches = 0
        for batch_idx, (data, labels) in enumerate(train_loader):
            if use_noise and epoch % 2 == 0:
                noise = torch.rand(*data.shape).cuda() * (torch.cuda.FloatTensor(*data.shape).uniform_() > 0.99).float()
                if use_cuda:
                    noise.cuda()
    
                data = data.clone() + noise
            recon_batch, mu, logvar = model(data, labels)
            optimizer.zero_grad()
            loss = loss_function(recon_batch, data)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.data
            batches += 1

        if epoch_loss != epoch_loss:
            raise Exception()

        epoch_loss /= batches        

        if epoch_loss < best_loss:
            best_epoch = epoch
            best_loss = epoch_loss
            best_params = model.state_dict()
    
        if epoch % 5 == 0:
            with torch.no_grad():
                test_loss = 0
                batches = 0
                for batch_idx, (data, labels) in enumerate(test_loader):
                    model.eval()
                    recon_batch, mu, logvar = model(data, labels)
                    loss = loss_function(recon_batch, data)

                    test_loss += loss.data         
                    batches += 1

                test_loss /= batches        

                print('Train Epoch: {}   \t Train Loss: {:.7f}  \t Test Loss: {:.7f}'.format(epoch, epoch_loss, test_loss))
except Exception as ex:
    print(ex)
    pass

finally:
    print('Saving model from epoch {} with loss {}'.format(best_epoch, best_loss))
    torch.save(best_params, 'best_model.pytch')
    
    torch.save(model.state_dict(), 'last_model.pytch')

    save_data = input('Would you like to save training/testing data? [y/N] ').lower()
    if save_data.startswith('y'):
        torch.save(dataTrain, 'output/training_data.pytch')
        torch.save(dataTest, 'output/testing_data.pytch')
        torch.save(labelsTrain, 'output/training_labels.pytch')
        torch.save(labelsTest, 'output/testing_labels.pytch')

model.load_state_dict(best_params)
model.eval()
c = torch.eye(num_classes, num_classes)
c = to_var(c, use_cuda)
z = to_var(torch.randn(num_classes, latent2_size), use_cuda)
examples = torch.sigmoid(model.decoder(z, c)).data.cpu().numpy()

np.save('output/examples.npy', examples)

recon_batch, mu, logvar = model(data, labels)
np.save('output/reconstructed.npy', torch.sigmoid(recon_batch).data.cpu().numpy())

examples = np.load('output/examples.npy')
for i in range(examples.shape[0]):
    rebuilt_data = rebuilt_data[i]
    rebuilt_data[rebuilt_data < 0.5] = 0
    rebuilt_data[rebuilt_data > 0] = 1
    song = samples_to_music21(rebuilt_data, 16)
    new_file = 'output/example_{}.mid'.format(i)
    song.write('midi', fp=new_file)
    midi_file = mido.MidiFile(new_file)
    midi_file.tracks[0].insert(1, mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(150)))
    midi_file.save(new_file)

