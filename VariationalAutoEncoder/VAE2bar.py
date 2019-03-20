import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

class VAE2bar(nn.Module):
	'''
	The architecture for this model is based on the paper: 
			https://nips2017creativity.github.io/doc/Hierarchical_Variational_Autoencoders_for_Music.pdf

	Data Format:
		Sequences of length 32 (2 measures of 16th notes)
		Notes take one of 90 different states (88 notes on a piano, hold state, stop state)

		Input Shape: [SeqLen, Batch, InputDim]
			SeqLen = 32
			InputDim = 90

	Encoder:
		Sinlge-layer bi-directional LSTM with 2048 hidden units
		Outputs concatenated
		Passed through linear layer to produce 512 outputs (256 mu and 256 p-sigma)

	Decoder:

	'''

	def __init__(input_dim=90, encoding_layers=1, encoding_units=2048, latent_size=512, decoding_layers=3, decoding_units=2048):

		super(VAE2bar, self).__init__()

		self.num_encoding_layers = encoding_layers
		self.encoding_units = encoding_size
		self.num_decoding_layers = decoding_layers
		self.decoding_units = decoding_units

		# Encoder
		self.encoding = nn.LSTM(input_size=input_dim,
								hidden_size=encoding_units,
								num_layers=encoding_layers,
								bidirectional=True)

		self.encodingFC = nn.Linear(encoding_units*2, latent_size)

		# Decoder
		self.decoding = nn.LSTM(input_size=latent_size,
								hidden_size=decoding_units,
								num_layers=decoding_layers)

		self.decodingFC = nn.Linear(decoding_units, input_dim)

		print('###################### Model Summary #####################')
		print('### input_dim: {}'.format(input_dim))
		print('### encoding_layers: {}'.format(encoding_layers))
		print('### encoding_units: {}'.format(encoding_units))
		print('### latent_size: {}'.format(latent_size))
		print('### decoding_layers: {}'.format(decoding_layers))
		print('### decoding_units: {}'.format(decoding_units))
		print('##########################################################')

	def encode(inputs):

