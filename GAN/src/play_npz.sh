#!/bin/bash
set -e

src=$1
dst='/home/david/musicGeneration/out.wav'

python npz_to_audio.py $dst $src
cvlc $dst