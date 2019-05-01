# music-generation-DL
*Contributers: Siddharth Venkatesan, Nick Adams, David Zhang, and Spencer Vagg*

---
## MusicRNN
This program generates music depending on the genre desired by using a Vanilla RNN. There are three RNN layers, two Fully Connected layers, and a Softmax Activation function at the end. To train the model, you can run the command 
```python
python MusicRNN.py genre_of_music num_epochs
```
The three genres to choose from are title_screen, battle, and piano. Each epoch the model's weights are saved, so training can stop at any point, and songs will still be able to be generated. To then generate a new song using these trained weights, you can run the command
```python
python Generate.py genre_of_music num_notes_to_generate
```
---
## FCE
1. Download .mid/.midi files and use the convert_to_npy.py script to convert them into simple numpy arrays
1. Create a data directory containing 3 subdirectories: title, battle, piano. Place .npy files in their proper location.
1. Confirm you have the Python packages PyTorch, music21, and mido installed.
1. Run `music.py`. Make sure it is at the same level as the data/ directory.
---
## HVAE

1. Download and install Anaconda. We use anaconda as our environment manager
2. Assuming you are on a linux machine, you can download and set up the magenta environmnet using
```bash
curl https://raw.githubusercontent.com/tensorflow/magenta/master/magenta/tools/magenta-install.sh > /tmp/magenta-install.sh
bash /tmp/magenta-install.sh
```
3. Open a new terminal window so that the changes take effect
4. Run 'source activate magenta' to enter the magenta environment
5. Change the BASE_DIRECTORY in vae_train.sh to your base directory
6. Run the vae_train.sh script to start the process of data formatting, training, and sample generation
---
## GAN

1. Download MuseGAN from https://github.com/salu133445/musegan.
2. Run gen_train.py to preprocess the MIDI files and save them into an npz file, which stores the data as a sparse matrix.
3. Create an experiment in museGAN. Set the training data name and other parameters. (See the documentation of MuseGAN.)
4. Train the network and obtain generated music.
5. Run play_npz.sh to post-process the musical phrases and convert them into a single MIDI file.
