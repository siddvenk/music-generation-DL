# Replace with your path to base directory
BASE_DIRECTORY="/Users/Siddharth/Desktop/College/Masters/Winter2019/eecs598/MusicGeneration/VariationalAutoEncoder/"

MIDI_DIRECTORY=$BASE_DIRECTORY"data/"

OUTPUT_TFRECORD=$BASE_DIRECTORY"notesequences.tfrecord"

# Create tfrecord dataset
convert_dir_to_note_sequences \
--input_dir=$MIDI_DIRECTORY \
--output_file=$OUTPUT_TFRECORD \
--recursive

# Train the network with simplified architecture
music_vae_train \
--config=hierdec-mel_16bar \
--run_dir=$BASE_DIRECTORY \
--mode=train \
--num_steps=500 \
--hparams=batch_size=32,learning_rate=0.0005,enc_rnn_size=[1024,1024],dec_rnn_size=[512,512,512]

# Make the cpkt file (can copy any ckpt number to get music from that state of the model, using 500 as it is the final step)
mkdir $BASE_DIRECTORY"ckpts/"
cp $BASE_DIRECTORY"train/model.ckpt-500.index" $BASE_DIRECTORY"ckpts/"
cp $BASE_DIRECTORY"train/model.ckpt-500.data-00000-of-00001" $BASE_DIRECTORY"ckpts/"
tar -zcvf $BASE_DIRECTORY"ckpts.tar" $BASE_DIRECTORY"ckpts/"

# Generate samples
music_vae_generate \
--config=hierdec-mel_16bar \
--checkpoint_file=$BASE_DIRECTORY"ckpts.tar" \
--mode=sample \
--num_outputs=5 \
--output_dir=$BASE_DIRECTORY"generated/"
