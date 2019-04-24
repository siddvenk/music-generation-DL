# music-generation-DL
*Contributers: Siddharth Venkatesan, Nick Adams, David Zhang, and Spencer Vagg*

---
###MusicRNN
This program generates music depending on the genre desired by using a Vanilla RNN. There are three RNN layers, two Fully Connected layers, and a Softmax Activation function at the end. To train the model, you can run the command 
```python
python MusicRNN.py genre_of_music num_epochs
```
The three genres to choose from are title_screen, battle, and piano. Each epoch the model's weights are saved, so training can stop at any point, and songs will still be able to be generated. To then generate a new song using these trained weights, you can run the command
```python
python Generate.py genre_of_music num_notes_to_generate
```
---
