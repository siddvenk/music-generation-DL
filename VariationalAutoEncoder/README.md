## Hierarchical Variational Autoencoder

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