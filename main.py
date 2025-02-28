import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from rich.traceback import install
from visualization.plot import plot_latent_space
install()
from data.data_loader import load_dataset
from models.autoencoder import build_encoder, build_decoder, build_autoencoder
from training.trainer import train_model
from utils.config import IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE, LATENT_DIM, INITIAL_SHAPE, EPOCHS, MODEL_SAVE_PATH

train_ds, val_ds = load_dataset(IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE)
train_ds = train_ds.map(lambda x, y: (x, x))
val_ds = val_ds.map(lambda x, y: (x, x))

def ds_to_numpy(ds):
    images, labels = [], []
    for batch in ds:
        imgs, labs = batch
        images.append(imgs.numpy())
        labels.append(labs.numpy())
    return (np.concatenate(images, axis=0), np.concatenate(labels, axis=0))
x_test, y_test = ds_to_numpy(val_ds)

input_shape = (IMG_HEIGHT, IMG_WIDTH, 1)
encoder = build_encoder(input_shape, latent_dim=LATENT_DIM)
decoder = build_decoder(initial_shape=INITIAL_SHAPE, latent_dim=LATENT_DIM)
autoencoder = build_autoencoder(encoder, decoder)

autoencoder.summary()

# Train the model (train_model should be defined in your training/trainer.py)
train_model(autoencoder, train_ds, val_ds, EPOCHS, MODEL_SAVE_PATH)

plot_latent_space(encoder, x_test, y_test, decoder)
