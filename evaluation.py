import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from rich.traceback import install
from visualization.plot import plot_latent_space
install()

from data.data_loader import load_dataset
from visualization.plot import plot_latent_space
from utils.config import IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE

_, test_ds = load_dataset(IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE)

def ds_to_numpy_with_labels(ds):
    images, labels = [], []
    for batch in ds:
        imgs, labs = batch
        images.append(imgs.numpy())
        labels.append(labs.numpy())
    return np.concatenate(images, axis=0), np.concatenate(labels, axis=0)

x_test, y_test = ds_to_numpy_with_labels(test_ds)
print("x_test.shape:", x_test.shape)  
print("y_test.shape:", y_test.shape) 

encoder = load_model("encoder_extracted.h5")
decoder = load_model("decoder_extracted.h5")
print("✅ Loaded encoder and decoder.")

plot_latent_space(encoder, x_test, y_test, decoder)
