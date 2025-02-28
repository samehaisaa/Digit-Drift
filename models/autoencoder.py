import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dense, Reshape
from tensorflow.keras.models import Model

def build_encoder(input_shape, latent_dim=2):
    input_img = Input(shape=input_shape)
    
    x = Conv2D(32, (3,3), activation="relu", padding="same")(input_img)
    x = MaxPooling2D((2,2), padding="same")(x)  # 128 -> 64
    x = Conv2D(64, (3,3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2,2), padding="same")(x)  # 64 -> 32
    
    x = Flatten()(x)
    latent = Dense(latent_dim, activation="relu", name="latent")(x)
    
    encoder = Model(input_img, latent, name="encoder")
    return encoder

def build_decoder(initial_shape=(32,32,64), latent_dim=2):
    latent_input = Input(shape=(latent_dim,))
    
    x = Dense(initial_shape[0] * initial_shape[1] * initial_shape[2], activation="relu")(latent_input)
    x = Reshape(initial_shape)(x)
    
    # (32,32) -> (64,64) -> (128,128)
    x = UpSampling2D((2,2))(x)  # (32,32,64) -> (64,64,64)
    x = Conv2D(64, (3,3), activation="relu", padding="same")(x)
    
    x = UpSampling2D((2,2))(x)  # (64,64,64) -> (128,128,64)
    decoded = Conv2D(1, (3,3), activation="sigmoid", padding="same")(x) 
    
    decoder = Model(latent_input, decoded, name="decoder")
    return decoder

def build_autoencoder(encoder, decoder):
    input_img = encoder.input
    latent = encoder(input_img)
    reconstructed = decoder(latent)
    autoencoder = Model(input_img, reconstructed, name="autoencoder")
    autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
    return autoencoder
