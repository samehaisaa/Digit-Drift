import tensorflow as tf
import time

def train_model(autoencoder, train_data, val_data, epochs, model_save_path):
    """Trains the autoencoder with detailed verbose logging."""
    
    print("\n🔍 Model Summary:")
    autoencoder.summary()

    if not hasattr(autoencoder, "optimizer"):
        autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    
    start_time = time.time()
    print("\n🚀 Training started...\n")

    history = autoencoder.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        verbose=1 )

    end_time = time.time()
    training_time = end_time - start_time

    autoencoder.save(model_save_path)
    print(f"\n✅ Model saved to {model_save_path}")

    print("\n📊 Training Summary:")
    print(f"🔹 Total epochs: {epochs}")
    print(f"🔹 Final training loss: {history.history['loss'][-1]:.4f}")
    print(f"🔹 Final validation loss: {history.history['val_loss'][-1]:.4f}")
    print(f"⏳ Total training time: {training_time:.2f} seconds\n")

    return history  

