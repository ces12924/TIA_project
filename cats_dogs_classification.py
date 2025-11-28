"""
Train a simple CNN to classify cats vs dogs using Keras.

HOW TO USE:
1. Prepare your data folder like this:

   /path/to/your/data/
       train/
           cats/
               cat001.jpg
               cat002.jpg
               ...
           dogs/
               dog001.jpg
               dog002.jpg
               ...
       val/
           cats/
               ...
           dogs/
               ...

2. Set DATA_DIR below to "/path/to/your/data"
3. Install TensorFlow (once):  pip install tensorflow
4. Run this script:  python train_cats_dogs_keras.py
"""

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ===================== CONFIG ===================== #
# >>>>> EDIT THIS LINE ONLY <<<<<
DATA_DIR = "/path/to/your/data"   # e.g. "C:/Users/you/datasets/cats_dogs"

IMG_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 10
# =================================================== #


def build_datasets():
    """Create training and validation datasets from folder structure."""
    train_dir = os.path.join(DATA_DIR, "train")
    val_dir   = os.path.join(DATA_DIR, "val")

    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        raise FileNotFoundError(
            f"Could not find 'train' or 'val' folders inside DATA_DIR: {DATA_DIR}\n"
            "Expected:\n"
            "  DATA_DIR/train/cats, DATA_DIR/train/dogs\n"
            "  DATA_DIR/val/cats,   DATA_DIR/val/dogs"
        )

    print(f"Using training data from: {train_dir}")
    print(f"Using validation data from: {val_dir}")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",   # 0/1 labels
        shuffle=True,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        shuffle=False,
    )

    # Optional: performance tweaks
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds   = val_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds


def build_model():
    """Build a simple CNN for binary classification (cat vs dog)."""
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ],
        name="data_augmentation",
    )

    normalization_layer = layers.Rescaling(1.0 / 255)

    inputs = keras.Input(shape=(*IMG_SIZE, 3))

    x = data_augmentation(inputs)
    x = normalization_layer(x)

    x = layers.Conv2D(32, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)  # output: probability of "dog"

    model = keras.Model(inputs, outputs, name="cats_vs_dogs_cnn")

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    return model


def main():
    print("Loading datasets...")
    train_ds, val_ds = build_datasets()

    print("\nBuilding model...")
    model = build_model()

    print("\nStarting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
    )

    # Save model
    save_path = "cats_vs_dogs_cnn.keras"
    model.save(save_path)
    print(f"\nModel saved to: {save_path}")

    # Optional: show final accuracies
    train_acc = history.history["accuracy"][-1]
    val_acc   = history.history["val_accuracy"][-1]
    print(f"Final training accuracy:   {train_acc:.3f}")
    print(f"Final validation accuracy: {val_acc:.3f}")


if __name__ == "__main__":
    main()

