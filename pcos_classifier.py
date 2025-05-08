#!/usr/bin/env python3
import os
import argparse
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
import numpy as np
import matplotlib.pyplot as plt


# Utility: Plot training curves

def plot_training_curves(history):
    # Plot Loss
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Plot Accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


# Training function using MobileNetV2 for Transfer Learning

def train_model(train_dir, val_dir, img_size=(224, 224), batch_size=32,
                epochs=20, model_save_path="pcos_classifier_mobilenetv2.keras"):
    # Load datasets from directory with RGB images
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='binary',
        color_mode='rgb'
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='binary',
        color_mode='rgb'
    )

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Data augmentation
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1)
    ])

    # Define the input shape (MobileNetV2 expects 3-channel images)
    input_shape = img_size + (3,)

    # Load the MobileNetV2 base model with pre-trained ImageNet weights (exclude top)
    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                                   include_top=False,
                                                   weights='imagenet')
    base_model.trainable = False  # Freeze the base model

    # Build the model
    inputs = tf.keras.Input(shape=input_shape)
    x = data_augmentation(inputs)
    # Preprocess inputs - required by MobileNetV2
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs, outputs)

    # Compile the model
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    # Define callbacks
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                            patience=2, min_lr=1e-6, verbose=1)
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=3,
                                         restore_best_weights=True, verbose=1)

    # Train the model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[reduce_lr, early_stop]
    )

    # Save the trained model
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

    # Plot training curves
    plot_training_curves(history)

    return model


# Inference: Load and preprocess image

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """
    Loads an image as RGB, resizes it, and converts it to a numpy array.
    """
    img = tf.keras.utils.load_img(image_path, target_size=target_size, color_mode='rgb')
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# Inference: Detect PCOS using the saved model

def detect_pcos(image_path, model_path="pcos_classifier_mobilenetv2.keras", threshold=0.5):
    """
    Loads the saved model and predicts if the input image shows PCOS.
    """
    model = tf.keras.models.load_model(model_path)
    img_array = load_and_preprocess_image(image_path)
    prediction = model.predict(img_array)
    if prediction[0][0] >= threshold:
        result = "PCOS not detected"
    else:
        result = "PCOS detected"
    print(f"Image: {image_path} => Prediction: {result} (score: {prediction[0][0]:.4f})")
    return prediction


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train PCOS classifier using transfer learning with MobileNetV2 or detect PCOS on an image."
    )
    parser.add_argument("--download", action="store_true", help="Download Kaggle dataset before training")
    parser.add_argument("--dataset", type=str, default="erniegomila/pcos-classifier",
                        help="Kaggle dataset slug (username/dataset-name)")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--detect", type=str, help="Path to an image for PCOS detection")
    parser.add_argument("--train_dir", type=str, default="data/train",
                        help="Path to the training data directory")
    parser.add_argument("--val_dir", type=str, default="data/val",
                        help="Path to the validation data directory")
    parser.add_argument("--model_path", type=str, default="pcos_classifier_mobilenetv2.keras",
                        help="Path to save or load the model")
    args = parser.parse_args()

    if args.download:
        download_and_extract_dataset(args.dataset, local_path='data', unzip=True)

    if args.train:
        train_model(args.train_dir, args.val_dir, model_save_path=args.model_path)
    elif args.detect:
        detect_pcos(args.detect, model_path=args.model_path)
    else:
        print("Please specify --download and/or --train, or --detect with a valid image path.")
