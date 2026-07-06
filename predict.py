"""
predict.py
-----------
Standalone command-line script for the Flower Classifier Deep Learning project.

Loads a trained Keras (.h5) CNN model and predicts the flower species
for one or more input images.

Usage:
    python predict.py --image path/to/flower.jpg
    python predict.py --image img1.jpg img2.jpg img3.jpg
    python predict.py --image path/to/flower.jpg --model models/flower_classifier_model.h5
    python predict.py --folder path/to/images/

Requirements (already in requirements.txt):
    tensorflow, numpy, opencv-python (or Pillow)
"""

import argparse
import os
import sys

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

# Update this list to match the classes your model was trained on
# (order must match the training generator's class_indices)
CLASS_NAMES = ["daisy", "dandelion", "rose", "sunflower", "tulip"]

# Update this to match the input size used during training
IMAGE_SIZE = (150, 150)

# Emoji labels purely for friendlier console output (optional)
EMOJI = {
    "rose": "🌹",
    "sunflower": "🌻",
    "tulip": "🌷",
    "daisy": "🌼",
    "dandelion": "🌺",
}


def load_and_preprocess_image(img_path, target_size=IMAGE_SIZE):
    """Load an image file and preprocess it for the CNN model."""
    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    img = keras_image.load_img(img_path, target_size=target_size)
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # normalize, same as training pipeline
    return img_array


def predict_image(model, img_path, class_names=CLASS_NAMES):
    """Run prediction on a single image and return (label, confidence)."""
    img_array = load_and_preprocess_image(img_path)
    predictions = model.predict(img_array, verbose=0)[0]

    top_index = int(np.argmax(predictions))
    label = class_names[top_index]
    confidence = float(predictions[top_index]) * 100

    return label, confidence, predictions


def print_result(img_path, label, confidence, predictions, class_names=CLASS_NAMES):
    emoji = EMOJI.get(label, "🌸")
    print(f"\nImage: {img_path}")
    print(f"Predicted Flower Species: {emoji} {label.capitalize()}")
    print(f"Confidence Score: {confidence:.2f}%")
    print("Class probabilities:")
    for name, prob in sorted(zip(class_names, predictions), key=lambda x: -x[1]):
        print(f"  {name:<12} {prob * 100:6.2f}%")


def gather_image_paths(args):
    paths = []
    if args.image:
        paths.extend(args.image)
    if args.folder:
        valid_ext = (".jpg", ".jpeg", ".png", ".bmp")
        for fname in sorted(os.listdir(args.folder)):
            if fname.lower().endswith(valid_ext):
                paths.append(os.path.join(args.folder, fname))
    return paths


def main():
    parser = argparse.ArgumentParser(
        description="Predict flower species from image(s) using a trained CNN model."
    )
    parser.add_argument(
        "--image", nargs="+", help="Path(s) to one or more flower image files."
    )
    parser.add_argument(
        "--folder", help="Path to a folder of images to classify in batch."
    )
    parser.add_argument(
        "--model",
        default="models/flower_classifier_model.h5",
        help="Path to the trained .h5 model file (default: %(default)s).",
    )
    args = parser.parse_args()

    if not args.image and not args.folder:
        parser.error("Provide at least one --image path or a --folder of images.")

    if not os.path.isfile(args.model):
        print(f"Error: model file not found at '{args.model}'.")
        print("Train the model in Flower_Classifier.ipynb first, or pass --model with the correct path.")
        sys.exit(1)

    print(f"Loading model from {args.model} ...")
    model = load_model(args.model)

    image_paths = gather_image_paths(args)
    if not image_paths:
        print("No images found to classify.")
        sys.exit(1)

    for img_path in image_paths:
        try:
            label, confidence, predictions = predict_image(model, img_path)
            print_result(img_path, label, confidence, predictions)
        except FileNotFoundError as e:
            print(f"\nSkipping: {e}")
        except Exception as e:
            print(f"\nError processing '{img_path}': {e}")


if __name__ == "__main__":
    main()
