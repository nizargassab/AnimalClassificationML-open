import argparse
import json
from pathlib import Path

from config import MODEL_DIR, TARGET_SIZE


def main():
    parser = argparse.ArgumentParser(description="Predict the class of an animal image.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f"Image does not exist: {args.image}")

    from tensorflow import keras

    try:
        class_names = json.loads(
            (args.model_dir / "class_names.json").read_text(encoding="utf-8")
        )
        model = keras.models.load_model(
            args.model_dir / "best_model.keras", compile=False
        )
        with keras.utils.load_img(args.image, target_size=TARGET_SIZE) as image:
            pixels = keras.utils.img_to_array(image)
        probabilities = model.predict(pixels[None], verbose=0)[0].tolist()
        if len(probabilities) != len(class_names):
            raise ValueError("The saved class labels do not match the model.")
        index = max(range(len(probabilities)), key=probabilities.__getitem__)
    except (ValueError, OSError) as error:
        parser.error(str(error))

    print(f"Predicted class: {class_names[index]}")
    print(f"Model confidence: {probabilities[index]:.2%}")


if __name__ == "__main__":
    main()
