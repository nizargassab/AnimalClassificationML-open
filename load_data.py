from tensorflow import keras

from config import BATCH_SIZE, SEED, TARGET_SIZE


def load_data(base_dir):
    options = {
        "image_size": TARGET_SIZE,
        "batch_size": BATCH_SIZE,
        "label_mode": "int",
    }
    train = keras.utils.image_dataset_from_directory(
        base_dir / "train", shuffle=True, seed=SEED, **options
    )
    class_names = train.class_names
    validation = keras.utils.image_dataset_from_directory(
        base_dir / "validation", class_names=class_names, shuffle=False, **options
    )
    test = keras.utils.image_dataset_from_directory(
        base_dir / "test", class_names=class_names, shuffle=False, **options
    )
    return train, validation, test, class_names
