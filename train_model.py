import json
from pathlib import Path
from tempfile import TemporaryDirectory

from tensorflow import keras

from build_model import build_model
from config import SEED
from load_data import load_data


def train_model(data_dir, output_dir, epochs, transfer_learning):
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError(f"Choose a new model output directory: {output_dir}")

    keras.utils.set_random_seed(SEED)
    train, validation, test, class_names = load_data(data_dir)
    model = build_model(len(class_names), use_transfer_learning=transfer_learning)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".animal-model-", dir=output_dir.parent) as tmp:
        stage = Path(tmp) / "model"
        stage.mkdir()
        checkpoint = stage / "best_model.keras"
        model.fit(
            train,
            validation_data=validation,
            epochs=epochs,
            callbacks=[
                keras.callbacks.ModelCheckpoint(
                    checkpoint, monitor="val_loss", save_best_only=True
                ),
                keras.callbacks.EarlyStopping(monitor="val_loss", patience=5),
            ],
        )
        best = keras.models.load_model(checkpoint)
        metrics = best.evaluate(test, return_dict=True)
        (stage / "class_names.json").write_text(
            json.dumps(class_names, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        stage.rename(output_dir)

    return metrics
