from tensorflow import keras

from config import TARGET_SIZE


def build_model(num_classes, use_transfer_learning=False):
    layers = [
        keras.Input(shape=(*TARGET_SIZE, 3)),
        keras.layers.Rescaling(1.0 / 255),
    ]
    if use_transfer_learning:
        backbone = keras.applications.MobileNetV2(
            weights="imagenet", include_top=False, input_shape=(*TARGET_SIZE, 3)
        )
        backbone.trainable = False
        layers.extend([
            keras.layers.Rescaling(2.0, offset=-1.0),
            backbone,
            keras.layers.GlobalAveragePooling2D(),
        ])
    else:
        for filters in (32, 64, 128):
            layers.extend([
                keras.layers.Conv2D(filters, 3, activation="relu"),
                keras.layers.MaxPooling2D(),
            ])
        layers.append(keras.layers.Flatten())

    layers.extend([
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(num_classes, activation="softmax"),
    ])
    model = keras.Sequential(layers)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
