import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, losses

data_dir = r"C:\Tasks\Steel Surface Defect Inspection System\data\NEU-DET"

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir + r"\train\images", image_size=(224, 224), batch_size=32, shuffle=True)
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir + r"\validation\images", image_size=(224, 224), batch_size=32, shuffle=False)


def build_simple_cnn(n_classes=6):
    return models.Sequential([
        layers.Input(shape=(224, 224, 3)),
        layers.Rescaling(1./255),
        layers.RandomFlip("horizontal_and_vertical"),
        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.BatchNormalization(),
        layers.Conv2D(128, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(n_classes),
    ])


model = build_simple_cnn()
model.compile(
    optimizer=optimizers.Adam(1e-3),
    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

history = model.fit(train_ds, validation_data=val_ds, epochs=15)