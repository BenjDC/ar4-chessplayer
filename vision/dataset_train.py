import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image_dataset_from_directory
from datetime import datetime

# --------------------------
# PARAMÈTRES GÉNÉRAUX
# --------------------------
DATASET_PATH = "dataset"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 30

# --------------------------
# LOAD DATASETS
# --------------------------
train_ds = image_dataset_from_directory(
    "dataset",
    label_mode="categorical",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="training",
    seed=42
)

val_ds = image_dataset_from_directory(
    "dataset",
    label_mode="categorical",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="validation",
    seed=42
)


class_names = train_ds.class_names
print("Classes détectées :", class_names)

# --------------------------
# DATA AUGMENTATION
# --------------------------
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.10),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.2),
])

# --------------------------
# BASE MODEL (PRETRAINED)
# --------------------------
base_model = MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False  # on gèle

# --------------------------
# MODÈLE FINAL
# --------------------------
inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)

x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)

outputs = layers.Dense(len(class_names), activation="softmax")(x)

model = models.Model(inputs, outputs)

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# --------------------------
# CALLBACKS (EARLY STOP)
# --------------------------
checkpoint_dir = f"model_occupation__TL_{datetime.now().strftime('%Y%m%d_%H%M')}.keras"

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        patience=4,
        restore_best_weights=True,
        monitor="val_accuracy"
    ),
    tf.keras.callbacks.ModelCheckpoint(
        checkpoint_dir,
        monitor="val_accuracy",
        save_best_only=True
    )
]

# --------------------------
# TRAIN
# --------------------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

print("📦 Modèle sauvegardé :", checkpoint_dir)