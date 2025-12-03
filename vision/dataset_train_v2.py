import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

dataset_path = "dataset"
IMG_SIZE = (128, 128)
BATCH = 32
VAL_SPLIT = 0.2

# ============
# DATA LOADER
# ============

train_datagen = ImageDataGenerator(
    rescale=1/255.,
    validation_split=VAL_SPLIT,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.05,
    height_shift_range=0.05,
    brightness_range=[0.6, 1.4]
)

train_gen = train_datagen.flow_from_directory(
    dataset_path,
    target_size=IMG_SIZE,
    batch_size=BATCH,
    subset="training",
    class_mode="categorical"
)

val_gen = train_datagen.flow_from_directory(
    dataset_path,
    target_size=IMG_SIZE,
    batch_size=BATCH,
    subset="validation",
    class_mode="categorical"
)

print("Classes détectées:", train_gen.class_indices)

# ============
#   MODEL
# ============

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation="relu", input_shape=(*IMG_SIZE, 3)),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3,3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.3),               # anti-surapprentissage
    layers.Dense(3, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ============
# TRAIN
# ============

callbacks = [
    EarlyStopping(
        monitor="val_accuracy",
        patience=4,
        restore_best_weights=True
    )
]

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=40,
    callbacks=callbacks
)

model.save("model_plateau.keras")
print("📦 Modèle sauvegardé !")