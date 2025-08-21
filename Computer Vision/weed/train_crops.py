# train_crops_strong.py
import os, json, math, datetime, collections
import tensorflow as tf
from tensorflow.keras import layers, models

# -------------------------
# CONFIG
# -------------------------
DATA_DIR   = r"C:\Users\ASUS\Downloads\weed"   # <- your dataset root (3 subfolders)
OUTPUT_DIR = os.path.join(DATA_DIR, "artifacts")
IMG_SIZE   = (224, 224)
BATCH_SIZE = 16
VAL_SPLIT  = 0.2
FREEZE_EPOCHS   = 8
FINETUNE_EPOCHS = 25
BASE_LR = 1e-4
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# Helper: count images per class to compute class weights
# -------------------------
def count_images_per_class(root):
    counts = {}
    for cls in sorted(os.listdir(root)):
        cpath = os.path.join(root, cls)
        if os.path.isdir(cpath):
            n = sum(1 for f in os.listdir(cpath) if f.lower().endswith((".jpg",".jpeg",".png",".bmp",".webp")))
            if n > 0:
                counts[cls] = n
    return counts

counts = count_images_per_class(DATA_DIR)
print("Image counts per class:", counts)

# Compute class weights (inverse frequency)
# They help when one class dominates (e.g., “cauliflower” everywhere).
class_indices_sorted = sorted(counts.keys())
total = sum(counts.values()) if counts else 1
class_weight = {}
if counts:
    # after flow_from_directory we'll know real mapping; we map later
    pass

# -------------------------
# Data generators with strong augmentation
# (use MobileNetV2 preprocess *in the generator*, not in the model)
# -------------------------
preproc_fn = tf.keras.applications.mobilenet_v2.preprocess_input

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=preproc_fn,
    validation_split=VAL_SPLIT,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.25,
    horizontal_flip=True,
    brightness_range=(0.7, 1.3),   # lighting jitter
    channel_shift_range=20.0,       # color jitter
    fill_mode="nearest",
)

val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=preproc_fn,
    validation_split=VAL_SPLIT
)

train_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

num_classes = train_gen.num_classes
print("Class indices:", train_gen.class_indices)

# Now compute class_weight dict matching generator’s indices
if counts:
    # remap: {class_index: weight}
    inv_freq = {}
    for name, idx in train_gen.class_indices.items():
        n = counts.get(name, 1)
        inv_freq[idx] = total / (num_classes * n)
    class_weight = inv_freq
    print("Class weights:", class_weight)

# -------------------------
# Build model (MobileNetV2 backbone)
# -------------------------
base = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    include_top=False,
    weights="imagenet"
)
base.trainable = False

inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = base(inputs, training=False)              # <— no preprocess here; it’s in the generator
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.35)(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.35)(x)
outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)
model = models.Model(inputs, outputs)

# label smoothing helps with noisy/limited data
model.compile(
    optimizer=tf.keras.optimizers.Adam(BASE_LR),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics=["accuracy"]
)

model.summary()

steps_per_epoch = max(1, math.ceil(train_gen.samples / BATCH_SIZE))
val_steps       = max(1, math.ceil(val_gen.samples / BATCH_SIZE))

callbacks_stage1 = [
    tf.keras.callbacks.ModelCheckpoint(
        os.path.join(OUTPUT_DIR, "best_frozen.h5"),
        monitor="val_accuracy", save_best_only=True, verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=6, restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, verbose=1, min_lr=1e-6
    )
]

print("\n[Stage 1] Train with backbone frozen…")
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=FREEZE_EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps,
    callbacks=callbacks_stage1,
    class_weight=class_weight if counts else None
)

# -------------------------
# Fine-tune: unfreeze last ~40 layers of backbone
# -------------------------
base.trainable = True
for layer in base.layers[:-40]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(BASE_LR/10),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics=["accuracy"]
)

callbacks_stage2 = [
    tf.keras.callbacks.ModelCheckpoint(
        os.path.join(OUTPUT_DIR, "best_finetune.h5"),
        monitor="val_accuracy", save_best_only=True, verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=8, restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, verbose=1, min_lr=5e-7
    )
]

print("\n[Stage 2] Fine-tuning top of backbone…")
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=FINETUNE_EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps,
    callbacks=callbacks_stage2,
    class_weight=class_weight if counts else None
)

# -------------------------
# Save final model + labels
# -------------------------
index_to_class = {v: k for k, v in train_gen.class_indices.items()}
labels_path = os.path.join(OUTPUT_DIR, "labels.json")
with open(labels_path, "w") as f:
    json.dump(index_to_class, f, indent=2)

stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
final_h5 = os.path.join(OUTPUT_DIR, f"cnn_mnv2_{stamp}.h5")
model.save(final_h5)

print("\nSaved model:", final_h5)
print("Saved labels:", labels_path)
