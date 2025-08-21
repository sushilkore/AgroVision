# classify_realtime_tta.py
import os, json, glob
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# -------------------------
# CONFIG
# -------------------------
BASE_DIR    = r"C:\Users\ASUS\Downloads\weed"
ART_DIR     = os.path.join(BASE_DIR, "artifacts")
MODEL_GLOB  = os.path.join(ART_DIR, "cnn_mnv2_20250815_165417.h5")   # pick newest
LABELS_PATH = os.path.join(ART_DIR, "labels.json")
IMG_SIZE = (224, 224)
CAMERA_INDEX = 0

def latest_model(path_glob):
    files = glob.glob(path_glob)
    if not files:
        raise FileNotFoundError(f"No models found at {path_glob}")
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]

model_path = latest_model(MODEL_GLOB)
print("Loading model:", model_path)
model = tf.keras.models.load_model(model_path)

with open(LABELS_PATH, "r") as f:
    idx2cls = json.load(f)
labels = [idx2cls[str(i)] for i in range(len(idx2cls))]
print("Labels:", labels)

def preprocess_frame(bgr):
    img = cv2.resize(bgr, IMG_SIZE).astype(np.float32)
    return preprocess_input(img)

# Test-Time Augmentation (TTA): center + 4 corner crops + flips (averaged)
def predict_with_tta(frame_bgr):
    h, w, _ = frame_bgr.shape
    # square crop around center
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    square = frame_bgr[y0:y0+side, x0:x0+side]

    # 5 crops: center + four corners (scaled)
    crops = []
    SCALES = [1.0, 0.85]  # two scales for variety
    for s in SCALES:
        new_side = int(side * s)
        y1 = (side - new_side) // 2
        x1 = (side - new_side) // 2
        center = square[y1:y1+new_side, x1:x1+new_side]
        # center crop
        crops.append(center)
        # four corners
        k = max(1, int(new_side * 0.85))
        crops.append(square[0:k, 0:k])
        crops.append(square[0:k, -k:])
        crops.append(square[-k:, 0:k])
        crops.append(square[-k:, -k:])

    # horizontal flips
    flips = [cv2.flip(c, 1) for c in crops]
    all_patches = crops + flips

    batch = np.stack([preprocess_frame(p) for p in all_patches], axis=0)
    preds = model.predict(batch, verbose=0)        # [N, C]
    mean_pred = preds.mean(axis=0)                 # average over TTA
    return mean_pred

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("Could not open camera")
    raise SystemExit

print("Press ESC to quit.")
while True:
    ok, frame = cap.read()
    if not ok:
        break

    probs = predict_with_tta(frame)
    idx = int(np.argmax(probs))
    conf = float(np.max(probs))
    label = f"{labels[idx]} ({conf:.2f})"

    # draw label (no boxes; whole-frame classification)
    cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Crop Classifier (TTA)", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
