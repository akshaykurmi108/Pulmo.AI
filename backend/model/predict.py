import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# =========================
# 1. Disease List
# =========================
diseases = [
    "Atelectasis","Cardiomegaly","Effusion","Infiltration","Mass",
    "Nodule","Pneumonia","Pneumothorax","Consolidation",
    "Edema","Emphysema","Fibrosis","Pleural_Thickening","Hernia"
]

# =========================
# 2. Load Model
# =========================
model_path = os.path.join(os.path.dirname(__file__), "lung_model.h5")

model = tf.keras.models.load_model(
    model_path,
    compile=False   # 🔥 important (focal loss error avoid)
)

# =========================
# 3. Prediction Function
# =========================
def predict_image(img_path):
    # Load image
    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img) / 255.0

    # Ensure RGB
    if img_array.shape[-1] == 1:
        img_array = np.repeat(img_array, 3, axis=-1)

    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    preds = model.predict(img_array)[0]

    # =========================
    # 4. Threshold + Sorting
    # =========================
    threshold = 0.2  # 🔥 best for medical

    results = []

    for i, prob in enumerate(preds):
        if prob >= threshold:
            results.append((diseases[i], prob))

    # Sort by probability
    results = sorted(results, key=lambda x: x[1], reverse=True)

    # =========================
    # 5. Output Logic
    # =========================

    # Case 1: Diseases found
    if len(results) > 0:
        output = []
        for disease, prob in results[:3]:   # top 3 only
            output.append(f"{disease} → {prob*100:.2f}%")

        return {
            "status": "Disease Detected",
            "predictions": output
        }

    # Case 2: No disease
    else:
        max_index = np.argmax(preds)
        return {
            "status": "Normal / Low Risk",
            "top_prediction": f"{diseases[max_index]} ({preds[max_index]*100:.2f}%)"
        }