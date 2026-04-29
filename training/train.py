import pandas as pd
import tensorflow as tf
import numpy as np

# =========================
# 1. Load Data
# =========================
df = pd.read_csv("../dataset/chest_xray_metadata_cleaned.csv")

diseases = [
    "Atelectasis","Cardiomegaly","Effusion","Infiltration","Mass",
    "Nodule","Pneumonia","Pneumothorax","Consolidation",
    "Edema","Emphysema","Fibrosis","Pleural_Thickening","Hernia"
]

# Multi-label encoding
for disease in diseases:
    df[disease] = df["Finding Labels"].apply(
        lambda x: 1 if disease in str(x) else 0
    )

# =========================
# 2. Data Generator (Augmentation)
# =========================
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

train_data = datagen.flow_from_dataframe(
    dataframe=df,
    directory="../dataset/images",
    x_col="Image Index",
    y_col=diseases,
    target_size=(224,224),
    batch_size=16,   # 🔥 reduced
    class_mode="raw",
    subset="training"
)

val_data = datagen.flow_from_dataframe(
    dataframe=df,
    directory="../dataset/images",
    x_col="Image Index",
    y_col=diseases,
    target_size=(224,224),
    batch_size=16,
    class_mode="raw",
    subset="validation"
)

# =========================
# 3. FOCAL LOSS (for imbalance)
# =========================
import tensorflow.keras.backend as K

def focal_loss(gamma=2., alpha=0.25):
    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, 1e-8, 1-1e-8)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1-y_pred)
        return -K.mean(alpha * K.pow(1. - pt, gamma) * K.log(pt))
    return loss

# =========================
# 4. Transfer Learning Model
# =========================
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224,224,3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(diseases), activation='sigmoid')
])

# =========================
# 5. Compile
# =========================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),
    loss=focal_loss(),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)

model.summary()

# =========================
# 6. Callbacks
# =========================
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "../backend/model/best_model.h5",
    monitor='val_auc',
    save_best_only=True,
    mode='max'
)

# =========================
# 7. TRAIN (Stage 1)
# =========================
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=15,
    callbacks=[early_stop, checkpoint]
)

# =========================
# 8. Fine-tuning (IMPORTANT)
# =========================
base_model.trainable = True

# freeze most layers, train last 30 layers only
for layer in base_model.layers[:-30]:
    layer.trainable = False

for layer in base_model.layers[-30:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=focal_loss(),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(),
        tf.keras.metrics.AUC(),
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall()
    ]
)

print("🔥 Fine-tuning started...")

model.fit(
    train_data,
    validation_data=val_data,
    epochs=5
)

# =========================
# 9. Save Model
# =========================
model.save("../backend/model/lung_model.h5")

print("✅ FINAL MODEL TRAINED & SAVED!")