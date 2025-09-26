import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.metrics import classification_report
import numpy as np

# ===============================
# SETTINGS
# ===============================
IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 10  

DATA_DIR_TRAIN = 'dataset/Train'
DATA_DIR_TEST = 'dataset/Test'

# ===============================
# LOAD DATA
# ===============================
train_ds = image_dataset_from_directory(
    DATA_DIR_TRAIN,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',  
    shuffle=True
)

test_ds = image_dataset_from_directory(
    DATA_DIR_TEST,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)
print("Classes:", class_names)

# Normalize + prefetch
normalization_layer = layers.Rescaling(1./255)
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y)).cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y)).cache().prefetch(buffer_size=AUTOTUNE)

# Data augmentation for training
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
])

# ===============================
# MODEL 1: Simple CNN
# ===============================
def build_simple_cnn():
    model = models.Sequential([
        layers.Input(shape=IMG_SIZE + (3,)),
        data_augmentation,
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model1 = build_simple_cnn()
model1.summary()

history1 = model1.fit(train_ds, epochs=EPOCHS, validation_data=test_ds)

# Evaluate Model 1
y_true = np.concatenate([y for x, y in test_ds], axis=0)
y_pred = model1.predict(test_ds)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_true, axis=1)

print("\n=== Model 1: Simple CNN Report ===")
print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))

# ===============================
# MODEL 2: MobileNetV2 Transfer Learning
# ===============================
def build_mobilenetv2():
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # freeze base layers initially

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model2 = build_mobilenetv2()
model2.summary()

history2 = model2.fit(train_ds, epochs=EPOCHS, validation_data=test_ds)

# Evaluate Model 2
y_pred2 = model2.predict(test_ds)
y_pred_classes2 = np.argmax(y_pred2, axis=1)

print("\n=== Model 2: MobileNetV2 Report ===")
print(classification_report(y_true_classes, y_pred_classes2, target_names=class_names))

# ===============================
# SAVE MODELS
# ===============================
model1.save("simple_cnn_model.h5")
print("Saved simple_cnn_model.h5")

model2.save("mobilenetv2_model.h5")
print("Saved mobilenetv2_model.h5")

# Save best automatically based on val_accuracy
acc1 = history1.history['val_accuracy'][-1]
acc2 = history2.history['val_accuracy'][-1]

if acc1 >= acc2:
    model1.save("best_model.h5")
    print(f"Best model = Simple CNN ({acc1:.2%}) saved as best_model.h5")
else:
    model2.save("best_model.h5")
    print(f"Best model = MobileNetV2 ({acc2:.2%}) saved as best_model.h5")
