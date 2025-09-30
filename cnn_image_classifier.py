# cnn_image_classifier.py

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image

# ----------------------------
# 0. Cleanup Dataset (Remove Bad / Ignorable Files)
# ----------------------------
IGNORABLE_EXTS = {".db", ".txt", ".json", ".csv", ".xml", ".zip", ".rar", ".gif"}  # extend as needed

def clean_dataset(folder):
    removed = 0
    for root, _, files in os.walk(folder):
        for f in files:
            path = os.path.join(root, f)
            ext = os.path.splitext(f)[1].lower()

            # Remove junk files
            if ext in IGNORABLE_EXTS:
                print(f"🗑️ Ignored & removed: {path}")
                os.remove(path)
                removed += 1
                continue

            # Validate images
            try:
                Image.open(path).verify()
            except:
                print(f"❌ Bad image removed: {path}")
                os.remove(path)
                removed += 1
    print(f"✅ Cleanup complete. Removed {removed} bad/ignorable files.")

dataset_path = r'C:\Users\rajab\OneDrive\Desktop\Image_Classifier\dataset\cats_vs_dogs'
clean_dataset(dataset_path)

# ----------------------------
# 1. Image Data Generators
# ----------------------------
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)
train_generator = datagen.flow_from_directory(
    r'C:\Users\rajab\OneDrive\Desktop\Image_Classifier\dataset\cats_vs_dogs',
    target_size=(128,128),
    batch_size=32,
    class_mode='binary',
    subset='training',
    shuffle=True,
)

train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(128,128),
    batch_size=32,
    class_mode='binary',
    subset='training',
    shuffle=True
)

validation_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(128,128),
    batch_size=32,
    class_mode='binary',
    subset='validation',
    shuffle=True
)

# ----------------------------
# 2. CNN Model
# ----------------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# ----------------------------
# 3. Callbacks
# ----------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# ----------------------------
# 4. Train the Model
# ----------------------------
print("🚀 Training started...")
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=20,
    callbacks=[early_stop]
)
print("🏁 Training complete.")

# ----------------------------
# 5. Save the Trained Model
# ----------------------------
model.save('cnn_image_classifier.h5')
print("✅ Model saved as cnn_image_classifier.h5")

# ----------------------------
#6. Predict and Display New Images
# ----------------------------
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('cnn_image_classifier.h5')

# Folder with new images
new_image_path = 'new_images'

for img_file in os.listdir(new_image_path):
    try:
        # Load and preprocess image
        img = image.load_img(os.path.join(new_image_path, img_file), target_size=(128,128))
        img_array = image.img_to_array(img)/255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array)
        class_label = 'Dog' if prediction[0][0] > 0.5 else 'Cat'

        # Display image with prediction
        plt.imshow(img)
        plt.title(f"Prediction → {class_label}")
        plt.axis('off')
        plt.show()

    except:
        print(f"⚠️ Skipping unreadable file: {img_file}")