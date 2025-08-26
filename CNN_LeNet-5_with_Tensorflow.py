# ==============================
# LeNet-5 pour Tifinagh Dataset
# ==============================

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

# ----------------------------------------------------
# 1. Fonction de chargement du dataset Tifinagh
# ----------------------------------------------------
def load_tifinagh_dataset(data_path='amhcd-data-64/tifinagh-images/'):
    assert os.path.exists(data_path), "Chemin des données invalide"
    images = []
    labels = []
    class_names = sorted(os.listdir(data_path))
    
    label_encoder = LabelEncoder()
    label_encoder.fit(class_names)
    
    for class_name in class_names:
        class_path = os.path.join(data_path, class_name)
        if not os.path.isdir(class_path):
            continue
            
        for img_file in os.listdir(class_path):
            if img_file.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_path, img_file)
                img = Image.open(img_path).convert('L')   # Niveaux de gris
                img = img.resize((32, 32))               # Redimensionner
                img_array = np.array(img) / 255.0        # Normalisation [0,1]
                images.append(img_array)
                labels.append(class_name)
    
    assert len(images) > 0, "Aucune image chargée"
    X = np.array(images).reshape(-1, 32, 32, 1)
    y = label_encoder.transform(labels)
    
    encoder = OneHotEncoder()
    y = encoder.fit_transform(y.reshape(-1, 1)).toarray()
    
    return X, y, label_encoder.classes_

# ----------------------------------------------------
# 2. Chargement des données
# ----------------------------------------------------
X, y, class_names = load_tifinagh_dataset("amhcd-data-64/tifinagh-images/")
num_classes = len(class_names)
print(f"Dataset chargé : {X.shape[0]} images, {num_classes} classes")

# Division train/val/test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# ----------------------------------------------------
# 3. Définition du modèle LeNet-5 modifié
# ----------------------------------------------------
def build_lenet5(input_shape=(32, 32, 1), num_classes=33):
    model = models.Sequential([
        # C1 : Convolution 5x5 avec 6 filtres
        layers.Conv2D(6, kernel_size=(5, 5), activation='tanh', input_shape=input_shape),
        layers.AveragePooling2D(pool_size=(2, 2), strides=2),

        # C3 : Convolution 5x5 avec 16 filtres
        layers.Conv2D(16, kernel_size=(5, 5), activation='tanh'),
        layers.AveragePooling2D(pool_size=(2, 2), strides=2),

        # C5 : Fully Connected
        layers.Flatten(),
        layers.Dense(120, activation='tanh'),
        layers.Dense(84, activation='tanh'),

        # Sortie
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

model = build_lenet5(num_classes=num_classes)

# ----------------------------------------------------
# 4. Compilation
# ----------------------------------------------------
model.compile(optimizer=optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ----------------------------------------------------
# 5. Entraînement
# ----------------------------------------------------
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=15,
                    batch_size=128,
                    verbose=1)

# ----------------------------------------------------
# 6. Évaluation sur Test
# ----------------------------------------------------
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_acc*100:.2f}%")

# ----------------------------------------------------
# 7. Visualisation des courbes
# ----------------------------------------------------
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Val Loss")
plt.legend()
plt.title("Courbe de perte (Loss)")

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label="Train Acc")
plt.plot(history.history['val_accuracy'], label="Val Acc")
plt.legend()
plt.title("Courbe d'accuracy")

plt.show()

# ----------------------------------------------------
# 8. Matrice de confusion
# ----------------------------------------------------
y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues, xticks_rotation=90)
plt.show()
