import os
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

image_size = (64, 64)
train_directory = 'DATASET/TRAIN'
test_directory = 'DATASET/TEST'

def load_images_from_folder(folder, label, image_size=image_size):
    images, labels = [], []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize(image_size)
            images.append(np.array(img))
            labels.append(label)
        except Exception as e:
            print(f"Error loading image {filename}: {e}")
    return images, labels

def load_dataset(base_dir):
    organic_dir = os.path.join(base_dir, 'O')
    recyclable_dir = os.path.join(base_dir, 'R')
    organic_images, organic_labels = load_images_from_folder(organic_dir, 0)
    recyclable_images, recyclable_labels = load_images_from_folder(recyclable_dir, 1)
    images = organic_images + recyclable_images
    labels = organic_labels + recyclable_labels
    return np.array(images), np.array(labels)

train_images, train_labels = load_dataset(train_directory)
test_images, test_labels = load_dataset(test_directory)

def flatten_images(images):
    return images.reshape(images.shape[0], -1)

train_images_flattened = flatten_images(train_images)
test_images_flattened = flatten_images(test_images)

pca = PCA(n_components=20)
train_images_pca = pca.fit_transform(train_images_flattened)
test_images_pca = pca.transform(test_images_flattened)

scaler = StandardScaler()
train_images_scaled = scaler.fit_transform(train_images_pca)
test_images_scaled = scaler.transform(test_images_pca)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt']
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(train_images_scaled, train_labels)

best_rf_model = grid_search.best_estimator_

best_predictions = best_rf_model.predict(test_images_scaled)
best_accuracy = accuracy_score(test_labels, best_predictions)
print(f'Optimized Accuracy: {best_accuracy * 100:.2f}%')
print('Optimized Classification Report:')
print(classification_report(test_labels, best_predictions, target_names=['Organic', 'Recyclable']))
print('Optimized Confusion Matrix:')
print(confusion_matrix(test_labels, best_predictions))
cv_scores = cross_val_score(best_rf_model, train_images_scaled, train_labels, cv=5)
print(f'Cross-Validation Scores: {cv_scores}')
print(f'Mean Cross-Validation Score: {np.mean(cv_scores) * 100:.2f}%')

model_path = "random_forest_model.joblib"
joblib.dump(best_rf_model, model_path)
print(f"Model saved to {model_path}")

rf_model = joblib.load(model_path)

model = keras.Sequential([
    layers.Dense(128, input_shape=(train_images_scaled.shape[1],), activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images_scaled, train_labels, epochs=50, batch_size=32, validation_split=0.2)

model.save("model.h5")
print("Model saved as model.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("model.tflite", "wb") as f:
    f.write(tflite_model)
print("Model saved as model.tflite")
