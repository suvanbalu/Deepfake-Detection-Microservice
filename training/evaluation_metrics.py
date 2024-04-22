import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, auc
import os

# Assuming the plotting functions are imported as before
from plots import plot_confusion_matrix, plot_roc_curve

# Load the model
model_path = 'models/deepfake_detection_models/faces_trained_model.h5'
model = load_model(model_path)

test_dir = './test_dataset'  # Adjust the path as necessary
images = []
labels = []

# Count the number of REAL images
real_path = os.path.join(test_dir, "REAL")
num_real_images = len(os.listdir(real_path))
# Calculate the limit for FAKE images
num_fake_limit = int(10 * num_real_images)

for idx, label in enumerate(["REAL", "FAKE"]):
    test_path = os.path.join(test_dir, label)
    image_paths = os.listdir(test_path)
    
    # If processing FAKE images, limit the number of images processed
    if label == "FAKE":
        image_paths = image_paths[:num_fake_limit]
    
    for img_path in map(lambda x: os.path.join(test_path, x), image_paths):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))  # Resize to match model input, adjust as necessary
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert color from BGR to RGB
        images.append(img)
        labels.append(idx)

images = np.array(images)
# The preprocess_input function expects float inputs
images = preprocess_input(images.astype('float32'))

labels = np.array(labels)

# Predict
predictions = model.predict(images)

# Convert predictions to label indices (0 or 1) based on a threshold (e.g., 0.5)
predicted_classes = (predictions > 0.5).astype('int').reshape(-1)

# Metrics
accuracy = accuracy_score(labels, predicted_classes)
precision = precision_score(labels, predicted_classes)
recall = recall_score(labels, predicted_classes)
print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}")

# Output directory for plots
output_dir = 'evaluation_plots'
os.makedirs(output_dir, exist_ok=True)

# Confusion Matrix Plot
plot_confusion_matrix(labels, predicted_classes, output_dir, "Confusion Matrix")

# ROC Curve Plot
fpr, tpr, _ = roc_curve(labels, predictions)
roc_auc = auc(fpr, tpr)
print(f"AUC: {roc_auc}")
plot_roc_curve(labels, predictions.ravel(), output_dir, "ROC Curve")
