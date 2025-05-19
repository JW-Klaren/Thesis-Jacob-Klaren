#Predict .keras file

import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    classification_report, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_curve, roc_auc_score, confusion_matrix
)

# ==== Paths ====
model_path = '/home/u247708/thesis_jwk/data/models/cnn_lstm/1base_ensemble_large_v1.keras'
test_csv_path = '/home/u247708/thesis_jwk/data/gaf_labels_testset.csv'
test_image_folder = '/home/u247708/thesis_jwk/data/GAF_images_testset'
plot_directory = '/home/u247708/thesis_jwk/data/plots/plots_ensemble_test/1plots_cnn_lstm_test_large_v1'

if not os.path.exists(plot_directory):
    os.makedirs(plot_directory)

# ==== Load Test Data ====
data = pd.read_csv(test_csv_path)
data['Price_Movement'] = data['Price_Movement'].astype(str)

# ==== Check for missing images ====
missing = [f for f in data['filename'] if not os.path.exists(os.path.join(test_image_folder, f))]
if missing:
    print("Missing images:", missing)

# ==== Image Generator ====
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_dataframe(
    data, directory=test_image_folder,
    x_col='filename',
    y_col='Price_Movement',
    target_size=(100, 100),
    batch_size=128,
    class_mode='binary',
    shuffle=False
)

# ==== Load Model ====
model = load_model(model_path)
print(f"Model loaded from {model_path}")
sys.stdout.flush()

# ==== Predict ====
y_pred = model.predict(test_generator)
y_pred_binary = (y_pred > 0.5).astype(int).flatten()
true_labels = test_generator.classes

# ==== Save Per-Sample Predictions ====
prediction_output_path = os.path.join(plot_directory, 'predictions_output.csv')

# Create a DataFrame with results
pred_df = pd.DataFrame({
    'filename': test_generator.filenames,
    'true_label': true_labels,
    'predicted_prob': y_pred.flatten(),
    'predicted_label': y_pred_binary
})

# Merge with original CSV for any extra columns (optional)
if 'Date' in data.columns:
    pred_df = pred_df.merge(data[['filename', 'Date']], on='filename', how='left')

# Save to CSV
pred_df.to_csv(prediction_output_path, index=False)
print(f"Saved per-sample predictions to: {prediction_output_path}")
sys.stdout.flush()

# ==== Metrics ====
fpr, tpr, thresholds = roc_curve(true_labels, y_pred)
auc_score = roc_auc_score(true_labels, y_pred)
classification_rep = classification_report(true_labels, y_pred_binary, output_dict=True)
conf_matrix = confusion_matrix(true_labels, y_pred_binary)

weighted_precision = classification_rep['weighted avg']['precision']
weighted_recall = classification_rep['weighted avg']['recall']
weighted_f1 = classification_rep['weighted avg']['f1-score']
precision = precision_score(true_labels, y_pred_binary, zero_division=1)
recall = recall_score(true_labels, y_pred_binary, zero_division=1)
f1 = f1_score(true_labels, y_pred_binary, zero_division=1)
mcc = matthews_corrcoef(true_labels, y_pred_binary)
accuracy = (y_pred_binary == true_labels).mean()

# ==== Print & Save Metrics ====
metrics_text = f"""
Test Set Metrics:

Accuracy: {accuracy:.4f}
Weighted Precision: {weighted_precision:.4f}
Weighted Recall: {weighted_recall:.4f}
Weighted F1-score: {weighted_f1:.4f}
Precision: {precision:.4f}
Recall: {recall:.4f}
F1-score: {f1:.4f}
Matthews Correlation Coefficient: {mcc:.4f}
AUC Score: {auc_score:.4f}

Confusion Matrix:
{conf_matrix}
"""

print(metrics_text)
sys.stdout.flush()

# Save metrics
metrics_path = os.path.join(plot_directory, "metrics_test_summary.txt")
with open(metrics_path, "w") as f:
    f.write(metrics_text)
print(f"Saved test metrics summary to: {metrics_path}")

# ==== ROC Curve ====
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Test Set)')
plt.legend(loc='lower right')

roc_path = os.path.join(plot_directory, 'roc_curve_test.png')
plt.savefig(roc_path)
plt.close()
print(f"Saved ROC curve to: {roc_path}")

# ==== Confusion Matrix Plot ====
cm = confusion_matrix(true_labels, y_pred_binary)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')

cm_path = os.path.join(plot_directory, 'confusion_matrix.png')
plt.savefig(cm_path)
plt.close()
print(f"Saved confusion matrix to: {cm_path}")