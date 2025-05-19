#Predict .h5 file

import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, RepeatVector, LSTM
from keras.models import Model
from keras.optimizers import Adam
from keras.initializers import glorot_uniform
from keras.regularizers import l1_l2
from sklearn.metrics import (
    classification_report, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_curve, roc_auc_score, confusion_matrix
)

# ==== Paths ====
weights_path = '/home/u247708/thesis_jwk/data/models/cnn/1cnn_train_large.weights.h5'
test_csv_path = '/home/u247708/thesis_jwk/data/gaf_labels_testset.csv'
test_image_folder = '/home/u247708/thesis_jwk/data/GAF_images_testset'
plot_directory = '/home/u247708/thesis_jwk/data/plots_cnn_test/1plots_cnn_test_large'

if not os.path.exists(plot_directory):
    os.makedirs(plot_directory)

# ==== Load Test CSV ====
data = pd.read_csv(test_csv_path)
data['Price_Movement'] = data['Price_Movement'].astype(str)

# ==== Check for Missing Images ====
missing = [f for f in data['filename'] if not os.path.exists(os.path.join(test_image_folder, f))]
if missing:
    print("Missing images:", missing)
    sys.stdout.flush()

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

# Create the base model
base_model = InceptionV3(weights=None, include_top=False, input_shape=(100, 100, 3))

# Additional layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu',kernel_regularizer=l1_l2(l1=1e-5, l2=1e-5), kernel_initializer=glorot_uniform())(x)
x = Dense(256, activation='relu',kernel_regularizer=l1_l2(l1=1e-5, l2=1e-5), kernel_initializer=glorot_uniform())(x)
x = Dropout(0.2)(x)
x = Dense(256, activation='relu',kernel_regularizer=l1_l2(l1=1e-5, l2=1e-5), kernel_initializer=glorot_uniform())(x)
x = Dropout(0.2)(x)
x = Dense(128, activation='relu',kernel_regularizer=l1_l2(l1=1e-5, l2=1e-5), kernel_initializer=glorot_uniform())(x)
predictions = Dense(1, activation='sigmoid')(x)

# Build the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy', 'AUC'])

# ==== Predict ====
y_pred = model.predict(test_generator)
y_pred_binary = (y_pred > 0.5).astype(int).flatten()
true_labels = test_generator.classes

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

# ==== Text Summary ====
metrics_text = f"""
Test Set Metrics (from weights):

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

# ==== Save Outputs ====

# 1. Save metrics summary to TXT
metrics_path = os.path.join(plot_directory, "metrics_test_from_weights.txt")
with open(metrics_path, "w") as f:
    f.write(metrics_text)
print(f"Saved test metrics to: {metrics_path}")
sys.stdout.flush()

# 2. Save predictions to CSV
prediction_df = pd.DataFrame({
    'filename': data['filename'],
    'true_label': true_labels,
    'predicted_prob': y_pred.flatten(),
    'predicted_label': y_pred_binary
})
predictions_path = os.path.join(plot_directory, "predictions_output.csv")
prediction_df.to_csv(predictions_path, index=False)
print(f"Saved predictions to: {predictions_path}")
sys.stdout.flush()

# 3. Save classification report to CSV
report_df = pd.DataFrame(classification_rep).transpose()
report_path = os.path.join(plot_directory, "classification_report.csv")
report_df.to_csv(report_path)
print(f"Saved classification report to: {report_path}")
sys.stdout.flush()

# 4. Save confusion matrix to CSV
conf_matrix_path = os.path.join(plot_directory, "confusion_matrix.csv")
conf_matrix_df = pd.DataFrame(conf_matrix, index=['True_0', 'True_1'], columns=['Pred_0', 'Pred_1'])
conf_matrix_df.to_csv(conf_matrix_path)
print(f"Saved confusion matrix CSV to: {conf_matrix_path}")
sys.stdout.flush()

# 5. Save ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Test Set - from weights)')
plt.legend(loc='lower right')

roc_path = os.path.join(plot_directory, 'roc_curve_test_from_weights.png')
plt.savefig(roc_path)
plt.close()
print(f"Saved ROC curve to: {roc_path}")
sys.stdout.flush()

# 6. Save Confusion Matrix Plot
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')

cm_plot_path = os.path.join(plot_directory, 'confusion_matrix.png')
plt.savefig(cm_plot_path)
plt.close()
print(f"Saved confusion matrix plot to: {cm_plot_path}")
sys.stdout.flush()