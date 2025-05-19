#Basic ensemble

import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, RepeatVector, LSTM
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.regularizers import l1_l2
from keras.initializers import glorot_uniform
from sklearn.metrics import (
    classification_report, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_curve, roc_auc_score
)

# Path to the directory where plots will be saved
plot_directory = '/home/u247708/thesis_jwk/data/plots_ensemble_small_v1'
if not os.path.exists(plot_directory):
    os.makedirs(plot_directory)

# ==== Paths ====
image_folder = '/home/u247708/thesis_jwk/data/GAF_images_train_small'
csv_file = '/home/u247708/thesis_jwk/data/gaf_labels_train_small.csv'

# ==== Load and Prepare CSV ====
data = pd.read_csv(csv_file)
data['Price_Movement'] = data['Price_Movement'].astype(str)

# ==== Check for Missing Images ====
non_existing_files = [
    filename for filename in data['filename']
    if not os.path.exists(os.path.join(image_folder, filename))
]
if non_existing_files:
    print("Missing images:", non_existing_files)

# ==== Custom Train/Validation Split ====
indices = np.arange(len(data))
np.random.shuffle(indices)
val_size = int(0.2 * len(data))

train_indices = indices[val_size:]
val_indices = indices[:val_size]

train_df = data.iloc[train_indices].reset_index(drop=True)
val_df = data.iloc[val_indices].reset_index(drop=True)

# ==== Image Generators ====
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    train_df, directory=image_folder,
    x_col='filename', 
    y_col='Price_Movement',
    target_size=(100, 100),
    batch_size=128,
    class_mode='binary', 
    shuffle=False
)

validation_generator = val_datagen.flow_from_dataframe(
    val_df, directory=image_folder,
    x_col='filename', 
    y_col='Price_Movement',
    target_size=(100, 100),
    batch_size=128,
    class_mode='binary', 
    shuffle=True
)

# ==== Data Leakage Check ====
intersection = set(train_df['filename']).intersection(set(val_df['filename']))
print("No data leakage detected." if not intersection else f"Warning: Leakage in {len(intersection)} images")

# ==== CNN Base ====
base_model = InceptionV3(weights=None, include_top=False, input_shape=(100, 100, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu', kernel_regularizer=l1_l2(1e-5, 1e-5), kernel_initializer=glorot_uniform())(x)
x = Dense(256, activation='relu', kernel_regularizer=l1_l2(1e-5, 1e-5), kernel_initializer=glorot_uniform())(x)
x = Dropout(0.2)(x)
x = Dense(256, activation='relu',kernel_regularizer=l1_l2(l1=1e-5, l2=1e-5), kernel_initializer=glorot_uniform())(x)
x = Dropout(0.2)(x)
x = Dense(128, activation='relu', kernel_regularizer=l1_l2(1e-5, 1e-5), kernel_initializer=glorot_uniform())(x)

# ==== LSTM Layer ====
x = RepeatVector(1)(x)
x = LSTM(64, return_sequences=False)(x)

# ==== Output ====
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# ==== Compile ====
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy', 'AUC'])

# ==== Callbacks ====
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# ==== Train ====
history = model.fit(train_generator, validation_data=validation_generator, epochs=50, callbacks=[early_stopping])

# ==== Save the full model for later testing ====
model_path = '/home/u247708/thesis_jwk/data/base_ensemble_small_v1.keras'
model.save(model_path)
print(f"Saved full model to: {model_path}")
sys.stdout.flush()

# Plot training curves
try:
    plt.figure(figsize=(12, 6))

    # Plot training & validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot training & validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    loss_plot_path = os.path.join(plot_directory, 'training_plots2.png')
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"Saved training plots to: {loss_plot_path}")
    sys.stdout.flush()

except Exception as e:
    print(f"Error while plotting training curves: {e}")
    sys.stdout.flush()

# Evaluate model & generate metrics
try:
    y_pred = model.predict(validation_generator)
    y_pred_binary = (y_pred > 0.5).astype(int).flatten()
    true_labels = validation_generator.classes

    fpr, tpr, thresholds = roc_curve(true_labels, y_pred_binary)
    auc_score = roc_auc_score(true_labels, y_pred_binary)
    classification_rep = classification_report(true_labels, y_pred_binary, output_dict=True)

    weighted_precision = classification_rep['weighted avg']['precision']
    weighted_recall = classification_rep['weighted avg']['recall']
    weighted_f1 = classification_rep['weighted avg']['f1-score']
    precision = precision_score(true_labels, y_pred_binary, zero_division=1)
    recall = recall_score(true_labels, y_pred_binary, zero_division=1)
    f1 = f1_score(true_labels, y_pred_binary, zero_division=1)
    mcc = matthews_corrcoef(true_labels, y_pred_binary)
    accuracy = history.history['val_accuracy'][-1]

    # Print and save metrics
    metrics_text = f"""
Validation Metrics:

Accuracy: {accuracy:.4f}
Weighted Precision: {weighted_precision:.4f}
Weighted Recall: {weighted_recall:.4f}
Weighted F1-score: {weighted_f1:.4f}
Precision: {precision:.4f}
Recall: {recall:.4f}
F1-score: {f1:.4f}
Matthews Correlation Coefficient: {mcc:.4f}
AUC Score: {auc_score:.4f}
"""

    print(metrics_text)
    sys.stdout.flush()

    # Save metrics to file
    metrics_path = os.path.join(plot_directory, "metrics_summary.txt")
    with open(metrics_path, "w") as f:
        f.write(metrics_text)
    print(f"Saved metrics summary to: {metrics_path}")
    sys.stdout.flush()

    # Plot ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')

    roc_path = os.path.join(plot_directory, 'roc_curve2.png')
    plt.savefig(roc_path)
    plt.close()
    print(f"Saved ROC curve to: {roc_path}")
    sys.stdout.flush()

except Exception as e:
    print(f"Error during evaluation or plotting ROC: {e}")
    sys.stdout.flush()
