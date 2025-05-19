# Create transfer learning weights

import pandas as pd
import tensorflow 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from keras.applications import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Dropout
from sklearn.metrics import classification_report, matthews_corrcoef, f1_score
from sklearn.metrics import precision_score, recall_score, f1_score
from keras.callbacks import EarlyStopping
from keras.regularizers import l1_l2
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from keras.initializers import glorot_uniform
from sklearn.model_selection import train_test_split
import os
import numpy as np
import sys


# Path to the directory where plots will be saved
plot_directory = '/home/u247708/thesis_jwk/data/plots_cnn_large_v1'
if not os.path.exists(plot_directory):
    os.makedirs(plot_directory)

# Paths to image folder and CSV file
image_folder = '/home/u247708/thesis_jwk/data/GAF_images_train_large'
csv_file = '/home/u247708/thesis_jwk/data/gaf_labels_train_large.csv'

# Load CSV file
data = pd.read_csv(csv_file)

# Convert labels to strings
data['Price_Movement'] = data['Price_Movement'].astype(str)

# Shuffle the indices of the dataset
indices = np.arange(len(data))
np.random.shuffle(indices)

# Determine the size of the validation set
val_size = int(0.2 * len(data)) 

# Split the shuffled indices into training and validation indices
train_indices = indices[val_size:]
val_indices = indices[:val_size]

# Split the dataset into training and validation sets using the shuffled indices
train_df = data.iloc[train_indices]
val_df = data.iloc[val_indices]

# Reset index to maintain order
train_df.reset_index(drop=True, inplace=True)
val_df.reset_index(drop=True, inplace=True)

# Initialize the ImageDataGenerator for training and validation (you can include data augmentation parameters in the training generator)
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

# Flow from DataFrame using the ImageDataGenerator
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=image_folder,
    x_col='filename',
    y_col='Price_Movement',
    target_size=(100, 100),
    batch_size=128,
    class_mode='binary',
    shuffle=False
)

validation_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=image_folder,
    x_col='filename',
    y_col='Price_Movement',
    target_size=(100, 100),
    batch_size=128,
    class_mode='binary',
    shuffle=True
)

train_images = set(train_generator.filenames)
val_images = set(validation_generator.filenames)

# Check for data leakage
train_set = set(train_df['filename'])
val_set = set(val_df['filename'])
intersection = train_set.intersection(val_set)

if intersection:
    print(f"Warning: Data leakage detected. {len(intersection)} images found in both training and validation sets.")
    print("Common images:", intersection)
else:
    print("No data leakage detected.")

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

# Define Early Stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model with Early Stopping
history = model.fit(train_generator, validation_data=validation_generator, epochs=50, callbacks=[early_stopping])

# Save the weights of the trained model
model.save_weights('/home/u247708/thesis_jwk/data/cnn_train_large_v1.weights.h5')

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
