# HAM10000 Skin Lesion Classifier with Image Testing Capability
# Optimized version for faster training and better metadata handling
# Enhanced with Grad-CAM visualization for explainable AI

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import itertools

# TensorFlow imports with memory optimization
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy
from PIL import Image
from matplotlib import cm

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define constants - optimized for speed and accuracy
IMAGE_SIZE = 192  # Reduced from 224 for faster processing
BATCH_SIZE = 32   # Increased for better parallelization
EPOCHS = 10       # Reduced with early stopping to prevent overfitting

# Paths - modify these to match your directory structure
BASE_DIR = 'skin_model'
DATA_DIR = 'D:/archive (1)'  # Change this to where your HAM10000 dataset is located
METADATA_PATH = os.path.join(DATA_DIR, 'HAM10000_metadata.csv')
IMAGES_PART1_PATH = os.path.join(DATA_DIR, 'HAM10000_images_part_1')
IMAGES_PART2_PATH = os.path.join(DATA_DIR, 'HAM10000_images_part_2')

# Map for diagnosis codes to full names
DIAGNOSIS_MAPPING = {
    'nv': 'Melanocytic nevi (benign mole)',
    'mel': 'Melanoma (malignant skin cancer)',
    'bkl': 'Benign keratosis (benign skin condition)',
    'bcc': 'Basal cell carcinoma (common skin cancer)',
    'akiec': 'Actinic Keratosis / Bowens disease (pre-cancerous)',
    'vasc': 'Vascular lesion (benign vascular condition)',
    'df': 'Dermatofibroma (benign skin lesion)'
}

# Grad-CAM functions
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Generate Grad-CAM heatmap for a given image and model"""
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def display_gradcam(image_path, model, last_conv_layer_name, pred_index=None):
    """Display Grad-CAM visualization for an image"""
    img = Image.open(image_path)
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = np.array(img)
    original_img = img_array.copy()
    
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index)
    
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((original_img.shape[1], original_img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
    
    superimposed_img = jet_heatmap * 0.4 + original_img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
    
    return superimposed_img

# [Rest of your original functions remain exactly the same until test_single_image]

def test_single_image(model, image_path, class_indices, show_gradcam=True):
    """Test a single image using the trained model with AI explanation and Grad-CAM"""
    # Preprocess the image
    img = Image.open(image_path)
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = np.array(img)
    
    # Expand dimensions to match model input shape
    img_array = np.expand_dims(img_array, axis=0)
    
    # Preprocess for MobileNetV2
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    
    # Make prediction
    predictions = model.predict(img_array)
    
    # Get class names
    class_names = {v: k for k, v in class_indices.items()}
    
    # Get top 3 predictions
    top_3_indices = np.argsort(predictions[0])[-3:][::-1]
    top_3_classes = [class_names[i] for i in top_3_indices]
    top_3_probabilities = [predictions[0][i] for i in top_3_indices]
    
    # Display results
    print("\nPrediction Results:")
    print("------------------")
    for i in range(3):
        print(f"{i+1}. {top_3_classes[i]} ({DIAGNOSIS_MAPPING[top_3_classes[i]]}): {top_3_probabilities[i]*100:.2f}%")
    
    # AI-powered explanation
    top_class = top_3_classes[0]
    confidence = top_3_probabilities[0]
    
    print("\n=== AI-Powered Analysis ===")
    
    if confidence > 0.85:
        print(f"High confidence prediction: {top_class} ({DIAGNOSIS_MAPPING[top_class]})")
        if top_class == 'mel':
            print("WARNING: This image shows high likelihood of melanoma (skin cancer).")
            print("Please consult a healthcare professional immediately.")
        elif top_class in ['bcc', 'akiec']:
            print("This image shows characteristics consistent with a potentially concerning skin condition.")
            print("Recommendation: Consult with a dermatologist for proper evaluation.")
        else:
            print(f"This image shows clear characteristics of {DIAGNOSIS_MAPPING[top_class]}.")
            print("While this prediction appears to be benign, always consult a healthcare professional for proper diagnosis.")
    elif confidence > 0.65:
        print(f"Medium confidence prediction: {top_class} ({DIAGNOSIS_MAPPING[top_class]})")
        print(f"Second possibility: {top_3_classes[1]} ({DIAGNOSIS_MAPPING[top_3_classes[1]]})")
        print("This image shows some characteristic features but with moderate certainty.")
        print("Recommendation: Consult with a healthcare professional for proper evaluation.")
    else:
        print("Low confidence prediction. This image is challenging to classify with certainty.")
        print(f"Possibilities include: {', '.join([f'{c} ({DIAGNOSIS_MAPPING[c]})' for c in top_3_classes])}")
        print("Recommendation: Consult with a dermatologist for proper evaluation.")
    
    print("\nDISCLAIMER: This AI analysis is for educational purposes only and does not replace professional medical advice.")
    
    # Display the image with prediction and Grad-CAM
    if show_gradcam:
        last_conv_layer_name = "out_relu"  # Specific to MobileNetV2
        
        try:
            superimposed_img = display_gradcam(image_path, model, last_conv_layer_name)
            
            plt.figure(figsize=(16, 8))
            
            plt.subplot(1, 2, 1)
            img = Image.open(image_path)
            plt.imshow(img)
            plt.title(f"Original Image\nPrediction: {top_class} ({top_3_probabilities[0]*100:.2f}%)")
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(superimposed_img)
            plt.title(f"Grad-CAM Visualization\nAreas influencing the prediction")
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"\nCould not generate Grad-CAM visualization: {e}")
            plt.figure(figsize=(8, 8))
            img = Image.open(image_path)
            plt.imshow(img)
            plt.title(f"Prediction: {top_class} ({top_3_probabilities[0]*100:.2f}%)")
            plt.axis('off')
            plt.show()
    else:
        plt.figure(figsize=(8, 8))
        img = Image.open(image_path)
        plt.imshow(img)
        plt.title(f"Prediction: {top_class} ({top_3_probabilities[0]*100:.2f}%)")
        plt.axis('off')
        plt.show()
    
    return top_3_classes, top_3_probabilities

def interactive_testing(model, class_indices):
    """Allow user to input image paths for testing"""
    while True:
        print("\n=== Skin Lesion Classifier: Image Testing Interface ===")
        print("This tool analyzes images of skin lesions and predicts their diagnosis.")
        print("DISCLAIMER: This is for educational purposes only and not a medical device.")
        
        image_path = input("\nEnter path to image file (or 'q' to quit): ")
        
        if image_path.lower() == 'q':
            break
        
        if not os.path.exists(image_path):
            print(f"Error: File {image_path} not found.")
            continue
        
        try:
            show_gradcam = input("Show Grad-CAM visualization? (y/n): ").lower() == 'y'
            test_single_image(model, image_path, class_indices, show_gradcam)
        except Exception as e:
            print(f"Error processing image: {e}")

# [All other original functions remain exactly the same]

def main():
    """Main function to orchestrate the entire process"""
    print("=== HAM10000 Skin Lesion Classifier ===")
    print("Optimized version for faster training and higher accuracy")
    print("Now with Grad-CAM visualization for explainable AI")
    
    # Check if model already exists
    model_path = os.path.join(BASE_DIR, 'best_model.h5')
    
    if os.path.exists(model_path):
        print(f"Found existing model at {model_path}")
        retrain = input("Do you want to retrain the model? (y/n): ").lower() == 'y'
    else:
        retrain = True
    
    if retrain:
        train_dir, val_dir = create_directory_structure()
        df_train, df_val = prepare_data(train_dir, val_dir)
        augment_data(train_dir)
        model, validation_generator = create_and_train_model(train_dir, val_dir)
        class_indices = evaluate_model(model, validation_generator)
    else:
        def top_2_accuracy(y_true, y_pred):
            return top_k_categorical_accuracy(y_true, y_pred, k=2)
        
        def top_3_accuracy(y_true, y_pred):
            return top_k_categorical_accuracy(y_true, y_pred, k=3)
        
        try:
            model = load_model(model_path, custom_objects={
                'top_2_accuracy': top_2_accuracy,
                'top_3_accuracy': top_3_accuracy
            })
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Creating and training a new model...")
            train_dir, val_dir = create_directory_structure()
            df_train, df_val = prepare_data(train_dir, val_dir)
            augment_data(train_dir)
            model, validation_generator = create_and_train_model(train_dir, val_dir)
            class_indices = evaluate_model(model, validation_generator)
            return
        
        val_dir = os.path.join(BASE_DIR, 'val_dir')
        if not os.path.exists(val_dir):
            print("Validation directory not found. Creating directory structure...")
            train_dir, val_dir = create_directory_structure()
            df_train, df_val = prepare_data(train_dir, val_dir)
        
        val_datagen = ImageDataGenerator(
            preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
        )
        validation_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=(IMAGE_SIZE, IMAGE_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        )
        class_indices = validation_generator.class_indices
    
    print("\nModel is ready for testing!")
    interactive_testing(model, class_indices)
    
    print("\nDone!")

if __name__ == "__main__":
    main()