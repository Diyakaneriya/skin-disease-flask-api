#!/usr/bin/env python
"""
Unified script for processing skin images:
1. Extracts features from skin images
2. Performs classification using the skin lesion classifier model

Usage:
    python process_image.py <image_path> [<output_json_path>]

If output_json_path is provided, results will be written to that file.
Otherwise, results will be printed to stdout.
"""

import os
import sys
import json
import base64
import io
import tensorflow as tf
from PIL import Image
import numpy as np
from utils.image_preprocessing import preprocess_image
from feature_extractors.skin_lesion_extractor import SkinLesionFeatureExtractor, format_features
from models.skin_lesion_classifier import DIAGNOSIS_MAPPING, IMAGE_SIZE, make_gradcam_heatmap

# Disable GPU memory growth to avoid memory errors
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def load_classifier_model():
    """Load the trained skin lesion classifier model"""
    try:
        # Custom metrics for model loading
        def top_2_accuracy(y_true, y_pred):
            return tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=2)
        
        def top_3_accuracy(y_true, y_pred):
            return tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)
        
        # Get the model path - adjust this path if needed
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                 'ml-service', 'models', 'best_model.h5')
        
        # Load the model with custom metrics
        model = tf.keras.models.load_model(model_path, custom_objects={
            'top_2_accuracy': top_2_accuracy,
            'top_3_accuracy': top_3_accuracy
        })
        
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def generate_gradcam(model, image_path, pred_index=None):
    """Generate Grad-CAM visualization for the image"""
    try:
        # Load and preprocess the image
        img = Image.open(image_path)
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
        img_array = np.array(img)
        original_img = img_array.copy()
        
        # Expand dimensions and preprocess for MobileNetV2
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        
        # Generate heatmap
        last_conv_layer_name = "out_relu"  # Specific to MobileNetV2
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index)
        
        # Convert heatmap to RGB
        heatmap = np.uint8(255 * heatmap)
        jet = tf.keras.utils.get_colormap('jet')
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]
        
        # Create superimposed image
        jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((original_img.shape[1], original_img.shape[0]))
        jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
        
        superimposed_img = jet_heatmap * 0.4 + original_img
        superimposed_img = np.uint8(superimposed_img)
        
        # Convert the superimposed image to base64 for sending to frontend
        superimposed_pil = Image.fromarray(superimposed_img)
        buffered = io.BytesIO()
        superimposed_pil.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Also convert original image to base64 for comparison
        original_pil = Image.fromarray(original_img)
        original_buffered = io.BytesIO()
        original_pil.save(original_buffered, format="PNG")
        original_img_str = base64.b64encode(original_buffered.getvalue()).decode()
        
        return {
            "success": True,
            "gradcam_image": img_str,
            "original_image": original_img_str
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def classify_image(model, image_path):
    """Classify a skin lesion image using the loaded model"""
    if model is None:
        return {
            "success": False,
            "error": "Model could not be loaded"
        }
    
    try:
        # Load and preprocess the image for classification
        img = Image.open(image_path)
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
        img_array = np.array(img)
        
        # Expand dimensions to match model input shape
        img_array = np.expand_dims(img_array, axis=0)
        
        # Preprocess for MobileNetV2
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        
        # Make prediction
        predictions = model.predict(img_array)
        
        # Get class names - we need to determine class indices
        # This is a simplified approach; in production, you should save class indices with the model
        class_indices = {
            0: 'nv',      # Melanocytic nevi
            1: 'mel',     # Melanoma
            2: 'bkl',     # Benign keratosis
            3: 'bcc',     # Basal cell carcinoma
            4: 'akiec',   # Actinic Keratosis
            5: 'vasc',    # Vascular lesion
            6: 'df'       # Dermatofibroma
        }
        
        class_names = {v: k for k, v in class_indices.items()}
        
        # Get top 3 predictions
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_3_classes = [class_indices[i] for i in top_3_indices]
        top_3_probabilities = [float(predictions[0][i]) for i in top_3_indices]  # Convert to float for JSON serialization
        
        # Format classification results
        classification_results = []
        for i in range(len(top_3_classes)):
            class_code = top_3_classes[i]
            classification_results.append({
                "rank": i + 1,
                "class_code": class_code,
                "class_name": DIAGNOSIS_MAPPING[class_code],
                "probability": top_3_probabilities[i],
                "confidence_percent": round(top_3_probabilities[i] * 100, 2)
            })
        
        # Determine confidence level and recommendation
        top_confidence = top_3_probabilities[0]
        recommendation = ""
        
        if top_confidence > 0.85:
            confidence_level = "high"
            if top_3_classes[0] == 'mel':
                recommendation = "WARNING: High likelihood of melanoma (skin cancer). Please consult a healthcare professional immediately."
            elif top_3_classes[0] in ['bcc', 'akiec']:
                recommendation = "This image shows characteristics consistent with a potentially concerning skin condition. Recommendation: Consult with a dermatologist for proper evaluation."
            else:
                recommendation = f"This image shows clear characteristics of {DIAGNOSIS_MAPPING[top_3_classes[0]]}. While this prediction appears to be benign, always consult a healthcare professional for proper diagnosis."
        elif top_confidence > 0.65:
            confidence_level = "medium"
            recommendation = f"This image shows some characteristic features but with moderate certainty. Possibilities include {DIAGNOSIS_MAPPING[top_3_classes[0]]} and {DIAGNOSIS_MAPPING[top_3_classes[1]]}. Recommendation: Consult with a healthcare professional for proper evaluation."
        else:
            confidence_level = "low"
            recommendation = "This image is challenging to classify with certainty. Consult with a dermatologist for proper evaluation."
        
        # Generate Grad-CAM visualization for the top prediction
        gradcam_result = generate_gradcam(model, image_path, top_3_indices[0])
        
        return {
            "success": True,
            "classification": classification_results,
            "confidence_level": confidence_level,
            "recommendation": recommendation,
            "disclaimer": "This AI analysis is for educational purposes only and does not replace professional medical advice.",
            "gradcam": gradcam_result
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def main():
    # Check command-line arguments
    if len(sys.argv) < 2:
        print("Error: Image path is required.")
        print(f"Usage: python {sys.argv[0]} <image_path> [<output_json_path>]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Check if the image exists
    if not os.path.exists(image_path):
        error_result = {
            "success": False,
            "image_path": image_path,
            "error": f"Image file not found at {image_path}"
        }
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(error_result, f, indent=2)
        else:
            print(json.dumps(error_result))
        
        sys.exit(1)
    
    try:
        # 1. Preprocess the image
        processed_image = preprocess_image(image_path)
        
        # 2. Extract skin lesion specific features
        skin_lesion_extractor = SkinLesionFeatureExtractor()
        skin_lesion_features = skin_lesion_extractor.extract_features(processed_image)
        formatted_features = format_features(skin_lesion_features)
        
        # 3. Load the classifier model
        model = load_classifier_model()
        
        # 4. Classify the image
        classification_result = classify_image(model, image_path)
        
        # 5. Format the combined result
        result = {
            "success": True,
            "image_path": image_path,
            "features": formatted_features,
            "classification": classification_result if classification_result["success"] else {
                "success": False,
                "error": classification_result.get("error", "Unknown classification error")
            }
        }
        
        # Output the result
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Results written to {output_path}")
        else:
            print(json.dumps(result))
        
        sys.exit(0)
    
    except Exception as e:
        error_result = {
            "success": False,
            "image_path": image_path,
            "error": str(e)
        }
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(error_result, f, indent=2)
            print(f"Error written to {output_path}")
        else:
            print(json.dumps(error_result))
        
        sys.exit(1)

if __name__ == "__main__":
    main()
