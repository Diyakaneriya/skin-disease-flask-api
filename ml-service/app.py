from flask import Flask, request, jsonify
import os
import logging
from flask_cors import CORS  
from feature_extractors.skin_lesion_extractor import SkinLesionFeatureExtractor, format_features
from utils.image_preprocessing import preprocess_image

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint to check if the service is running"""
    return jsonify({"status": "healthy", "service": "skin-disease-feature-extractor"})

@app.route('/extract-features', methods=['POST'])
def process_image():
    """
    Extract features from the uploaded skin image
    
    Expected JSON payload:
    {
        "image_path": "relative/path/to/image.jpg",
        "user_id": "user_id",
        "image_id": "image_id"
    }
    """
    try:
        data = request.json
        if not data or 'image_path' not in data:
            return jsonify({"error": "Missing required parameters"}), 400
        
        # Get image path from request
        relative_image_path = data['image_path']
        user_id = data.get('user_id')
        image_id = data.get('image_id')
        
        # Convert relative path to absolute path
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        absolute_image_path = os.path.join(base_dir, relative_image_path)
        
        logger.info(f"Processing image at: {absolute_image_path}")
        
        if not os.path.exists(absolute_image_path):
            return jsonify({"error": f"Image file not found at {absolute_image_path}"}), 404
        
        # Preprocess the image
        preprocessed_image = preprocess_image(absolute_image_path)
        
        # Extract skin lesion specific features
        skin_lesion_extractor = SkinLesionFeatureExtractor()
        skin_lesion_features = skin_lesion_extractor.extract_features(preprocessed_image)
        formatted_features = format_features(skin_lesion_features)
        
        # Return the extracted features
        response = {
            "success": True,
            "user_id": user_id,
            "image_id": image_id,
            "features": formatted_features
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)
