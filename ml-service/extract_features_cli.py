#!/usr/bin/env python
"""
Command-line script for extracting features from skin images.
This can be called directly from Node.js using child_process.

Usage:
    python extract_features_cli.py <image_path> [<output_json_path>]

If output_json_path is provided, results will be written to that file.
Otherwise, results will be printed to stdout.
"""

import os
import sys
import json
from utils.image_preprocessing import preprocess_image
from feature_extractors.skin_lesion_extractor import SkinLesionFeatureExtractor, format_features

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
        print(f"Error: Image file not found at {image_path}")
        sys.exit(1)
    
    try:
        # Preprocess the image
        processed_image = preprocess_image(image_path)
        
        # Extract skin lesion specific features
        skin_lesion_extractor = SkinLesionFeatureExtractor()
        skin_lesion_features = skin_lesion_extractor.extract_features(processed_image)
        formatted_features = format_features(skin_lesion_features)
        
        # Format the result
        result = {
            "success": True,
            "image_path": image_path,
            "features": formatted_features
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
