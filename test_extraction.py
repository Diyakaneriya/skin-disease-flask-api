import os
import sys
from utils.image_preprocessing import preprocess_image
from feature_extractors.extractor import extract_features
import json

def test_feature_extraction(image_path):
    """
    Test feature extraction on a single image
    """
    print(f"Testing feature extraction on image: {image_path}")
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return
    
    try:
        # Preprocess the image
        print("Preprocessing image...")
        processed_image = preprocess_image(image_path)
        
        # Extract features
        print("Extracting features...")
        features = extract_features(processed_image)
        
        # Print features summary
        print("\nFeature extraction successful!")
        print(f"Total features extracted: {len(features)}")
        
        # Print some features
        print("\nSample features:")
        for i, (key, value) in enumerate(features.items()):
            if i < 5:  # Only show first 5 features
                print(f"  {key}: {value if not isinstance(value, list) else '(list with ' + str(len(value)) + ' elements)'}")
            else:
                break
        
        print("\nFeature extraction completed successfully!")
        return features
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Check if image path is provided as command line argument
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Look for sample images in the uploads folder
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        uploads_dir = os.path.join(base_dir, "skin-disease-backend", "uploads")
        
        if os.path.exists(uploads_dir):
            # Get the first image in the uploads folder
            image_files = [f for f in os.listdir(uploads_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            
            if image_files:
                image_path = os.path.join(uploads_dir, image_files[0])
                print(f"Found image in uploads folder: {image_files[0]}")
            else:
                print(f"No images found in uploads folder: {uploads_dir}")
                print("Please provide an image path as command line argument")
                sys.exit(1)
        else:
            print(f"Uploads folder not found: {uploads_dir}")
            print("Please provide an image path as command line argument")
            sys.exit(1)
    
    # Run the test
    test_feature_extraction(image_path)
