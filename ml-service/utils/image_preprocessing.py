import cv2
import numpy as np
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def preprocess_image(image_path):
    """
    Preprocess a skin image for feature extraction.
    
    Parameters:
    -----------
    image_path : str
        Path to the image file
    
    Returns:
    --------
    numpy.ndarray
        Preprocessed image ready for feature extraction
    """
    logger.info(f"Preprocessing image: {image_path}")
    
    try:
        # Check if file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image file: {image_path}")
        
        # Convert to RGB (OpenCV reads as BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image to standard size (adjust size as needed)
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
        
        # Remove hair (optional - uncomment if needed)
        # image = remove_hair(image)
        
        # Apply color correction (adjust as needed)
        # image = apply_color_correction(image)
        
        # Apply contrast enhancement
        image = enhance_contrast(image)
        
        logger.info("Image preprocessing completed successfully")
        return image
    
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise

def enhance_contrast(image):
    """
    Enhance contrast of the image using CLAHE
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image
    
    Returns:
    --------
    numpy.ndarray
        Contrast enhanced image
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    # Split channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # Merge channels
    enhanced_lab = cv2.merge((cl, a, b))
    
    # Convert back to RGB
    enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    
    return enhanced_rgb

def remove_hair(image):
    """
    Remove hair from skin images
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image
    
    Returns:
    --------
    numpy.ndarray
        Image with hair removed
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Create a kernel for morphological filtering
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    
    # Morphological blackhat operation
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    
    # Threshold to create a binary mask
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    
    # Inpaint using the mask
    result = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    
    return result

def apply_color_correction(image):
    """
    Apply color correction to handle different lighting conditions
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image
    
    Returns:
    --------
    numpy.ndarray
        Color corrected image
    """
    # Split into RGB channels
    r, g, b = cv2.split(image)
    
    # Apply histogram equalization to each channel
    r_eq = cv2.equalizeHist(r)
    g_eq = cv2.equalizeHist(g)
    b_eq = cv2.equalizeHist(b)
    
    # Merge equalized channels
    image_eq = cv2.merge((r_eq, g_eq, b_eq))
    
    return image_eq

def segment_lesion(image):
    """
    Segment the skin lesion from the background
    
    Parameters:
    -----------
    image : numpy.ndarray
        Input image
    
    Returns:
    --------
    tuple
        (segmented_image, mask)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Otsu's thresholding
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a mask
    mask = np.zeros_like(thresh)
    
    # If contours are found, take the largest one
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [max_contour], 0, 255, -1)
    
    # Apply mask to original image
    segmented = image.copy()
    segmented[mask == 0] = [0, 0, 0]
    
    return segmented, mask
