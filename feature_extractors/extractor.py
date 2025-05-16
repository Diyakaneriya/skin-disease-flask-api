import numpy as np
import cv2
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_features(image):
    """
    Extract features from a skin image.
    This is a placeholder implementation that you should replace with your actual feature extraction code.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Preprocessed image in RGB format
    
    Returns:
    --------
    dict
        Dictionary containing the extracted features
    """
    logger.info("Extracting features from image...")
    
    # This is a placeholder. Replace with your actual feature extraction code.
    # Example features that might be extracted:
    
    # 1. Color features
    features = {}
    
    # Calculate color histograms
    try:
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Calculate histograms for each channel
        hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256])
        
        hist_l = cv2.calcHist([lab], [0], None, [32], [0, 256])
        hist_a = cv2.calcHist([lab], [1], None, [32], [0, 256])
        hist_b = cv2.calcHist([lab], [2], None, [32], [0, 256])
        
        # Normalize histograms
        cv2.normalize(hist_h, hist_h, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist_s, hist_s, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist_v, hist_v, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist_l, hist_l, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist_a, hist_a, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist_b, hist_b, 0, 1, cv2.NORM_MINMAX)
        
        # Add histograms to features
        features['hist_h'] = hist_h.flatten().tolist()
        features['hist_s'] = hist_s.flatten().tolist()
        features['hist_v'] = hist_v.flatten().tolist()
        features['hist_l'] = hist_l.flatten().tolist()
        features['hist_a'] = hist_a.flatten().tolist()
        features['hist_b'] = hist_b.flatten().tolist()
        
        # 2. Color statistics
        colors = ['red', 'green', 'blue']
        for i, color in enumerate(colors):
            features[f'mean_{color}'] = float(np.mean(image[:,:,i]))
            features[f'std_{color}'] = float(np.std(image[:,:,i]))
            features[f'min_{color}'] = int(np.min(image[:,:,i]))
            features[f'max_{color}'] = int(np.max(image[:,:,i]))
        
        # 3. Texture features (using Haralick features as an example)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        glcm = calculate_glcm(gray)
        
        # Calculate Haralick features
        haralick_features = calculate_haralick(glcm)
        for i, feature_name in enumerate([
            'contrast', 'dissimilarity', 'homogeneity', 'energy', 
            'correlation', 'ASM'
        ]):
            if i < len(haralick_features):
                features[f'haralick_{feature_name}'] = float(haralick_features[i])
        
        # 4. Shape features
        # These would typically be extracted after segmentation
        # For now, just adding placeholder values
        features['asymmetry'] = 0.5  # placeholder
        features['border_irregularity'] = 0.3  # placeholder
        features['diameter'] = 15.0  # placeholder in mm
        
        logger.info(f"Extracted {len(features)} features")
        
    except Exception as e:
        logger.error(f"Error during feature extraction: {str(e)}")
        # Return basic features in case of error
        features = {
            'error': str(e),
            'mean_red': 0,
            'mean_green': 0,
            'mean_blue': 0
        }
    
    return features

def calculate_glcm(gray_image, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256):
    """
    Calculate Gray-Level Co-occurrence Matrix for texture analysis.
    
    Parameters:
    -----------
    gray_image : numpy.ndarray
        Grayscale image
    distances : list
        List of distance values
    angles : list
        List of angle values in radians
    levels : int
        Number of gray levels
        
    Returns:
    --------
    numpy.ndarray
        GLCM matrix
    """
    # Normalize image to reduce the number of intensity values
    gray_image = (gray_image / 16).astype(np.uint8)
    levels = 16  # Reduced levels
    
    # Calculate GLCM
    glcm = np.zeros((levels, levels, len(distances), len(angles)), dtype=np.float32)
    
    for i, d in enumerate(distances):
        for j, a in enumerate(angles):
            # Define offset based on distance and angle
            dx = int(d * np.cos(a))
            dy = int(d * np.sin(a))
            
            # Create shifted image
            rows, cols = gray_image.shape
            shifted_image = np.zeros_like(gray_image)
            
            # Only consider valid indices
            valid_rows = np.arange(rows - abs(dy)) if dy > 0 else np.arange(abs(dy), rows)
            valid_cols = np.arange(cols - abs(dx)) if dx > 0 else np.arange(abs(dx), cols)
            
            # Create shifted image
            if dy > 0:
                if dx > 0:
                    shifted_image[0:rows-dy, 0:cols-dx] = gray_image[dy:rows, dx:cols]
                elif dx < 0:
                    shifted_image[0:rows-dy, abs(dx):cols] = gray_image[dy:rows, 0:cols-abs(dx)]
                else:
                    shifted_image[0:rows-dy, :] = gray_image[dy:rows, :]
            elif dy < 0:
                if dx > 0:
                    shifted_image[abs(dy):rows, 0:cols-dx] = gray_image[0:rows-abs(dy), dx:cols]
                elif dx < 0:
                    shifted_image[abs(dy):rows, abs(dx):cols] = gray_image[0:rows-abs(dy), 0:cols-abs(dx)]
                else:
                    shifted_image[abs(dy):rows, :] = gray_image[0:rows-abs(dy), :]
            else:
                if dx > 0:
                    shifted_image[:, 0:cols-dx] = gray_image[:, dx:cols]
                elif dx < 0:
                    shifted_image[:, abs(dx):cols] = gray_image[:, 0:cols-abs(dx)]
            
            # Calculate co-occurrence matrix
            for r in range(rows):
                for c in range(cols):
                    if 0 <= r + dy < rows and 0 <= c + dx < cols:
                        i_val = gray_image[r, c]
                        j_val = gray_image[r+dy, c+dx]
                        glcm[i_val, j_val, i, j] += 1
    
    # Normalize GLCM
    for i in range(len(distances)):
        for j in range(len(angles)):
            glcm[:, :, i, j] = glcm[:, :, i, j] / np.sum(glcm[:, :, i, j])
    
    return glcm

def calculate_haralick(glcm):
    """
    Calculate Haralick texture features from GLCM.
    
    Parameters:
    -----------
    glcm : numpy.ndarray
        Gray-Level Co-occurrence Matrix
        
    Returns:
    --------
    list
        List of Haralick features
    """
    # Average over all directions
    glcm_avg = np.mean(glcm, axis=(2, 3))
    
    # Get dimensions
    num_levels = glcm_avg.shape[0]
    
    # Create normalized GLCM
    glcm_sum = np.sum(glcm_avg)
    if glcm_sum > 0:
        glcm_norm = glcm_avg / glcm_sum
    else:
        return [0] * 6  # Return zeros if GLCM is empty
    
    # Create matrices for calculations
    i_indices, j_indices = np.meshgrid(np.arange(num_levels), np.arange(num_levels), indexing='ij')
    
    # Calculate features
    # 1. Contrast
    contrast = np.sum(glcm_norm * ((i_indices - j_indices) ** 2))
    
    # 2. Dissimilarity
    dissimilarity = np.sum(glcm_norm * np.abs(i_indices - j_indices))
    
    # 3. Homogeneity
    homogeneity = np.sum(glcm_norm / (1 + (i_indices - j_indices) ** 2))
    
    # 4. Energy / Angular Second Moment (ASM)
    asm = np.sum(glcm_norm ** 2)
    energy = np.sqrt(asm)
    
    # 5. Correlation
    i_mean = np.sum(i_indices * glcm_norm)
    j_mean = np.sum(j_indices * glcm_norm)
    i_var = np.sum(glcm_norm * ((i_indices - i_mean) ** 2))
    j_var = np.sum(glcm_norm * ((j_indices - j_mean) ** 2))
    
    if i_var > 0 and j_var > 0:
        correlation = np.sum(glcm_norm * (i_indices - i_mean) * (j_indices - j_mean)) / np.sqrt(i_var * j_var)
    else:
        correlation = 0
    
    return [contrast, dissimilarity, homogeneity, energy, correlation, asm]
