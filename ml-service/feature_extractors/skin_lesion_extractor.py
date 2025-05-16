import cv2
import numpy as np
from skimage import feature, color, segmentation
from skimage.feature import graycomatrix, graycoprops
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SkinLesionFeatureExtractor:
    def __init__(self):
        self.features = {}

    def calculate_asymmetry(self, img_gray):
        """Calculate asymmetry score (0/1/2)"""
        # Threshold the image
        _, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0

        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Calculate moments
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            return 0

        # Calculate centroid
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Split image into quadrants and compare areas
        height, width = img_gray.shape

        # Vertical split
        left = np.sum(thresh[:, :cx])
        right = np.sum(thresh[:, cx:])
        diff_vertical = abs(left - right) / max(left, right)

        # Horizontal split
        top = np.sum(thresh[:cy, :])
        bottom = np.sum(thresh[cy:, :])
        diff_horizontal = abs(top - bottom) / max(top, bottom)

        # Determine asymmetry score
        if max(diff_vertical, diff_horizontal) > 0.2:
            return 2 if min(diff_vertical, diff_horizontal) > 0.1 else 1
        return 0

    def analyze_pigment_network(self, img_gray):
        """Analyze pigment network (AT/T)"""
        # Apply edge detection
        edges = feature.canny(img_gray, sigma=2)

        # Calculate edge density
        edge_density = np.mean(edges)

        # Analyze pattern regularity using GLCM
        glcm = graycomatrix(img_gray, [1], [0], symmetric=True, normed=True)
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]

        return 'AT' if edge_density > 0.15 and homogeneity < 0.7 else 'T'

    def detect_dots_globules(self, img_gray):
        """
        Detect dots and globules (A/AT/T)
        A - Absent
        AT - Atypical
        T - Typical
        """
        # Apply blob detection
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 5
        params.maxArea = 100

        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(img_gray)

        num_blobs = len(keypoints)

        if num_blobs < 5:  # Very few or no dots/globules
            return 'A'     # Absent

        # Analyze blob distribution and sizes
        if len(keypoints) > 0:
            # Get blob sizes
            sizes = [kp.size for kp in keypoints]
            size_std = np.std(sizes)

            # Get blob positions
            positions = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints])

            # Calculate average distance between blobs
            if len(positions) > 1:
                from scipy.spatial.distance import pdist
                distances = pdist(positions)
                distance_std = np.std(distances)

                # If high variation in sizes or irregular distribution
                if size_std > 5 or distance_std > 50:
                    return 'AT'  # Atypical

        return 'T'  # Typical pattern

    def detect_streaks(self, img_gray):
        """Detect streaks (A/P)"""
        sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)

        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        streak_density = np.mean(gradient_magnitude > np.mean(gradient_magnitude))

        return 'P' if streak_density > 0.1 else 'A'

    def detect_regression_areas(self, img_rgb):
        """Detect regression areas (A/P)"""
        img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        l_channel = img_lab[:,:,0]
        regression_mask = (l_channel > 160) & (np.std(img_rgb, axis=2) < 30)
        regression_ratio = np.sum(regression_mask) / regression_mask.size
        return 'P' if regression_ratio > 0.05 else 'A'

    def detect_blue_whitish_veil(self, img_rgb):
        """Detect blue-whitish veil (A/P)"""
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(img_hsv, lower_blue, upper_blue)
        blue_ratio = np.sum(blue_mask > 0) / blue_mask.size
        return 'P' if blue_ratio > 0.05 else 'A'

    def analyze_colors(self, img_rgb):
        """Analyze presence of specific colors"""
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

        color_ranges = {
            'white': ([0, 0, 200], [180, 30, 255]),
            'red': ([0, 100, 100], [10, 255, 255]),
            'light_brown': ([10, 30, 100], [20, 150, 200]),
            'dark_brown': ([10, 100, 50], [20, 255, 150]),
            'blue_gray': ([100, 30, 50], [130, 150, 200]),
            'black': ([0, 0, 0], [180, 255, 50])
        }

        color_presence = {}
        for color, (lower, upper) in color_ranges.items():
            lower = np.array(lower)
            upper = np.array(upper)
            mask = cv2.inRange(img_hsv, lower, upper)
            ratio = np.sum(mask > 0) / mask.size
            color_presence[color] = ratio > 0.05

        return color_presence

    def extract_features(self, img):
        """Main function to extract all features from the image"""
        logger.info("Extracting skin lesion features from image...")
        
        # Convert PIL Image to cv2 format if needed
        if isinstance(img, Image.Image):
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # Convert to RGB (OpenCV uses BGR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert to grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        try:
            # Get basic features
            self.features = {
                'asymmetry': self.calculate_asymmetry(img_gray),
                'pigment_network': self.analyze_pigment_network(img_gray),
                'dots_globules': self.detect_dots_globules(img_gray),
                'streaks': self.detect_streaks(img_gray),
                'regression_areas': self.detect_regression_areas(img_rgb),
                'blue_whitish_veil': self.detect_blue_whitish_veil(img_rgb)
            }

            # Add color features
            color_features = self.analyze_colors(img_rgb)
            self.features.update(color_features)
            
            logger.info(f"Extracted {len(self.features)} skin lesion features")
        except Exception as e:
            logger.error(f"Error during skin lesion feature extraction: {str(e)}")
            # Return basic features in case of error
            self.features = {
                'error': str(e),
                'asymmetry': 0,
                'pigment_network': 'T',
                'dots_globules': 'A',
                'streaks': 'A',
                'regression_areas': 'A',
                'blue_whitish_veil': 'A',
                'white': False,
                'red': False,
                'light_brown': False,
                'dark_brown': False,
                'blue_gray': False,
                'black': False
            }

        return self.features

def format_features(features):
    """Format features in the required output format"""
    return {
        'Asymmetry': features['asymmetry'],
        'Pigment_Network': features['pigment_network'],
        'Dots_Globules': features['dots_globules'],
        'Streaks': features['streaks'],
        'Regression_Areas': features['regression_areas'],
        'Blue_Whitish_Veil': features['blue_whitish_veil'],
        'White': 'X' if features['white'] else '',
        'Red': 'X' if features['red'] else '',
        'Light_Brown': 'X' if features['light_brown'] else '',
        'Dark_Brown': 'X' if features['dark_brown'] else '',
        'Blue_Gray': 'X' if features['blue_gray'] else '',
        'Black': 'X' if features['black'] else ''
    }
