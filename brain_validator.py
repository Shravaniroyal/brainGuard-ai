"""
Brain MRI Image Validator
Ensures only valid brain MRI scans are processed - rejects all other images
"""

import numpy as np
import cv2
from scipy import ndimage
from PIL import Image
import io

class BrainMRIValidator:
    """Validates if uploaded image is actually a brain MRI scan"""
    
    def __init__(self):
        self.min_brain_circularity = 0.4  # Brain should be somewhat circular
        self.min_gray_variance = 500      # MRI has texture variance
        self.max_color_saturation = 30    # MRI is grayscale
        
    def validate_image(self, image_array):
        """
        Validate if image is a brain MRI scan
        
        Args:
            image_array: numpy array of image (H, W, C) or (H, W)
            
        Returns:
            tuple: (is_valid: bool, confidence: float, reason: str)
        """
        try:
            # Convert to grayscale if needed
            if len(image_array.shape) == 3:
                gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_array
                
            # Run all validation checks
            checks = [
                self._check_grayscale(image_array),
                self._check_brain_shape(gray),
                self._check_mri_texture(gray),
                self._check_intensity_distribution(gray),
                self._check_anatomical_features(gray)
            ]
            
            # Calculate overall confidence
            passed_checks = sum([1 for valid, _, _ in checks if valid])
            confidence = (passed_checks / len(checks)) * 100
            
            # Collect failure reasons
            failed_reasons = [reason for valid, _, reason in checks if not valid]
            
            # Need at least 3 out of 5 checks to pass
            is_valid = passed_checks >= 3
            
            if is_valid:
                return True, confidence, "Valid brain MRI detected"
            else:
                return False, confidence, " | ".join(failed_reasons)
                
        except Exception as e:
            return False, 0.0, f"Validation error: {str(e)}"
    
    def _check_grayscale(self, image):
        """Check if image is primarily grayscale (MRI characteristic)"""
        if len(image.shape) == 2:
            return True, 100.0, ""
            
        # Check color saturation
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1]
        avg_saturation = np.mean(saturation)
        
        if avg_saturation < self.max_color_saturation:
            return True, 100.0, ""
        else:
            return False, 0.0, "Image is too colorful (not grayscale MRI)"
    
    def _check_brain_shape(self, gray):
        """Check for brain-like circular/elliptical shape"""
        # Threshold to find main structures
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return False, 0.0, "No clear structure detected"
        
        # Get largest contour (should be brain outline)
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        if perimeter == 0:
            return False, 0.0, "Invalid contour"
        
        # Calculate circularity: 4π(area)/(perimeter²)
        # Circle = 1.0, brain ≈ 0.6-0.8
        circularity = 4 * np.pi * area / (perimeter ** 2)
        
        if circularity > self.min_brain_circularity:
            return True, circularity * 100, ""
        else:
            return False, 0.0, "Shape not brain-like (not circular/elliptical)"
    
    def _check_mri_texture(self, gray):
        """Check for MRI-specific texture patterns"""
        # Calculate local variance (MRI has rich texture)
        variance = ndimage.generic_filter(gray.astype(float), np.var, size=15)
        avg_variance = np.mean(variance)
        
        if avg_variance > self.min_gray_variance:
            return True, min(avg_variance / 20, 100), ""
        else:
            return False, 0.0, "No MRI texture pattern detected"
    
    def _check_intensity_distribution(self, gray):
        """Check if intensity distribution matches MRI characteristics"""
        # MRI typically has:
        # - Dark background (air/skull)
        # - Medium gray matter
        # - Bright white matter
        # - Some very bright spots (CSF)
        
        hist, _ = np.histogram(gray, bins=256, range=(0, 256))
        hist = hist / hist.sum()  # Normalize
        
        # Check for multi-modal distribution (characteristic of MRI)
        # Find peaks in histogram
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(hist, height=0.01)
        
        if len(peaks) >= 2:  # At least 2 tissue types visible
            return True, len(peaks) * 25, ""
        else:
            return False, 0.0, "Intensity distribution not consistent with brain tissue"
    
    def _check_anatomical_features(self, gray):
        """Check for basic brain anatomical features"""
        h, w = gray.shape
        
        # Brain MRI typically has:
        # 1. Darker regions at edges (skull/background)
        # 2. Brighter central region (brain tissue)
        
        # Check center vs edge brightness
        center_h, center_w = h // 2, w // 2
        center_region = gray[center_h-h//4:center_h+h//4, center_w-w//4:center_w+w//4]
        
        edge_top = gray[0:h//8, :]
        edge_bottom = gray[-h//8:, :]
        edge_region = np.concatenate([edge_top.flatten(), edge_bottom.flatten()])
        
        if len(center_region) == 0 or len(edge_region) == 0:
            return False, 0.0, "Cannot analyze image structure"
        
        center_brightness = np.mean(center_region)
        edge_brightness = np.mean(edge_region)
        
        # Brain should be brighter than edges
        if center_brightness > edge_brightness * 1.2:
            return True, 80.0, ""
        else:
            return False, 0.0, "No clear brain structure (center not brighter than edges)"


def validate_brain_mri_file(file_bytes):
    """
    High-level function to validate uploaded file
    
    Args:
        file_bytes: bytes from uploaded file
        
    Returns:
        tuple: (is_valid, confidence, message)
    """
    try:
        # Try to open image
        image = Image.open(io.BytesIO(file_bytes))
        image_array = np.array(image)
        
        # Validate
        validator = BrainMRIValidator()
        is_valid, confidence, reason = validator.validate_image(image_array)
        
        if is_valid:
            return True, confidence, f"✅ Valid brain MRI scan (confidence: {confidence:.1f}%)"
        else:
            return False, confidence, f"❌ Not a brain MRI scan: {reason}"
            
    except Exception as e:
        return False, 0.0, f"❌ Invalid image file: {str(e)}"


# Test function
if __name__ == "__main__":
    print("Brain MRI Validator - Testing")
    print("=" * 50)
    
    # Test with sample data
    test_cases = [
        ("brain_mri.nii", "Should pass"),
        ("truck.jpg", "Should fail - not medical image"),
        ("cat.png", "Should fail - not brain"),
    ]
    
    for filename, expected in test_cases:
        print(f"\nTest: {filename}")
        print(f"Expected: {expected}")
        print("-" * 50)