import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, morphology, exposure
from skimage.morphology import disk
import tkinter as tk
from tkinter import filedialog
import warnings
warnings.filterwarnings('ignore')

class DocumentEnhancer:
    def __init__(self, image_path):
        """
        Initialize the DocumentEnhancer with an image path
        """
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Could not load image from {image_path}")
        self.processed_image = self.original_image.copy()
        
    def detect_document_corners(self, image):
        """
        Detect corners of the document using contour detection
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Morphological operations to close gaps
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Find the largest rectangular contour
        for contour in contours:
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # If we found a quadrilateral with sufficient area
            if len(approx) == 4 and cv2.contourArea(contour) > 10000:
                return self.order_corners(approx.reshape(4, 2))
        
        # If no contour found, use image boundaries
        h, w = gray.shape
        return np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    
    def order_corners(self, corners):
        """
        Order corners in top-left, top-right, bottom-right, bottom-left order
        """
        # Sort by y-coordinate
        corners = corners[np.argsort(corners[:, 1])]
        
        # Top two points
        top = corners[:2]
        top = top[np.argsort(top[:, 0])]  # Sort by x
        
        # Bottom two points
        bottom = corners[2:]
        bottom = bottom[np.argsort(bottom[:, 0])]  # Sort by x
        
        return np.array([top[0], top[1], bottom[1], bottom[0]], dtype=np.float32)
    
    def warp_perspective(self, corners=None):
        """
        Apply perspective correction to straighten the document
        """
        if corners is None:
            corners = self.detect_document_corners(self.processed_image)
        
        # Calculate target dimensions
        width_top = np.linalg.norm(corners[1] - corners[0])
        width_bottom = np.linalg.norm(corners[2] - corners[3])
        width = int(max(width_top, width_bottom))
        
        height_left = np.linalg.norm(corners[3] - corners[0])
        height_right = np.linalg.norm(corners[2] - corners[1])
        height = int(max(height_left, height_right))
        
        # Define destination points
        dst_corners = np.array([
            [0, 0],
            [width, 0],
            [width, height],
            [0, height]
        ], dtype=np.float32)
        
        # Calculate perspective transform matrix
        transform_matrix = cv2.getPerspectiveTransform(corners, dst_corners)
        
        # Apply perspective correction
        self.processed_image = cv2.warpPerspective(
            self.processed_image, transform_matrix, (width, height)
        )
        
        print(f"Perspective correction applied. New dimensions: {width}x{height}")
    
    def enhance_document(self):
        """
        Apply optimized document enhancement techniques for text documents
        """
        # Convert to grayscale for processing
        gray = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
        
        # 1. Gentle noise reduction - preserve text edges
        denoised = cv2.medianBlur(gray, 3)
        
        # 2. Histogram equalization for better contrast
        equalized = cv2.equalizeHist(denoised)
        
        # 3. Gentle Gaussian blur to smooth out noise
        blurred = cv2.GaussianBlur(equalized, (3, 3), 0)
        
        # 4. Otsu's thresholding for optimal binary conversion
        _, otsu_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 5. Morphological operations to clean up small artifacts
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(otsu_thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        # Convert back to BGR for consistency
        self.processed_image = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
        
        print("Document enhancement completed")
    
    def process_document(self, manual_corners=None):
        """
        Complete document processing pipeline
        """
        print("Starting document processing...")
        
        # Step 1: Perspective correction
        print("Step 1: Applying perspective correction...")
        self.warp_perspective(manual_corners)
        
        # Step 2: Document enhancement
        print("Step 2: Enhancing document quality...")
        self.enhance_document()
        
        print("Document processing completed!")
    
    def enhance_document_advanced(self):
        """
        Advanced enhancement with multiple methods to choose from
        """
        gray = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
        
        # Method 1: Simple Otsu thresholding
        _, method1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Method 2: Adaptive thresholding
        method2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 15, 10)
        
        # Method 3: Enhanced preprocessing + Otsu
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, method3 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Method 4: Morphological preprocessing
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        enhanced = cv2.add(gray, tophat)
        enhanced = cv2.subtract(enhanced, blackhat)
        _, method4 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return method1, method2, method3, method4
    
    def display_enhancement_comparison(self):
        """
        Display different enhancement methods for comparison
        """
        methods = self.enhance_document_advanced()
        method_names = ['Otsu Only', 'Adaptive Threshold', 'Blur + Otsu', 'Morphological + Otsu']
        
        # Fixed layout: 2 rows, 3 columns for original + corrected + 4 methods
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original
        axes[0, 0].imshow(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Perspective corrected
        axes[0, 1].imshow(cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title('After Perspective Correction', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        
        # Empty subplot for top row
        axes[0, 2].axis('off')
        
        # Enhancement methods - arrange in bottom row and use subplots
        for i, (method, name) in enumerate(zip(methods[:3], method_names[:3])):
            axes[1, i].imshow(method, cmap='gray')
            axes[1, i].set_title(f'{name}', fontsize=12, fontweight='bold')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Display the 4th method separately if needed
        if len(methods) > 3:
            fig2, ax = plt.subplots(1, 1, figsize=(8, 10))
            ax.imshow(methods[3], cmap='gray')
            ax.set_title(f'{method_names[3]}', fontsize=14, fontweight='bold')
            ax.axis('off')
            plt.tight_layout()
            plt.show()
    
    def display_results(self, figsize=(15, 10)):
        """
        Display original and processed images side by side
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Original image
        axes[0].imshow(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Document', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Processed image
        axes[1].imshow(cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2RGB))
        axes[1].set_title('Enhanced Document', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def select_best_enhancement_method(self, method_index=0):
        """
        Select and apply the best enhancement method
        method_index: 0=Otsu, 1=Adaptive, 2=Blur+Otsu, 3=Morphological+Otsu
        """
        methods = self.enhance_document_advanced()
        selected_method = methods[method_index]
        self.processed_image = cv2.cvtColor(selected_method, cv2.COLOR_GRAY2BGR)
        
        method_names = ['Otsu Only', 'Adaptive Threshold', 'Blur + Otsu', 'Morphological + Otsu']
        print(f"Applied enhancement method: {method_names[method_index]}")
        
        return methods
    
    def save_result(self, output_path):
        """
        Save the processed image
        """
        success = cv2.imwrite(output_path, self.processed_image)
        if success:
            print(f"Enhanced document saved to: {output_path}")
        else:
            print("Error saving the enhanced document")

# Usage example with improved enhancement
def main():
    root = tk.Tk()
    root.withdraw()  # sembunyikan jendela utama
    file_path = filedialog.askopenfilename(
        title="Pilih gambar dokumen",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    if not file_path:
        print("Tidak ada file yang dipilih. Program dibatalkan.")
        return

    enhancer = DocumentEnhancer(file_path)
    # Step 1: Apply perspective correction only
    print("Applying perspective correction...")
    enhancer.warp_perspective()
    
    # Step 2: Compare different enhancement methods
    print("Comparing enhancement methods...")
    enhancer.display_enhancement_comparison()
    
    # Step 3: Select the best method (you can change the index)
    # 0=Otsu Only, 1=Adaptive Threshold, 2=Blur+Otsu, 3=Morphological+Otsu
    best_method_index = 1  # Try different values: 0, 1, 2, or 3
    enhancer.select_best_enhancement_method(best_method_index)
    
    # Display final results
    enhancer.display_results()
    
    # Save the enhanced document
    # enhancer.save_result("enhanced_document_improved.jpg")

# Function to test all methods and save results
def test_all_enhancement_methods(image_path):
    """
    Test all enhancement methods and save results
    """
    enhancer = DocumentEnhancer(image_path)
    enhancer.warp_perspective()
    
    methods = enhancer.enhance_document_advanced()
    method_names = ['otsu_only', 'adaptive_thresh', 'blur_otsu', 'morphological_otsu']
    
    for i, (method, name) in enumerate(zip(methods, method_names)):
        # Convert to BGR for saving
        method_bgr = cv2.cvtColor(method, cv2.COLOR_GRAY2BGR)
        output_path = f"enhanced_{name}.jpg"
        cv2.imwrite(output_path, method_bgr)
        print(f"Saved: {output_path}")
    
    print("All enhancement methods saved. Compare them to choose the best one!")

# Alternative display function that shows all methods in a grid
def display_all_methods_grid(enhancer):
    """
    Display all enhancement methods in a proper grid layout
    """
    methods = enhancer.enhance_document_advanced()
    method_names = ['Otsu Only', 'Adaptive Threshold', 'Blur + Otsu', 'Morphological + Otsu']
    
    # Create a 2x3 grid to accommodate all images
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original image
    axes[0, 0].imshow(cv2.cvtColor(enhancer.original_image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Document', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Perspective corrected
    axes[0, 1].imshow(cv2.cvtColor(enhancer.processed_image, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('Perspective Corrected', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Method 1: Otsu Only
    axes[0, 2].imshow(methods[0], cmap='gray')
    axes[0, 2].set_title(method_names[0], fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Method 2: Adaptive Threshold
    axes[1, 0].imshow(methods[1], cmap='gray')
    axes[1, 0].set_title(method_names[1], fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Method 3: Blur + Otsu
    axes[1, 1].imshow(methods[2], cmap='gray')
    axes[1, 1].set_title(method_names[2], fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Method 4: Morphological + Otsu
    axes[1, 2].imshow(methods[3], cmap='gray')
    axes[1, 2].set_title(method_names[3], fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()