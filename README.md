# Document Enhancement Tool

A Python-based document enhancement tool that automatically detects, corrects perspective distortion, and enhances the readability of scanned or photographed documents using computer vision techniques.

## Features

- **Automatic Document Detection**: Detects document boundaries using contour detection
- **Perspective Correction**: Warps skewed documents to a rectangular view
- **Adaptive Enhancement**: Applies adaptive thresholding for optimal text clarity
- **Quality Metrics**: Evaluates enhancement quality using CNR (Contrast-to-Noise Ratio) and edge density
- **Visual Processing Pipeline**: Shows all intermediate processing steps
- **Easy File Selection**: GUI-based file picker for convenience

## How It Works

The tool processes documents through several stages:

1. **Edge Detection**: Converts to grayscale, applies Gaussian blur, and detects edges using Canny edge detection
2. **Document Boundary Detection**: Finds the largest quadrilateral contour representing the document
3. **Perspective Correction**: Applies perspective transformation to create a rectangular view
4. **Adaptive Thresholding**: Enhances text readability using Gaussian-based adaptive thresholding  
5. **Quality Evaluation**: Calculates metrics to assess enhancement effectiveness

## Requirements

- Python 3.7 or higher
- Required libraries (see requirements.txt)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/AndiRoihan/DocumentEnhancer.git
cd DocumentEnhancer
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the program:
```bash
python enhancer.py
```

1. A file dialog will open - select your document image (JPG, JPEG, PNG, or BMP)
2. The program will automatically process the image through all enhancement stages
3. View the results in the matplotlib windows that appear
4. The enhanced image will be saved to `output/output_final.jpg`

## Output

- **Console Output**: Processing progress and quality metrics (CNR and Edge Density)
- **Visual Output**: Two matplotlib figures showing:
  - Complete processing pipeline with all intermediate steps
  - Side-by-side comparison of input and final output
- **File Output**: Enhanced image saved to the specified path

## Quality Metrics

- **CNR (Contrast-to-Noise Ratio)**: Measures the contrast between text and background
- **Edge Density**: Quantifies the sharpness and detail preservation in the enhanced image

## Supported File Formats

- **Input**: JPG, JPEG, PNG, BMP
- **Output**: JPG (can be modified in the code)

## Technical Details

### Key Technologies Used

- **OpenCV**: Computer vision operations (contour detection, perspective transformation, adaptive thresholding)
- **NumPy**: Numerical computations and array operations
- **Matplotlib**: Visualization of processing steps and results
- **Tkinter**: GUI file selection dialog

### Processing Parameters

- **Gaussian Blur**: 5x5 kernel for noise reduction
- **Canny Edge Detection**: Low threshold 50, high threshold 150
- **Adaptive Threshold**: Block size 15, C constant 10
- **Minimum Contour Area**: 10,000 pixels for document detection
