# Disease Image Scan

This Python project uses ONNX Runtime and OpenCV to classify disease-related medical images. The pipeline includes loading a pre-trained ONNX model, processing input images, and generating a predicted disease class.

## Features
- **Model Integration**: Utilizes ONNX Runtime for efficient inference on medical image datasets.
- **Image Preprocessing**: Automatically resizes and normalizes images to meet the model’s input specifications.
- **Disease Prediction**: Outputs a classification result to assist in early disease detection.

## Requirements
- Python 3.7 or higher
- `opencv-python`
- `onnxruntime`

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/username/disease-image-scan.git
   cd disease-image-scan
   ```
2. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Add your ONNX model and test image to the project directory:
   ```
   model.onnx
   test_image.jpg
   ```

## Usage
1. Run the disease image scanner:
   ```bash
   python main.py
   ```
2. The script will preprocess the image, run inference, and output the predicted disease class.

## Notes
- Ensure that the input image dimensions match the model’s expected input size.
- The ONNX model used should be trained on relevant medical datasets for reliable disease classification results.

## Contributing
Feel free to open issues, suggest enhancements, or submit pull requests to improve this project.
