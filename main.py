import cv2
import numpy as np
import onnxruntime as ort
print(cv2.__version__)

def preprocess_image(image_path, input_size=(224, 224)):
    """Load an image from disk and preprocess it for inference."""
    # Load image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image at path '{image_path}' not found")
    
    # Resize image to the input size of the model
    resized_image = cv2.resize(image, input_size)
    
    # Normalize pixel values to the range [0, 1] and add channel dimension
    normalized_image = resized_image.astype(np.float32) / 255.0
    expanded_image = np.expand_dims(normalized_image, axis=0)  # Add batch dimension

    return expanded_image

def load_model(model_path):
    """Load an ONNX model for inference."""
    session = ort.InferenceSession(model_path)
    return session

def run_inference(session, input_tensor):
    """Run inference on an input tensor and return the output."""
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Perform inference
    result = session.run([output_name], {input_name: input_tensor})
    return result[0]

def main():
    # Path to your ONNX model and input image
    model_path = "model.onnx"
    image_path = "medical_image.jpg"

    # Preprocess the image
    input_tensor = preprocess_image(image_path)

    # Load the ONNX model
    session = load_model(model_path)

    # Run inference
    predictions = run_inference(session, input_tensor)

    # Interpret and print the result
    predicted_class = np.argmax(predictions)
    print(f"Predicted class: {predicted_class}")

if __name__ == "__main__":
    main()
