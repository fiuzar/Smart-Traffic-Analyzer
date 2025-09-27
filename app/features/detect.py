import numpy as np
import onnxruntime as ort
import cv2

def preprocess_image(image: np.ndarray, target_size=(320, 320)) -> np.ndarray:
    
    """Resize and normalize the image for model input."""
    
    image_resized = cv2.resize(image, target_size)
    image_normalized = image_resized.astype(np.float32) / 255.0

    image_transposed = np.transpose(image_normalized, (2, 0, 1))  # HWC to CHW
    image_batched = np.expand_dims(image_transposed, axis=0)  # Add batch dimension

    return image_batched

def run_detections(session: ort.InferenceSession, input_image: np.ndarray, conf_threshold=0.25) -> dict:
    
    """
    Run inference on the input image and return detections.
    
    Args:
        session (ort.InferenceSession): The ONNX Runtime inference session.
        input_image (np.ndarray): Preprocessed input image.
        conf_threshold (float): Confidence threshold to filter detections.

    Returns:
        List of detections: [{"box": [x1, y1, x2, y2], "score": float, "class_id": int}, ...]
    """
    
    input_tensor = preprocess_image(input_image)
    inputs = session.get_inputs()[0].name

    outputs = session.run(None, {inputs: input_tensor})

    preds = outputs[0][0]  # Assuming the first output contains the predictions

    detections = []
    for pred in preds:
        conf = pred[4]

        if conf < conf_threshold:
            continue

        class_id = np.argmax(pred[5:])
        x, y, w, h = pred[0:4]

        x1 = int((x - w / 2) * input_image.shape[1])
        y1 = int((y - h / 2) * input_image.shape[0])
        x2 = int((x + w / 2) * input_image.shape[1])
        y2 = int((y + h / 2) * input_image.shape[0])

        detections.append({"box": [x1, y1, x2, y2], "score": float(conf), "class": int(class_id)})

    return detections

def draw_boxes(image: np.ndarray, detections: list, class_names = None) -> np.ndarray:
    
    """
    Draw bounding boxes and labels on the image.
    
    Args:
        image (np.ndarray): Original image. (H, W, C)
        detections (list): List of detections from run_detections.
        class_names (list): List of class names corresponding to class IDs.
    """
    
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        score = det["score"]
        class_id = det["class"]
        label = f"{class_names[class_id] if class_names else class_names}: {score:.2f}"

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 250, 0))
        cv2.putText(image, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return image
