import numpy as np
import onnxruntime as ort
import cv2

def preprocess_image(image: np.ndarray, input_size=(320, 320)) -> np.ndarray:
    """
    Preprocess the input image for the segmentation model.
    Resize, normalize, and add batch dimension.
    """
    # Resize image
    image_resized = cv2.resize(image, input_size)
    # Normalize to [0, 1]
    image_normalized = image_resized.astype(np.float32) / 255.0

    img = np.transpose(image_normalized, (2, 0, 1))  # Change data layout from HWC to CHW
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    return img


def run_segmentation(model: ort.InferenceSession, image: np.ndarray) -> np.ndarray:
    """
    Run the segmentation model on the preprocessed image and return the mask.
    """
    input_tensor = preprocess_image(image)

    #ONNX input name
    input_name = model.get_inputs()[0].name
    outputs = model.run(None, {input_name: input_tensor})
    mask = outputs[0][0, 0]

    mask = (mask * 255).astype(np.uint8)  # Scale mask to [0, 255]
    return mask


def apply_mask_to_image(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Overlay road segmentation mask on top of the original image.
    - image: original BGR frame
    - mask: 2D binary mask (0 = background, 1 = road)
    - alpha: transparency of overlay
    """

    if mask.ndim == 3:
        mask = mask.squeeze()

    # Resize mask to match image size
    mask_resized = cv2.resize(mask.astype(np.uint8),
                              (image.shape[1], image.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

    # Create colored mask (green for road)
    road_color = np.array([0, 255, 0], dtype=np.uint8)
    colored_mask = np.zeros_like(image, dtype=np.uint8)
    colored_mask[mask_resized == 1] = road_color

    # Blend
    overlay = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)

    return overlay