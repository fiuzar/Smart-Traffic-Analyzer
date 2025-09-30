import onnxruntime as ort
import os

def load_onnx_model(model_path: str, use_gpu: bool = False) -> ort.InferenceSession:
    """
    Load an ONNX model with the appropriate execution provider

    Args:
        model_path (str): path to the .onnx file
        use_gpu (bool): Whether to use CUDAExecutionProvider if available

    Returns:
        ort.InferenceSession
    """

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ONNX model not found at: {model_path}")
    
    providers = ['CPUExecutionProvider']
    if use_gpu:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(model_path, sess_options=session_options, providers=providers)
    
    return session

def load_detection_model(model_path: str = None, use_gpu: bool = False) -> ort.InferenceSession:
    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), "..", "models", "v1", "object-detection.onnx")
        model_path = os.path.abspath(model_path)
    return load_onnx_model(model_path, use_gpu)

def load_segmentation_model(model_path: str = None, use_gpu: bool = False) -> ort.InferenceSession:
    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), "..", "models", "v1", "road-segmentation.onnx")
        model_path = os.path.abspath(model_path)
    return load_onnx_model(model_path, use_gpu)