import cv2
import numpy as np
from easyocr import Reader

_reader = Reader(["en"])

def preprocess_image(img: np.ndarray) -> np.ndarray:
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Light denoising to reduce sensor/compression noise
    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    # Upscale slightly to help small text
    h, w = denoised.shape
    scale = 1.5
    resized = cv2.resize(denoised, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    # CLAHE for local contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(resized)

    # Final light blur to smooth ringing
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

    # Otsu binarization for clean black/white text
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return thresh

def extract_text_from_image(img: np.ndarray):
    thresh = preprocess_image(img)
    result = _reader.readtext(thresh)

    detections = []
    texts = []
    confs = []

    for item in result:
        bbox, text, conf = item
        bbox_list = [[float(x), float(y)] for x, y in bbox]
        detections.append({
            "bbox": bbox_list,
            "text": str(text),
            "confidence": float(conf)
        })
        texts.append(str(text))
        confs.append(float(conf))

    full_text = " ".join(texts)
    avg_conf = float(np.mean(confs)) if confs else 0.0

    return {
        "full_text": full_text,
        "avg_confidence": avg_conf,
        "detections": detections
    }

def load_image_from_path(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not load image: {path}")
    return img