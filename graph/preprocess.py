"""Image preprocessing module for receipt/invoice scanning.

Applies a pipeline of transformations to improve OCR accuracy:
1. Auto-crop: Uses contour detection to find the receipt/document rectangle
   and perspective-corrects it
2. Upscale: Ensures the cropped document is large enough for OCR
3. Contrast enhancement: Makes text stand out more
4. Sharpening: Crisps up edges of characters
5. Size cap: Ensures output doesn't exceed PaddleOCR's max_side_limit
"""
import cv2
import numpy as np
from PIL import Image, ImageEnhance


MIN_OCR_WIDTH = 1800
MAX_OCR_SIDE = 3900


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]

    return rect


def _perspective_crop(img_array: np.ndarray, contour: np.ndarray) -> np.ndarray:
    """Apply perspective transform to extract and flatten a quadrilateral region."""
    pts = contour.reshape(4, 2).astype(np.float32)
    rect = _order_points(pts)

    width_a = np.linalg.norm(rect[2] - rect[3])
    width_b = np.linalg.norm(rect[1] - rect[0])
    max_width = int(max(width_a, width_b))

    height_a = np.linalg.norm(rect[1] - rect[2])
    height_b = np.linalg.norm(rect[0] - rect[3])
    max_height = int(max(height_a, height_b))

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img_array, M, (max_width, max_height))

    return warped


def _auto_crop_receipt(image: Image.Image) -> Image.Image:
    """Detect the receipt/document using edge/contour detection and crop it out.

    Strategy:
    1. Canny edge detection to find edges
    2. Morphological dilation to close gaps between edge fragments
    3. Find contours sorted by area
    4. Look for the largest quadrilateral (4-sided polygon) → perspective warp
    5. Fallback: bounding box of the largest suitable contour
    """
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    img_area = h * w

    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dilated = cv2.dilate(edges, kernel, iterations=3)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("[PREPROCESS] No contours found, skipping crop")
        return image

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Try to find a 4-sided contour (the receipt/document)
    for contour in contours[:10]:
        area = cv2.contourArea(contour)
        if area < img_area * 0.10 or area > img_area * 0.95:
            continue

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        if len(approx) == 4:
            warped = _perspective_crop(img_array, approx)
            wh, ww = warped.shape[:2]
            print(f"[PREPROCESS] Perspective crop: {w}x{h} -> {ww}x{wh}")
            return Image.fromarray(warped)

    # Fallback: bounding rect of the largest valid contour
    for contour in contours[:5]:
        area = cv2.contourArea(contour)
        if area < img_area * 0.10 or area > img_area * 0.95:
            continue

        x, y, cw, ch = cv2.boundingRect(contour)
        margin = 10
        x = max(0, x - margin)
        y = max(0, y - margin)
        x2 = min(w, x + cw + 2 * margin)
        y2 = min(h, y + ch + 2 * margin)

        cropped = image.crop((x, y, x2, y2))
        print(f"[PREPROCESS] Bounding box crop: {w}x{h} -> {x2 - x}x{y2 - y}")
        return cropped

    print("[PREPROCESS] No suitable document contour found, skipping crop")
    return image


def _upscale_if_small(image: Image.Image) -> Image.Image:
    """Upscale the image if it's too small for reliable OCR."""
    w, h = image.size

    if w >= MIN_OCR_WIDTH:
        return image

    scale = min(MIN_OCR_WIDTH / w, 3.0)
    new_w = int(w * scale)
    new_h = int(h * scale)

    upscaled = image.resize((new_w, new_h), Image.LANCZOS)
    print(f"[PREPROCESS] Upscaled: {w}x{h} -> {new_w}x{new_h} ({scale:.1f}x)")

    return upscaled


def _enhance_for_ocr(image: Image.Image) -> Image.Image:
    """Apply contrast enhancement and sharpening for better OCR."""
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.5)

    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(2.0)

    print("[PREPROCESS] Enhanced: contrast=1.5, sharpness=2.0")
    return image


def _cap_size(image: Image.Image) -> Image.Image:
    """Ensure the image doesn't exceed PaddleOCR's max_side_limit of 4000px."""
    w, h = image.size
    max_side = max(w, h)

    if max_side <= MAX_OCR_SIDE:
        return image

    ratio = MAX_OCR_SIDE / max_side
    new_w = int(w * ratio)
    new_h = int(h * ratio)
    image = image.resize((new_w, new_h), Image.LANCZOS)
    print(f"[PREPROCESS] Downscaled to fit PaddleOCR limits: {w}x{h} -> {new_w}x{new_h}")

    return image


def preprocess_for_ocr(image: Image.Image) -> Image.Image:
    """Full preprocessing pipeline for receipt/invoice images before OCR.

    Steps:
    1. Auto-crop to document region (contour detection + perspective warp)
    2. Upscale if the document is too small
    3. Enhance contrast and sharpness
    4. Cap size to PaddleOCR's limits
    """
    print(f"[PREPROCESS] Input image: {image.size[0]}x{image.size[1]}")

    image = _auto_crop_receipt(image)
    image = _upscale_if_small(image)
    image = _enhance_for_ocr(image)
    image = _cap_size(image)

    print(f"[PREPROCESS] Output image: {image.size[0]}x{image.size[1]}")
    return image
