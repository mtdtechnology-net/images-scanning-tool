import base64
import io
from PIL import Image
import pypdfium2 as pdfium

MAX_DIMENSION = 2048


def _resize_if_needed(image: Image.Image) -> Image.Image:
    """Downscale image so its longest side is at most MAX_DIMENSION pixels."""
    w, h = image.size
    if max(w, h) <= MAX_DIMENSION:
        return image
    ratio = MAX_DIMENSION / max(w, h)
    new_size = (int(w * ratio), int(h * ratio))
    return image.resize(new_size, Image.LANCZOS)


def image_to_base64(image: Image.Image, fmt: str = "JPEG") -> str:
    """Convert a PIL Image to a base64 string.

    Uses JPEG by default for smaller size (vs PNG).
    """
    buffer = io.BytesIO()
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    image.save(buffer, format=fmt, quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def parse_pdf_to_images(file_bytes: bytes) -> list[str]:
    """Convert each page of a PDF to a base64-encoded image."""
    pdf = pdfium.PdfDocument(file_bytes)
    images_b64 = []

    for page_index in range(len(pdf)):
        page = pdf[page_index]

        bitmap = page.render(scale=1.5)
        pil_image = bitmap.to_pil()
        pil_image = _resize_if_needed(pil_image)
        images_b64.append(image_to_base64(pil_image))

    pdf.close()
    return images_b64


def parse_image_to_base64(file_bytes: bytes) -> list[str]:
    """Convert an uploaded image file to a base64 string."""
    image = Image.open(io.BytesIO(file_bytes))
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    image = _resize_if_needed(image)
    return [image_to_base64(image)]


def parse_document_to_images(file_bytes: bytes, content_type: str) -> list[str]:
    """Parse a document into a list of base64-encoded images.

    Args:
        file_bytes: Raw bytes of the uploaded file
        content_type: MIME type of the file

    Returns:
        List of base64-encoded image strings (one per page for PDFs)

    Raises:
        ValueError: If the file type is not supported
    """
    if content_type == "application/pdf":
        return parse_pdf_to_images(file_bytes)
    elif content_type in ("image/png", "image/jpeg", "image/jpg", "image/webp"):
        return parse_image_to_base64(file_bytes)
    else:
        raise ValueError(
            f"Unsupported file type: {content_type}. "
            "Supported types: PDF, PNG, JPG, WEBP"
        )
