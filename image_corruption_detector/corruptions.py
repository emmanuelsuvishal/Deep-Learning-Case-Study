import io
import random
import numpy as np
from PIL import Image, ImageFilter

def corrupt_pil(image, kind=None, severity=None):
    """Apply a synthetic corruption to an RGB PIL image."""
    kind = kind or random.choice([
        "Gaussian noise", "Blur", "Brightness corruption",
        "Contrast corruption", "JPEG compression artifacts", "Mixed corruption"
    ])
    severity = int(severity or random.randint(1, 3))
    severity = max(1, min(3, severity))
    img = image.convert("RGB")

    if kind == "Gaussian noise":
        arr = np.asarray(img).astype(np.float32)
        sigma = [12, 25, 40][severity - 1]
        noise = np.random.normal(0, sigma, arr.shape)
        return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))

    if kind == "Blur":
        return img.filter(ImageFilter.GaussianBlur(radius=[0.7, 1.4, 2.2][severity - 1]))

    if kind == "Brightness corruption":
        from PIL import ImageEnhance
        return ImageEnhance.Brightness(img).enhance([0.55, 0.35, 0.20][severity - 1])

    if kind == "Contrast corruption":
        from PIL import ImageEnhance
        return ImageEnhance.Contrast(img).enhance([0.65, 0.40, 0.20][severity - 1])

    if kind == "JPEG compression artifacts":
        quality = [35, 18, 8][severity - 1]
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

    if kind == "Mixed corruption":
        # One or two randomly selected corruptions, with moderate severity.
        operations = ["Gaussian noise", "Blur", "Brightness corruption", "Contrast corruption", "JPEG compression artifacts"]
        chosen = random.sample(operations, k=random.choice([2, 3]))
        result = img
        for op in chosen:
            result = corrupt_pil(result, op, severity)
        return result

    return img
