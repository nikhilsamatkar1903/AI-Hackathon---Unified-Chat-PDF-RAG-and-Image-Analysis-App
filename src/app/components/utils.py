from PIL import Image
from io import BytesIO
import base64
from typing import Any


def img_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def safe_extract_model_names(models_info: Any) -> tuple:
    # handle different shapes from ollama.list()
    if hasattr(models_info, "models"):
        models_list = models_info.models
    elif isinstance(models_info, dict):
        models_list = models_info.get("models", [])
    else:
        models_list = list(models_info) if models_info is not None else []

    def _extract(entry):
        if isinstance(entry, dict):
            return entry.get("model") or entry.get("name")
        try:
            return getattr(entry, "model", getattr(entry, "name", None))
        except Exception:
            return None

    names = tuple(n for n in (_extract(m) for m in models_list) if n)
    return tuple(list(names) + ["gpt-4o-mini"])
