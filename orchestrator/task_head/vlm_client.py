"""Pluggable VLM backends for the Task Head.

The orchestrator only talks to `VLMClient.generate_structured()`, so swapping
Claude for a local model (Qwen2.5-VL, etc.) never touches planning logic.
"""

import base64
import io
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def encode_image(image: Any) -> tuple[str, str]:
    """Normalize an image to (base64_string, media_type).

    Accepts a file path (str), raw bytes (PNG/JPEG), a PIL.Image, or an
    HxWx3 uint8 numpy array (e.g. straight from OmniGibson's RGB sensor).
    """
    if isinstance(image, str):
        with open(image, "rb") as f:
            data = f.read()
        media = "image/jpeg" if image.lower().endswith((".jpg", ".jpeg")) else "image/png"
        return base64.standard_b64encode(data).decode("utf-8"), media

    if isinstance(image, (bytes, bytearray)):
        media = "image/jpeg" if bytes(image[:3]) == b"\xff\xd8\xff" else "image/png"
        return base64.standard_b64encode(bytes(image)).decode("utf-8"), media

    # PIL image
    if hasattr(image, "save") and hasattr(image, "mode"):
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        return base64.standard_b64encode(buf.getvalue()).decode("utf-8"), "image/png"

    # numpy array
    if hasattr(image, "shape") and hasattr(image, "dtype"):
        from PIL import Image  # lazy import; only needed for array inputs

        arr = image[..., :3]  # drop alpha channel if present
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        return base64.standard_b64encode(buf.getvalue()).decode("utf-8"), "image/png"

    raise TypeError(f"Unsupported image type: {type(image)}")


class VLMClient(ABC):
    """Minimal interface the orchestrator depends on."""

    @abstractmethod
    def generate_structured(
        self,
        system: str,
        text: str,
        images: Optional[List[Any]] = None,
        schema: Type[T] = BaseModel,
    ) -> T:
        """Send text + images, get back a validated instance of `schema`."""


class ClaudeVLMClient(VLMClient):
    """Claude backend. Reads ANTHROPIC_API_KEY (or an `ant auth login`
    profile) from the environment."""

    def __init__(self, model: str = "claude-opus-4-8", max_tokens: int = 16000):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    def generate_structured(
        self,
        system: str,
        text: str,
        images: Optional[List[Any]] = None,
        schema: Type[T] = BaseModel,
    ) -> T:
        content: List[dict] = []
        for img in images or []:
            b64, media = encode_image(img)
            content.append(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": media, "data": b64},
                }
            )
        content.append({"type": "text", "text": text})

        response = self.client.messages.parse(
            model=self.model,
            max_tokens=self.max_tokens,
            thinking={"type": "adaptive"},
            system=system,
            messages=[{"role": "user", "content": content}],
            output_format=schema,
        )
        return response.parsed_output


class QwenVLClient(VLMClient):
    """Placeholder for a fully-local backend (vLLM or HF transformers serving
    Qwen2.5-VL with guided JSON decoding). Implement generate_structured()
    to hit your local endpoint and validate with `schema.model_validate_json`.
    """

    def __init__(self, endpoint: str = "http://localhost:8000/v1"):
        self.endpoint = endpoint

    def generate_structured(self, system, text, images=None, schema=BaseModel):
        raise NotImplementedError(
            "Local backend not wired up yet. Serve Qwen2.5-VL behind an "
            "OpenAI-compatible endpoint (vLLM --guided-decoding) and parse "
            "the JSON response into `schema` here."
        )
