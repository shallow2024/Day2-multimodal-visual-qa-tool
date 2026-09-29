"""Day 2: ask a Groq vision model questions about local or online images."""

from __future__ import annotations

import argparse
import base64
import io
import json
import mimetypes
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx
from groq import Groq
from PIL import Image

DEFAULT_MODEL = "llama-3.2-11b-vision-instruct"
DEFAULT_MAX_BYTES = 10 * 1024 * 1024
SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}


class VisionQaError(RuntimeError):
    """Raised for expected input, configuration, or provider errors."""


@dataclass(frozen=True)
class ImagePayload:
    data_url: str
    media_type: str
    size_bytes: int


@dataclass(frozen=True)
class VisionAnswer:
    answer: str
    objects: dict[str, int]
    model: str
    source: str


def _is_url(source: str) -> bool:
    return source.startswith(("http://", "https://"))


def _guess_media_type(source: str, data: bytes) -> str:
    try:
        with Image.open(io.BytesIO(data)) as image:
            format_name = (image.format or "").upper()
    except Exception as exc:
        raise VisionQaError("The source is not a readable image.") from exc

    format_map = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp", "GIF": "image/gif"}
    media_type = format_map.get(format_name)
    if media_type not in SUPPORTED_IMAGE_TYPES:
        raise VisionQaError("Supported image types are JPEG, PNG, WEBP, and GIF.")
    return media_type


def load_image_bytes(source: str, max_bytes: int = DEFAULT_MAX_BYTES) -> bytes:
    """Load image bytes from a local path or an HTTP(S) URL."""
    if max_bytes < 1:
        raise ValueError("max_bytes must be positive.")

    if _is_url(source):
        try:
            response = httpx.get(source, follow_redirects=True, timeout=20.0)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise VisionQaError(f"Could not download image URL: {exc}") from exc
        data = response.content
    else:
        path = Path(source).expanduser()
        if not path.is_file():
            raise VisionQaError(f"Image file was not found: {path}")
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise VisionQaError(f"Could not read image file: {exc}") from exc

    if len(data) > max_bytes:
        raise VisionQaError(f"Image is too large ({len(data)} bytes); maximum is {max_bytes} bytes.")
    return data


def encode_image(source: str, max_bytes: int = DEFAULT_MAX_BYTES) -> ImagePayload:
    """Validate an image and return a Base64 data URL for a vision request."""
    data = load_image_bytes(source, max_bytes=max_bytes)
    media_type = _guess_media_type(source, data)
    encoded = base64.b64encode(data).decode("ascii")
    return ImagePayload(data_url=f"data:{media_type};base64,{encoded}", media_type=media_type, size_bytes=len(data))


def _extract_json(text: str) -> dict[str, Any]:
    """Parse JSON returned directly or inside a Markdown code fence."""
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    candidate = fenced.group(1) if fenced else cleaned
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise VisionQaError("The vision model did not return valid JSON.") from exc
    if not isinstance(value, dict):
        raise VisionQaError("The vision model returned JSON in an unexpected format.")
    return value


def _normalise_objects(value: Any) -> dict[str, int]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise VisionQaError("The vision model's objects field must be a JSON object.")
    result: dict[str, int] = {}
    for name, count in value.items():
        try:
            number = int(count)
        except (TypeError, ValueError) as exc:
            raise VisionQaError("Object counts must be integers.") from exc
        if number < 0:
            raise VisionQaError("Object counts cannot be negative.")
        result[str(name)] = number
    return result


def analyze_image(
    source: str,
    question: str,
    *,
    client: Any | None = None,
    model: str = DEFAULT_MODEL,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> VisionAnswer:
    """Encode an image, call Groq's vision model, and parse a structured answer."""
    if not question.strip():
        raise VisionQaError("Question cannot be empty.")
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if client is None:
        if not api_key:
            raise VisionQaError("GROQ_API_KEY is missing. Set it before asking the vision model a question.")
        client = Groq(api_key=api_key)

    image = encode_image(source, max_bytes=max_bytes)
    prompt = (
        "Analyze the image and answer the user's question using only visible evidence. "
        "Return ONLY valid JSON with this exact shape: "
        '{"answer":"short detailed answer", "objects":{"object_name": 0}}. '
        "Use an empty objects object when counting is not applicable. Do not guess hidden objects.\n\n"
        f"USER QUESTION: {question}"
    )
    try:
        completion = client.chat.completions.create(
            model=model,
            temperature=0.1,
            max_tokens=500,
            messages=[
                {"role": "system", "content": "You are a careful visual question-answering assistant."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image.data_url}},
                    ],
                },
            ],
        )
    except Exception as exc:
        raise VisionQaError(f"Groq vision request failed: {exc}") from exc

    try:
        content = completion.choices[0].message.content
    except (AttributeError, IndexError, TypeError) as exc:
        raise VisionQaError("Groq returned an unexpected response shape.") from exc
    if not content:
        raise VisionQaError("Groq returned an empty response.")

    parsed = _extract_json(content)
    answer = parsed.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise VisionQaError("The vision model response did not include a text answer.")
    return VisionAnswer(
        answer=answer.strip(),
        objects=_normalise_objects(parsed.get("objects")),
        model=model,
        source=source,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask Groq's vision model a question about an image.")
    parser.add_argument("image", help="Local image path or HTTP(S) image URL")
    parser.add_argument("question", help="Question to ask about the image")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Groq model (default: {DEFAULT_MODEL})")
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES, help="Maximum image size")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = analyze_image(args.image, args.question, model=args.model, max_bytes=args.max_bytes)
        if args.json:
            print(json.dumps(asdict(result), indent=2))
        else:
            print(f"Model: {result.model}")
            print(f"Source: {result.source}")
            print(f"\nAnswer:\n{result.answer}")
            print("\nObject counts:")
            if result.objects:
                for name, count in result.objects.items():
                    print(f"- {name}: {count}")
            else:
                print("- No object counts returned")
        return 0
    except (VisionQaError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
