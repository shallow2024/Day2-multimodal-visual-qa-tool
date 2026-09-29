from __future__ import annotations

import base64
import io
from types import SimpleNamespace

import pytest
from PIL import Image

from vision_qa import (
    DEFAULT_MODEL,
    VisionQaError,
    analyze_image,
    encode_image,
    load_image_bytes,
)


def make_test_image(tmp_path):
    path = tmp_path / "test.png"
    Image.new("RGB", (8, 8), color=(40, 120, 200)).save(path, format="PNG")
    return path


def fake_completion(content: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def test_encode_image_creates_png_data_url(tmp_path):
    path = make_test_image(tmp_path)
    payload = encode_image(str(path))

    assert payload.media_type == "image/png"
    assert payload.data_url.startswith("data:image/png;base64,")
    encoded = payload.data_url.split(",", 1)[1]
    assert base64.b64decode(encoded).startswith(b"\x89PNG")


def test_missing_file_is_rejected(tmp_path):
    with pytest.raises(VisionQaError, match="was not found"):
        load_image_bytes(str(tmp_path / "missing.png"))


def test_image_size_limit_is_enforced(tmp_path):
    path = make_test_image(tmp_path)
    with pytest.raises(VisionQaError, match="too large"):
        encode_image(str(path), max_bytes=2)


def test_analyze_image_mocks_groq_and_parses_json(tmp_path):
    path = make_test_image(tmp_path)
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: fake_completion(
                    '{"answer":"A blue square is visible.", "objects":{"square": 1}}'
                )
            )
        )
    )

    result = analyze_image(str(path), "What is visible?", client=client)

    assert result.model == DEFAULT_MODEL
    assert result.answer == "A blue square is visible."
    assert result.objects == {"square": 1}


def test_analyze_image_accepts_fenced_json(tmp_path):
    path = make_test_image(tmp_path)
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: fake_completion(
                    '```json\n{"answer":"A plain image.", "objects":{}}\n```'
                )
            )
        )
    )

    result = analyze_image(str(path), "Describe it.", client=client)

    assert result.answer == "A plain image."
    assert result.objects == {}


def test_analyze_image_rejects_invalid_model_json(tmp_path):
    path = make_test_image(tmp_path)
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: fake_completion("not json"))
        )
    )

    with pytest.raises(VisionQaError, match="valid JSON"):
        analyze_image(str(path), "Describe it.", client=client)


def test_analyze_image_rejects_empty_question(tmp_path):
    path = make_test_image(tmp_path)
    with pytest.raises(VisionQaError, match="cannot be empty"):
        analyze_image(str(path), "  ", client=object())
