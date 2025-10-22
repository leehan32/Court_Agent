"""Naver Clova OCR helper utilities."""
from __future__ import annotations

import base64
import json
import os
import time
import uuid
from dataclasses import dataclass
from typing import Iterable, List, Optional

import fitz  # PyMuPDF
import requests


class NaverOCRError(RuntimeError):
    """Raised when the Naver OCR API returns an error response."""


@dataclass
class OCRField:
    """Lightweight container for OCR inference fields."""

    text: str
    x: float
    y: float
    line_break: bool


def _normalise_vertices(field: dict) -> tuple[float, float]:
    vertices = field.get("boundingPoly", {}).get("vertices", [])
    if not vertices:
        return 0.0, 0.0
    first = vertices[0]
    return float(first.get("x", 0.0)), float(first.get("y", 0.0))


def _extract_fields(response_payload: dict) -> List[OCRField]:
    images = response_payload.get("images", [])
    fields: List[OCRField] = []
    for image in images:
        if image.get("inferResult") != "SUCCESS":
            continue
        for field in image.get("fields", []) or []:
            text = (field.get("inferText") or "").strip()
            if not text:
                continue
            x, y = _normalise_vertices(field)
            fields.append(
                OCRField(
                    text=text,
                    x=x,
                    y=y,
                    line_break=bool(field.get("lineBreak")),
                )
            )
    return fields


def _fields_to_text(fields: Iterable[OCRField]) -> str:
    sorted_fields = sorted(fields, key=lambda item: (round(item.y, 2), item.x))
    lines: List[str] = []
    buffer: List[str] = []
    last_y: Optional[float] = None

    for item in sorted_fields:
        if last_y is not None:
            y_diff = abs(item.y - last_y)
            threshold = 10 if max(item.y, last_y) > 10 else 0.02
            if y_diff > threshold:
                if buffer:
                    lines.append(" ".join(buffer))
                    buffer = []
                last_y = None

        buffer.append(item.text)
        last_y = item.y

        if item.line_break:
            if buffer:
                lines.append(" ".join(buffer))
                buffer = []
            last_y = None

    if buffer:
        lines.append(" ".join(buffer))

    return "\n".join(lines).strip()


class NaverOCRClient:
    """Thin client around the Naver Clova OCR REST API."""

    def __init__(
        self,
        *,
        invoke_url: Optional[str] = None,
        secret_key: Optional[str] = None,
        api_key_id: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 120,
    ) -> None:
        self.invoke_url = invoke_url or os.getenv("NAVER_OCR_INVOKE_URL")
        self.secret_key = secret_key or os.getenv("NAVER_OCR_SECRET_KEY")
        self.api_key_id = api_key_id or os.getenv("NAVER_OCR_API_KEY_ID")
        self.api_key = api_key or os.getenv("NAVER_OCR_API_KEY")
        self.timeout = timeout

        if not self.invoke_url:
            raise ValueError("Naver OCR invoke URL이 설정되어 있지 않습니다.")
        if not self.secret_key:
            raise ValueError("Naver OCR secret key가 설정되어 있지 않습니다.")

    # ------------------------------------------------------------------
    def _post_request(self, images: List[dict]) -> dict:
        payload = {
            "images": images,
            "requestId": str(uuid.uuid4()),
            "version": "V2",
            "timestamp": int(time.time() * 1000),
        }

        headers = {"X-OCR-SECRET": self.secret_key}
        if self.api_key_id:
            headers["X-NCP-APIGW-API-KEY-ID"] = self.api_key_id
        if self.api_key:
            headers["X-NCP-APIGW-API-KEY"] = self.api_key

        response = requests.post(
            self.invoke_url,
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:  # pragma: no cover - network failure
            raise NaverOCRError(f"OCR 요청 실패: {exc}") from exc

        try:
            payload = response.json()
        except json.JSONDecodeError as exc:  # pragma: no cover - unexpected response
            raise NaverOCRError("OCR 응답을 JSON으로 파싱할 수 없습니다.") from exc

        if payload.get("errorMessage"):
            raise NaverOCRError(payload["errorMessage"])
        return payload

    # ------------------------------------------------------------------
    def infer_image(self, *, image_bytes: bytes, image_format: str = "PNG") -> str:
        encoded = base64.b64encode(image_bytes).decode("utf-8")
        response_payload = self._post_request(
            [
                {
                    "format": image_format.upper(),
                    "name": "ocr-image",
                    "data": encoded,
                }
            ]
        )
        fields = _extract_fields(response_payload)
        return _fields_to_text(fields)

    # ------------------------------------------------------------------
    def infer_file(self, file_path: str) -> str:
        suffix = os.path.splitext(file_path)[1].lower()
        if suffix == ".pdf":
            return self._infer_pdf(file_path)
        if suffix in {".png", ".jpg", ".jpeg"}:
            with open(file_path, "rb") as f:
                return self.infer_image(image_bytes=f.read(), image_format=suffix[1:])
        raise ValueError(f"지원하지 않는 OCR 파일 형식입니다: {suffix}")

    # ------------------------------------------------------------------
    def _infer_pdf(self, file_path: str) -> str:
        doc = fitz.open(file_path)
        page_texts: List[str] = []
        for page_number in range(doc.page_count):
            page = doc.load_page(page_number)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            text = self.infer_image(image_bytes=pix.tobytes("png"), image_format="PNG")
            page_texts.append(text)
        return "\n\n".join(filter(None, (text.strip() for text in page_texts)))

