"""Upload validation and sample-path confinement."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_VIDEO_BYTES = 80 * 1024 * 1024

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm", ".mkv"}


class UploadRejected(ValueError):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def _looks_like_image(data: bytes) -> bool:
    if data.startswith(b"\xff\xd8\xff"):
        return True
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return True
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return True
    if len(data) > 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return True
    if data.startswith(b"BM"):
        return True
    return False


def _looks_like_video(data: bytes) -> bool:
    if len(data) < 12:
        return False
    if data[4:8] == b"ftyp":
        return True
    if data.startswith(b"\x1a\x45\xdf\xa3"):
        return True
    if data.startswith(b"RIFF") and data[8:12] == b"AVI ":
        return True
    return False


def classify_upload(filename: str, content_type: str, data: bytes) -> str:
    """Return 'image' or 'video' from file content. Filename and MIME are hints only."""
    if not data:
        raise UploadRejected("The upload is empty.")
    name = (filename or "").lower()
    suffix = Path(name).suffix
    mime = (content_type or "").lower()

    image_hint = mime.startswith("image/") or suffix in IMAGE_EXTENSIONS
    video_hint = mime.startswith("video/") or suffix in VIDEO_EXTENSIONS

    if _looks_like_image(data):
        if len(data) > MAX_IMAGE_BYTES:
            raise UploadRejected(f"Image exceeds the {MAX_IMAGE_BYTES // (1024 * 1024)} MB limit.")
        return "image"
    if _looks_like_video(data):
        if len(data) > MAX_VIDEO_BYTES:
            raise UploadRejected(f"Video exceeds the {MAX_VIDEO_BYTES // (1024 * 1024)} MB limit.")
        return "video"
    if image_hint or video_hint:
        raise UploadRejected("The file extension or MIME type does not match the file contents.")
    raise UploadRejected("Unsupported file. Send a JPEG, PNG, WEBP, BMP, or MP4/WEBM video.")


def metadata_region(file_bytes: bytes) -> bytes:
    """Bytes that can hold metadata, excluding compressed pixel payloads."""
    if not file_bytes:
        return b""
    if file_bytes.startswith(b"\xff\xd8"):
        sos = file_bytes.find(b"\xff\xda")
        return file_bytes[: sos if sos != -1 else min(len(file_bytes), 65536)]
    if file_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        out = bytearray()
        i = 8
        while i + 8 <= len(file_bytes):
            length = int.from_bytes(file_bytes[i : i + 4], "big")
            ctype = file_bytes[i + 4 : i + 8]
            start = i + 8
            end = start + length
            if end + 4 > len(file_bytes):
                break
            if ctype in {b"tEXt", b"iTXt", b"zTXt", b"eXIf"}:
                out += file_bytes[start:end]
            if ctype == b"IDAT":
                break
            i = end + 4
        return bytes(out)
    return file_bytes[:4096]


def resolve_dataset_file(path: str, dataset_root: Path, extra_roots: Optional[list[Path]] = None) -> Path:
    """Resolve a sample path and refuse anything outside the allowed directories."""
    if not path or "\x00" in path or path != path.strip():
        raise UploadRejected("Invalid sample path.")
    allowed_roots = [dataset_root.resolve()]
    if extra_roots:
        allowed_roots.extend([r.resolve() for r in extra_roots])
    raw = Path(path)
    candidates = []
    if raw.is_absolute():
        candidates.append(raw.resolve())
    else:
        candidates.append((Path.cwd() / raw).resolve())
        for root in allowed_roots:
            candidates.append((root / raw).resolve())
            if raw.parts and raw.parts[0].lower() == root.name.lower():
                candidates.append((root.joinpath(*raw.parts[1:])).resolve())
    allowed = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
    for candidate in candidates:
        is_safe = False
        for root in allowed_roots:
            try:
                candidate.relative_to(root)
                is_safe = True
                break
            except ValueError:
                continue
        if not is_safe:
            continue
        if candidate.suffix.lower() not in allowed:
            raise UploadRejected("Sample path is not an allowed media file.")
        if candidate.is_file():
            return candidate
    raise UploadRejected("Sample image not found.")


def client_key(host: Optional[str]) -> str:
    return host or "unknown"


class SlidingRateLimiter:
    def __init__(self, limit: int = 30, window_seconds: float = 60.0):
        self.limit = limit
        self.window = window_seconds
        self._hits: dict[str, list[float]] = {}

    def allow(self, key: str, now: float) -> Tuple[bool, int]:
        bucket = [t for t in self._hits.get(key, []) if now - t < self.window]
        if len(bucket) >= self.limit:
            self._hits[key] = bucket
            return False, 0
        bucket.append(now)
        self._hits[key] = bucket
        return True, self.limit - len(bucket)
