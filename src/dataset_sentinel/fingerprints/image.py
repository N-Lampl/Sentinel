"""Image fingerprints.

For every image we compute:

* ``content_sha256``: hash of the raw file bytes (byte-identical copies);
* ``pixel_hash``: hash of the decoded RGB pixels (re-saved / re-encoded
  lossless copies, format conversions such as PNG -> BMP);
* ``dhash``: 64-bit difference hash of a 9x8 grayscale thumbnail. Robust to
  resizing, re-compression, small colour/brightness changes;
* ``dhash_dihedral``: the dHash of each of the 8 dihedral transforms
  (identity, rotations by 90/180/270 degrees, horizontal/vertical flips,
  transpose, transverse). Used to catch flipped / rotated derivatives;
* ``dhash_regions``: the dHash of regular sub-regions (2x2 tiles, halves,
  centre crops). A sample whose identity hash matches a region hash of
  another sample is a tile / crop of it;
* ``thumb16``: a 16x16 grayscale thumbnail (256 bytes) used to verify hash
  candidates with a normalised correlation.

Only numpy and Pillow are required.
"""

from __future__ import annotations

import base64
import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

Image.MAX_IMAGE_PIXELS = None  # datasets legitimately contain very large images

DIHEDRAL_NAMES: Tuple[str, ...] = (
    "identity",
    "rotate_90",
    "rotate_180",
    "rotate_270",
    "flip_horizontal",
    "flip_vertical",
    "transpose",
    "transverse",
)

_TRANSPOSE_OPS = [
    None,
    Image.Transpose.ROTATE_90,
    Image.Transpose.ROTATE_180,
    Image.Transpose.ROTATE_270,
    Image.Transpose.FLIP_LEFT_RIGHT,
    Image.Transpose.FLIP_TOP_BOTTOM,
    Image.Transpose.TRANSPOSE,
    Image.Transpose.TRANSVERSE,
]

_THUMB = 64
_DHASH_W, _DHASH_H = 9, 8
_VERIFY = 16

#: inverse of each dihedral transform
DIHEDRAL_INVERSE: Dict[str, str] = {
    "identity": "identity",
    "rotate_90": "rotate_270",
    "rotate_180": "rotate_180",
    "rotate_270": "rotate_90",
    "flip_horizontal": "flip_horizontal",
    "flip_vertical": "flip_vertical",
    "transpose": "transpose",
    "transverse": "transverse",
}

# numpy equivalents of ``_TRANSPOSE_OPS`` (verified against Pillow in the test-suite)
_NP_TRANSFORMS = [
    lambda a: a,
    lambda a: np.rot90(a, 1),
    lambda a: np.rot90(a, 2),
    lambda a: np.rot90(a, 3),
    lambda a: a[:, ::-1],
    lambda a: a[::-1, :],
    lambda a: a.T,
    lambda a: np.rot90(a, 2).T,
]

#: regular sub-regions used for crop / tile detection
REGION_NAMES: Tuple[str, ...] = (
    "tile_top_left",
    "tile_top_right",
    "tile_bottom_left",
    "tile_bottom_right",
    "left_half",
    "right_half",
    "top_half",
    "bottom_half",
    "center_crop_50",
    "center_crop_75",
)


def region_slices(size: int) -> Dict[str, Tuple[slice, slice]]:
    """(row, col) slices of every region for a square array of ``size``."""
    h = size // 2
    q = size // 4
    m = max(1, round(size * 0.125))
    return {
        "tile_top_left": (slice(0, h), slice(0, h)),
        "tile_top_right": (slice(0, h), slice(h, size)),
        "tile_bottom_left": (slice(h, size), slice(0, h)),
        "tile_bottom_right": (slice(h, size), slice(h, size)),
        "left_half": (slice(0, size), slice(0, h)),
        "right_half": (slice(0, size), slice(h, size)),
        "top_half": (slice(0, h), slice(0, size)),
        "bottom_half": (slice(h, size), slice(0, size)),
        "center_crop_50": (slice(q, q + h), slice(q, q + h)),
        "center_crop_75": (slice(m, size - m), slice(m, size - m)),
    }


_REGIONS_64 = region_slices(_THUMB)
_REGIONS_16 = region_slices(_VERIFY)


@dataclass
class ImageFingerprint:
    sample_id: str
    path: str
    ok: bool
    error: Optional[str] = None
    file_size: int = 0
    #: SHA-256 of the file bytes (matches ``sha256sum``)
    content_sha256: Optional[str] = None
    #: BLAKE2b-128 of "WxH" + decoded RGB pixels; None in fast mode
    pixel_hash: Optional[str] = None
    #: "exact" (full decode + pixel hash) or "fast" (reduced JPEG decode, no pixel hash)
    mode: str = "exact"
    width: int = 0
    height: int = 0
    image_mode: Optional[str] = None
    format: Optional[str] = None
    dhash: Optional[int] = None
    dhash_dihedral: Optional[List[int]] = None
    dhash_regions: Optional[List[int]] = None
    #: 16x16 grayscale thumbnail, row-major uint8 bytes
    thumb16: Optional[bytes] = None
    mean_luma: Optional[float] = None
    std_luma: Optional[float] = None
    exif_orientation: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        d["thumb16"] = base64.b64encode(self.thumb16).decode("ascii") if self.thumb16 else None
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImageFingerprint":
        data = dict(data)
        t = data.get("thumb16")
        if isinstance(t, str):
            data["thumb16"] = base64.b64decode(t)
        elif isinstance(t, list):  # cache written by an older version
            data["thumb16"] = bytes(t)
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})

    @property
    def is_blank(self) -> bool:
        return self.ok and self.std_luma is not None and self.std_luma < 0.5


def dhash_from_gray(gray: np.ndarray) -> int:
    """Difference hash of a (8, 9) grayscale float/uint8 array -> 64-bit int."""
    if gray.shape != (_DHASH_H, _DHASH_W):
        raise ValueError(f"expected shape {(_DHASH_H, _DHASH_W)}, got {gray.shape}")
    diff = gray[:, 1:] > gray[:, :-1]
    bits = np.packbits(diff.astype(np.uint8).ravel())
    return int.from_bytes(bits.tobytes(), "big")


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _dhash_of_float_array(arr: np.ndarray) -> int:
    img = Image.fromarray(np.ascontiguousarray(arr, dtype=np.float32), mode="F")
    small = img.resize((_DHASH_W, _DHASH_H), Image.Resampling.BOX)
    return dhash_from_gray(np.asarray(small, dtype=np.float32))


def _thumb_hashes(thumb: np.ndarray) -> List[int]:
    """dHash of the 8 dihedral transforms of a float32 square thumbnail.

    The transforms are applied to the float thumbnail (no 8-bit rounding
    between steps), so ``hash(T(A))`` equals the hash computed from a file
    that stores ``T(A)`` up to floating-point noise.
    """
    return [_dhash_of_float_array(fn(thumb)) for fn in _NP_TRANSFORMS]


def _region_hashes(thumb: np.ndarray) -> List[int]:
    return [_dhash_of_float_array(thumb[_REGIONS_64[name]]) for name in REGION_NAMES]


def thumb_array(fp: "ImageFingerprint") -> Optional[np.ndarray]:
    if not fp.thumb16 or len(fp.thumb16) != _VERIFY * _VERIFY:
        return None
    return np.frombuffer(fp.thumb16, dtype=np.uint8).reshape(_VERIFY, _VERIFY).astype(np.float32)


def transform_thumb(arr: np.ndarray, transform_index: int) -> np.ndarray:
    return np.ascontiguousarray(_NP_TRANSFORMS[transform_index](arr))


def normalized_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation of two equally shaped arrays in [-1, 1] (0 when flat)."""
    a = a.astype(np.float32).ravel()
    b = b.astype(np.float32).ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if denom < 1e-6:
        return 0.0
    return float((a * b).sum() / denom)


def region_correlation(parent_thumb: np.ndarray, child_thumb: np.ndarray, region: str) -> float:
    """Correlation between a region of the parent thumbnail and the whole
    child thumbnail resized to that region's shape."""
    sl = _REGIONS_16[region]
    part = np.ascontiguousarray(parent_thumb[sl], dtype=np.float32)
    h, w = part.shape
    child = Image.fromarray(np.ascontiguousarray(child_thumb, dtype=np.float32), mode="F").resize((w, h), Image.Resampling.BOX)
    return normalized_correlation(part, np.asarray(child, dtype=np.float32))


#: minimum size (shorter side) kept before hashing; larger images are box-reduced first
_REDUCE_TARGET = 256
#: requested decode size for JPEG draft mode
_DRAFT_SIZE = 512


def fingerprint_image(path: Path | str, sample_id: str, *, fast: bool = False) -> ImageFingerprint:
    """Compute the fingerprint for a single image file. Never raises.

    ``fast=True`` decodes JPEGs at reduced resolution (DCT scaling) and skips
    the pixel hash: 3-5x faster on large photos; byte-identical and
    perceptual detection still work, pixel-identical detection does not.
    """
    p = Path(path)
    fp = ImageFingerprint(sample_id=sample_id, path=str(p), ok=False, mode="fast" if fast else "exact")
    try:
        data = p.read_bytes()
    except FileNotFoundError:
        fp.error = "file not found"
        return fp
    except OSError as exc:
        fp.error = f"unreadable: {exc}"
        return fp

    fp.file_size = len(data)
    fp.content_sha256 = hashlib.sha256(data).hexdigest()
    if not data:
        fp.error = "empty file (0 bytes)"
        return fp

    try:
        with Image.open(io.BytesIO(data)) as im:
            fp.format = im.format
            fp.image_mode = im.mode
            fp.width, fp.height = im.size  # original size, before any draft decode
            try:
                exif = im.getexif()
                fp.exif_orientation = int(exif.get(0x0112)) if exif and exif.get(0x0112) else None
            except Exception:
                fp.exif_orientation = None
            if fp.width <= 0 or fp.height <= 0:
                fp.error = "zero-sized image"
                return fp
            if fast and im.format == "JPEG":
                im.draft("RGB" if im.mode not in ("L", "RGB") else im.mode, (_DRAFT_SIZE, _DRAFT_SIZE))
            im.load()

            rgb = im.convert("RGB") if im.mode != "RGB" else im
            if not fast:
                h = hashlib.blake2b(digest_size=16)
                h.update(f"{fp.width}x{fp.height}".encode())
                h.update(rgb.tobytes())
                fp.pixel_hash = h.hexdigest()

            # Integer box reduction first (fast C path, symmetric under rotation
            # because the factor depends on the shorter side only), then a float
            # ("F") thumbnail so no 8-bit rounding happens between resize passes;
            # this keeps the dihedral hashes symmetric.
            factor = max(1, min(rgb.width, rgb.height) // _REDUCE_TARGET)
            small = rgb.reduce(factor) if factor > 1 else rgb
            gray = ImageOps.grayscale(small).convert("F")
            thumb = np.asarray(gray.resize((_THUMB, _THUMB), Image.Resampling.BOX), dtype=np.float32)
            fp.mean_luma = float(thumb.mean())
            fp.std_luma = float(thumb.std())
            hashes = _thumb_hashes(thumb)
            fp.dhash = hashes[0]
            fp.dhash_dihedral = hashes
            fp.dhash_regions = _region_hashes(thumb)
            small = Image.fromarray(thumb, mode="F").resize((_VERIFY, _VERIFY), Image.Resampling.BOX)
            fp.thumb16 = np.clip(np.rint(np.asarray(small, dtype=np.float32)), 0, 255).astype(np.uint8).tobytes()
            fp.ok = True
    except UnidentifiedImageError:
        fp.error = "not a recognised image format"
    except Image.DecompressionBombError as exc:  # pragma: no cover - MAX_IMAGE_PIXELS disabled
        fp.error = f"decompression bomb: {exc}"
    except (OSError, ValueError, SyntaxError) as exc:
        fp.error = f"decode error: {exc}"
    except Exception as exc:  # pragma: no cover - defensive
        fp.error = f"unexpected error: {type(exc).__name__}: {exc}"
    return fp


def make_thumbnail_data_uri(path: Path | str, size: int = 96) -> Optional[str]:
    """Small base64 JPEG data URI for HTML reports (``None`` on failure)."""
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im) or im
            im = im.convert("RGB")
            im.thumbnail((size, size), Image.Resampling.BILINEAR, reducing_gap=2.0)
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=70, optimize=True)
            return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return None
