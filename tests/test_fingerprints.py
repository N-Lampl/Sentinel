from __future__ import annotations

import io
import random

import numpy as np
import pytest
from PIL import Image

from conftest import image_from_seed
from dataset_sentinel.fingerprints.image import (
    _NP_TRANSFORMS,
    _TRANSPOSE_OPS,
    DIHEDRAL_NAMES,
    fingerprint_image,
    hamming,
    normalized_correlation,
    thumb_array,
    transform_thumb,
)
from dataset_sentinel.fingerprints.index import HammingIndex, popcount64, to_uint64, union_find_groups


def _save(tmp_path, name, img, **kw):
    p = tmp_path / name
    img.save(p, **kw)
    return p


def test_fingerprint_basic_fields(tmp_path):
    img = image_from_seed(1)
    p = _save(tmp_path, "a.jpg", img, quality=92)
    fp = fingerprint_image(p, "a")
    assert fp.ok and fp.error is None
    assert (fp.width, fp.height) == img.size
    assert fp.content_sha256 and fp.pixel_hash
    assert fp.dhash is not None and 0 <= fp.dhash < 2**64
    assert fp.dhash_dihedral is not None and len(fp.dhash_dihedral) == 8 and fp.dhash_dihedral[0] == fp.dhash
    assert fp.thumb16 is not None and len(fp.thumb16) == 256
    assert not fp.is_blank


def test_fingerprint_errors(tmp_path):
    missing = fingerprint_image(tmp_path / "nope.jpg", "m")
    assert not missing.ok and missing.error == "file not found"
    empty = tmp_path / "empty.jpg"
    empty.write_bytes(b"")
    fp = fingerprint_image(empty, "e")
    assert not fp.ok and fp.error.startswith("empty file")
    garbage = tmp_path / "garbage.jpg"
    garbage.write_bytes(b"not an image at all" * 10)
    fp = fingerprint_image(garbage, "g")
    assert not fp.ok and "not a recognised image" in fp.error


def test_blank_image_detected(tmp_path):
    p = _save(tmp_path, "blank.png", Image.new("RGB", (64, 64), (120, 120, 120)))
    fp = fingerprint_image(p, "b")
    assert fp.ok and fp.is_blank


def test_dhash_robust_to_reencode_and_resize(tmp_path):
    img = image_from_seed(2)
    a = fingerprint_image(_save(tmp_path, "a.png", img), "a")
    b = fingerprint_image(_save(tmp_path, "b.jpg", img, quality=50), "b")
    c = fingerprint_image(_save(tmp_path, "c.jpg", img.resize((80, 60)), quality=90), "c")
    assert a.content_sha256 != b.content_sha256
    assert hamming(a.dhash, b.dhash) <= 3
    assert hamming(a.dhash, c.dhash) <= 4
    assert normalized_correlation(thumb_array(a), thumb_array(b)) > 0.95


def test_pixel_hash_matches_lossless_resave(tmp_path):
    img = image_from_seed(3)
    a = fingerprint_image(_save(tmp_path, "a.png", img), "a")
    b = fingerprint_image(_save(tmp_path, "b.bmp", img), "b")
    assert a.content_sha256 != b.content_sha256
    assert a.pixel_hash == b.pixel_hash


@pytest.mark.parametrize("t", range(1, 8))
def test_dihedral_hash_matches_transformed_file(tmp_path, t):
    img = image_from_seed(10 + t)
    original = fingerprint_image(_save(tmp_path, "o.png", img), "o")
    transformed = fingerprint_image(_save(tmp_path, f"t{t}.png", img.transpose(_TRANSPOSE_OPS[t])), "t")
    # hash of T(original) ~ hash of the transformed file
    assert hamming(original.dhash_dihedral[t], transformed.dhash) <= 2, DIHEDRAL_NAMES[t]
    # and the thumbnail verification agrees
    corr = normalized_correlation(transform_thumb(thumb_array(original), t), thumb_array(transformed))
    assert corr > 0.95


@pytest.mark.parametrize("t", range(8))
def test_numpy_transforms_match_pillow(t):
    a = np.arange(256, dtype=np.uint8).reshape(16, 16)
    a[0, 0], a[0, 15], a[15, 0] = 255, 200, 100
    via_pil = np.asarray(Image.fromarray(a).transpose(_TRANSPOSE_OPS[t])) if _TRANSPOSE_OPS[t] is not None else a
    assert np.array_equal(via_pil, _NP_TRANSFORMS[t](a))


def test_popcount64():
    x = np.array([0, 1, 0xFF, 0xFFFFFFFFFFFFFFFF, 1 << 63], dtype=np.uint64)
    assert popcount64(x).tolist() == [0, 1, 8, 64, 1]


@pytest.mark.parametrize("threshold", [0, 1, 3, 6, 10])
def test_hamming_index_matches_bruteforce(threshold):
    rng = random.Random(threshold)
    base = [rng.getrandbits(64) for _ in range(60)]
    codes = list(base)
    # add perturbed copies within/just outside threshold
    for c in base[:30]:
        v = c
        for _ in range(rng.randint(0, threshold + 2)):
            v ^= 1 << rng.randrange(64)
        codes.append(v)
    arr = to_uint64(codes)
    index = HammingIndex(arr, threshold)
    got = {(a, b) for a, b, _ in index.pairs_within()}
    expected = {(i, j) for i in range(len(codes)) for j in range(i + 1, len(codes)) if bin(codes[i] ^ codes[j]).count("1") <= threshold}
    assert got == expected
    # query API: each code must find itself and its neighbours
    q = index.query(arr[:10])
    got_q = {(qi, i) for qi, i, _ in q}
    expected_q = {(qi, i) for qi in range(10) for i in range(len(codes)) if bin(codes[qi] ^ codes[i]).count("1") <= threshold}
    assert got_q == expected_q


def test_union_find_groups():
    comps = union_find_groups(6, [(0, 1), (1, 2), (4, 5)])
    assert sorted(comps) == [[0, 1, 2], [4, 5]]


def test_large_image_fingerprint_does_not_choke(tmp_path):
    img = image_from_seed(5, size=(3000, 2000))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    p = tmp_path / "big.jpg"
    p.write_bytes(buf.getvalue())
    fp = fingerprint_image(p, "big")
    assert fp.ok and fp.width == 3000
