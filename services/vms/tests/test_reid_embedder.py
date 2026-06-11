"""ReIDEmbedder tests: preprocessing, L2-normalization, batching, quality gate.

These tests exercise the real :class:`app.reid.embedder.ReIDEmbedder` contract
WITHOUT a GPU, a real OSNet model, or onnxruntime installed:

  * A fake ``onnxruntime`` module is registered in ``sys.modules`` exposing a
    deterministic ``InferenceSession`` whose output is the L2-norm of the mean
    of the input tensor broadcast to 512 dims — enough to assert the embedder
    wires preprocessing -> inference -> postprocess correctly and returns a
    512-d unit vector, plus that ``embed_batch`` aligns rows with inputs.
  * ``crop_quality`` / ``is_quality_crop`` are pure NumPy and tested directly.

The contract under test mirrors :mod:`app.faces.recognizer`:
  * ``ReIDEmbedder(model_path, input_w, input_h, device, embedding_dim)``
  * ``embed(bgr_crop) -> 512-d L2-normalized float32 | None``
  * ``embed_batch([crops]) -> list aligned with input (None for bad crops)``
  * ``l2_normalize``, ``crop_quality``, ``is_quality_crop`` helpers
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

# Make ``vms/`` importable so ``app.reid.embedder`` resolves.
VMS_ROOT = Path(__file__).resolve().parents[1]
if str(VMS_ROOT) not in sys.path:
    sys.path.insert(0, str(VMS_ROOT))


# ---------------------------------------------------------------------------
# Fake onnxruntime + a real-enough model file so _ensure_loaded() succeeds.
# ---------------------------------------------------------------------------

EMB_DIM = 512
INPUT_W = 128
INPUT_H = 256


class _FakeInput:
    name = "images"
    type = "tensor(float)"
    shape = ["batch", 3, INPUT_H, INPUT_W]


class _FakeOutput:
    name = "features"


class _FakeSession:
    """Deterministic stand-in: feature = unit-norm of (mean(input)+ramp).

    The output depends on the input content so different crops give different
    (but reproducible) embeddings — enough to test alignment & normalization.
    """

    def __init__(self, model_path, sess_options=None, providers=None):
        self._providers = list(providers or ["CPUExecutionProvider"])

    def get_providers(self):
        return self._providers

    def get_inputs(self):
        return [_FakeInput()]

    def get_outputs(self):
        return [_FakeOutput()]

    def run(self, output_names, feeds):
        blob = np.asarray(next(iter(feeds.values())), dtype=np.float32)
        n = blob.shape[0]
        ramp = np.linspace(1.0, 2.0, EMB_DIM, dtype=np.float32)
        out = np.empty((n, EMB_DIM), dtype=np.float32)
        for i in range(n):
            out[i] = blob[i].mean() * ramp + ramp
        return [out]


def _install_fake_ort():
    ort = types.ModuleType("onnxruntime")

    class _SessionOptions:
        def __init__(self):
            self.log_severity_level = 0

    ort.SessionOptions = _SessionOptions  # type: ignore[attr-defined]
    ort.InferenceSession = _FakeSession  # type: ignore[attr-defined]
    ort.get_available_providers = lambda: ["CPUExecutionProvider"]  # type: ignore[attr-defined]
    sys.modules["onnxruntime"] = ort


_install_fake_ort()

# cv2 may not be installed in CI; the embedder imports it lazily inside
# _preprocess. Provide a tiny resize stub if missing.
if "cv2" not in sys.modules:
    try:
        import cv2  # noqa: F401
    except Exception:
        cv2 = types.ModuleType("cv2")
        cv2.INTER_LINEAR = 1  # type: ignore[attr-defined]
        cv2.COLOR_BGR2GRAY = 6  # type: ignore[attr-defined]
        cv2.CV_64F = 6  # type: ignore[attr-defined]

        def _resize(img, size, interpolation=1):
            w, h = size
            ys = (np.linspace(0, img.shape[0] - 1, h)).astype(int)
            xs = (np.linspace(0, img.shape[1] - 1, w)).astype(int)
            return img[np.ix_(ys, xs)]

        cv2.resize = _resize  # type: ignore[attr-defined]
        sys.modules["cv2"] = cv2


embedder_mod = pytest.importorskip(
    "app.reid.embedder", reason="app.reid.embedder not importable"
)
ReIDEmbedder = embedder_mod.ReIDEmbedder
l2_normalize = embedder_mod.l2_normalize
crop_quality = embedder_mod.crop_quality
is_quality_crop = embedder_mod.is_quality_crop


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def model_file(tmp_path: Path) -> str:
    # The embedder only checks the path EXISTS (onnxruntime is faked).
    p = tmp_path / "osnet.onnx"
    p.write_bytes(b"not-a-real-onnx")
    return str(p)


@pytest.fixture()
def emb(model_file: str) -> ReIDEmbedder:
    return ReIDEmbedder(
        model_path=model_file,
        input_w=INPUT_W,
        input_h=INPUT_H,
        device="cpu",
        embedding_dim=EMB_DIM,
    )


def _person_crop(seed: int, h: int = 256, w: int = 128) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# l2_normalize
# ---------------------------------------------------------------------------


def test_l2_normalize_unit_length():
    v = l2_normalize(np.array([3.0, 4.0], dtype=np.float32))
    assert v.dtype == np.float32
    assert np.isclose(float(np.linalg.norm(v)), 1.0, atol=1e-6)


def test_l2_normalize_zero_vector_safe():
    v = l2_normalize(np.zeros(8, dtype=np.float32))
    assert np.all(v == 0.0)


# ---------------------------------------------------------------------------
# embed: shape + normalization + missing-model handling
# ---------------------------------------------------------------------------


def test_embed_returns_unit_512d(emb: ReIDEmbedder):
    out = emb.embed(_person_crop(1))
    assert out is not None
    assert out.shape == (EMB_DIM,)
    assert out.dtype == np.float32
    assert np.isclose(float(np.linalg.norm(out)), 1.0, atol=1e-5)


def test_embed_empty_crop_returns_none(emb: ReIDEmbedder):
    assert emb.embed(None) is None
    assert emb.embed(np.zeros((0, 0, 3), dtype=np.uint8)) is None


def test_embed_picks_up_baked_input_shape(emb: ReIDEmbedder):
    # The fake session declares shape [batch, 3, 256, 128]; _ensure_loaded
    # should adopt input_h=256, input_w=128 from it.
    emb.embed(_person_crop(2))
    assert emb.input_h == INPUT_H
    assert emb.input_w == INPUT_W


def test_missing_model_raises(tmp_path: Path):
    bad = ReIDEmbedder(model_path=str(tmp_path / "nope.onnx"), device="cpu")
    with pytest.raises(FileNotFoundError):
        bad.embed(_person_crop(3))


# ---------------------------------------------------------------------------
# embed_batch: alignment + parity with embed()
# ---------------------------------------------------------------------------


def test_embed_batch_aligns_and_skips_invalid(emb: ReIDEmbedder):
    # Use crops with clearly different content so the (fake) model yields
    # distinguishable embeddings — proves rows map to the right inputs.
    dark = np.full((256, 128, 3), 10, dtype=np.uint8)
    bright = np.full((256, 128, 3), 240, dtype=np.uint8)
    crops = [dark, None, bright, np.zeros((0, 0, 3), np.uint8)]
    out = emb.embed_batch(crops)
    assert len(out) == 4
    assert out[0] is not None and out[0].shape == (EMB_DIM,)
    assert out[1] is None
    assert out[2] is not None and out[2].shape == (EMB_DIM,)
    assert out[3] is None
    # Distinct crops -> distinct embeddings; also confirms row alignment.
    assert not np.allclose(out[0], out[2])
    assert np.allclose(out[0], emb.embed(dark), atol=1e-5)
    assert np.allclose(out[2], emb.embed(bright), atol=1e-5)


def test_embed_batch_matches_single(emb: ReIDEmbedder):
    crop = _person_crop(20)
    single = emb.embed(crop)
    batched = emb.embed_batch([crop])[0]
    assert single is not None and batched is not None
    assert np.allclose(single, batched, atol=1e-5)


def test_embed_batch_all_invalid_returns_all_none(emb: ReIDEmbedder):
    out = emb.embed_batch([None, np.zeros((0, 0, 3), np.uint8)])
    assert out == [None, None]


def test_embed_batch_empty_input():
    e = ReIDEmbedder(model_path="unused", device="cpu")
    assert e.embed_batch([]) == []


# ---------------------------------------------------------------------------
# crop_quality / is_quality_crop
# ---------------------------------------------------------------------------


def test_crop_quality_measurements():
    crop = _person_crop(30, h=256, w=128)
    q = crop_quality(crop)
    assert q["width"] == 128
    assert q["height"] == 256
    assert q["area"] == 128 * 256
    assert q["aspect"] == pytest.approx(2.0)
    assert q["blur"] >= 0.0


def test_crop_quality_empty():
    q = crop_quality(None)
    assert q["ok"] is False
    assert q["area"] == 0


def test_is_quality_crop_rejects_sliver():
    sliver = _person_crop(31, h=20, w=200)  # wide sliver, aspect 0.1
    assert is_quality_crop(sliver, min_aspect=1.0) is False


def test_is_quality_crop_accepts_standing_body():
    body = _person_crop(32, h=300, w=120)  # tall, aspect 2.5
    assert is_quality_crop(body, min_aspect=1.0) is True


def test_is_quality_crop_area_fraction_gate():
    body = _person_crop(33, h=100, w=50)  # area 5000
    frame_area = 1920 * 1080
    # 5000 / (1920*1080) ~ 0.0024 < 0.01 -> rejected.
    assert is_quality_crop(body, frame_area=frame_area, min_area_frac=0.01) is False
    # Lower the bar -> accepted.
    assert is_quality_crop(body, frame_area=frame_area, min_area_frac=0.001) is True
