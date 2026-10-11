"""CLIP-ViT-B/16 on host CPU.

This module is the PoC's reason for existing and also its longest-lived artifact:
it becomes the `CpuNpu` backend behind the interface in
../../IMPLEMENTATION_PLAN.md §1.7, and the numbers it produces are the §8
benchmark baseline.

One warning carried over from PLAN.md §2: when these timings are used as that
baseline, the backend must be ONNX Runtime, not PyTorch. A naive PyTorch
baseline flatters the accelerator by strawmanning the alternative.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

import numpy as np

from . import DIM, MODEL, PRETRAINED

log = logging.getLogger(__name__)


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)


class ClipCpu:
    def __init__(self, threads: int | None = None):
        import open_clip
        import torch

        if threads:
            torch.set_num_threads(threads)
        self.torch = torch
        log.info("loading %s/%s on CPU…", MODEL, PRETRAINED)
        model, _, preprocess = open_clip.create_model_and_transforms(
            MODEL, pretrained=PRETRAINED, device="cpu"
        )
        model.eval()
        self.model = model
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(MODEL)

    def encode_images(self, paths: list[Path]) -> dict[Path, np.ndarray]:
        """Encode a batch. Unreadable files are skipped, not fatal."""
        from PIL import Image

        tensors, kept = [], []
        for p in paths:
            try:
                with Image.open(p) as im:
                    tensors.append(self.preprocess(im.convert("RGB")))
                kept.append(p)
            except Exception as exc:  # corrupt, truncated, or not an image after all
                log.warning("skip %s: %s", p, exc)
        if not kept:
            return {}

        with self.torch.no_grad():
            batch = self.torch.stack(tensors)
            feats = self.model.encode_image(batch).cpu().numpy()
        feats = _l2(feats.astype(np.float32))
        return dict(zip(kept, feats, strict=True))

    def encode_text(self, text: str) -> np.ndarray:
        with self.torch.no_grad():
            toks = self.tokenizer([text])
            feat = self.model.encode_text(toks).cpu().numpy()
        return _l2(feat.astype(np.float32))[0]


def check_dim(v: np.ndarray) -> np.ndarray:
    if v.shape[-1] != DIM:
        raise SystemExit(
            f"model produced {v.shape[-1]}-d vectors, schema expects {DIM}. "
            f"Check MODEL in sospoc/__init__.py."
        )
    return v


class FakeEncoder:
    """Deterministic stand-in for CLIP. Test and CI use only.

    Vectors are derived from a string: an image's filename stem, or the query
    text. So searching "cat" ranks `cat.jpg` first, which lets the end-to-end
    tests assert on ordering without downloading a model. It says nothing about
    retrieval quality — that is what the real run in PLAN.md §6 is for.
    """

    def __init__(self, threads: int | None = None):
        del threads

    @staticmethod
    def _vec(text: str) -> np.ndarray:
        seed = hashlib.blake2b(text.strip().lower().encode(), digest_size=8).digest()
        rng = np.random.default_rng(int.from_bytes(seed, "little"))
        return _l2(rng.normal(size=DIM).astype(np.float32).reshape(1, DIM))[0]

    def encode_images(self, paths: list[Path]) -> dict[Path, np.ndarray]:
        return {p: self._vec(p.stem) for p in paths if p.exists()}

    def encode_text(self, text: str) -> np.ndarray:
        return self._vec(text)


def get_encoder(threads: int | None = None):
    """ClipCpu, unless SOS_FAKE_ENCODER=1 selects the deterministic stub."""
    if os.environ.get("SOS_FAKE_ENCODER") == "1":
        log.warning("SOS_FAKE_ENCODER=1 — using FakeEncoder, results are meaningless")
        return FakeEncoder(threads)
    return ClipCpu(threads)
