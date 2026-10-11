"""Text recognition for images.

proposal.md §4.5 argues for a dedicated detector + recognizer rather than
leaning on a VLM: two orders of magnitude cheaper, better on small glyphs, and
it keeps text search alive if the VLM never lands. Tesseract is the PoC's
stand-in for that pair — not what production would ship, but the right shape.

Two things from the proposal carry over and matter:

* **Decode scale.** §4.4's cheap 224 px decode destroys exactly the pixels OCR
  needs. The CLIP path and the OCR path therefore read the file separately, at
  different resolutions. That duplicated work is deliberate.
* **A gate.** §6.1 only sends flagged files down the OCR path. Tesseract on
  every holiday photo is wasted seconds for empty strings, so `looks_texty`
  screens cheaply first.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)

GATE_SIDE = 512  # small text survives this; 320 blurs it below the edge threshold
GATE_THRESHOLD = 0.015
TIMEOUT_S = 30


class NoOcr:
    """Disabled. Returns nothing for everything."""

    available = False
    name = "disabled"

    def text(self, path: Path) -> str:
        return ""


class Tesseract:
    name = "tesseract"

    def __init__(self, gate: bool = True):
        self.binary = shutil.which("tesseract")
        self.available = self.binary is not None
        self.gate = gate
        if not self.available:
            log.warning("tesseract not found — in-image text will not be searchable")

    def edge_fraction(self, path: Path) -> float:
        """Fraction of strong horizontal gradients at a fixed small size."""
        import numpy as np
        from PIL import Image

        with Image.open(path) as im:
            im = im.convert("L")
            im.thumbnail((GATE_SIDE, GATE_SIDE))
            a = np.asarray(im, dtype=np.int16)
        edges = np.abs(np.diff(a, axis=1))
        if edges.size == 0:
            # a single-column (or empty) image: np.diff gives nothing and
            # mean() of an empty array is NaN, which silently compares False
            # against every threshold.
            return 0.0
        return float((edges > 40).mean())

    def looks_texty(self, path: Path) -> bool:
        """Cheap stand-in for proposal §4.5's text detector.

        Glyphs produce dense small-scale contrast; photographs mostly do not.
        Measured on the fixtures: text 0.032-0.12, non-text 0.000-0.009. The
        threshold sits well below the text range deliberately — the errors are
        asymmetric. A false positive wastes a second; a false negative makes a
        document permanently unsearchable. An earlier 0.045 cutoff sat *inside*
        the text range and silently skipped code screenshots.

        These figures come from synthetic fixtures, which are cleaner than real
        photographs. Re-check on a real corpus; --ocr-all bypasses the gate.
        """
        try:
            return self.edge_fraction(path) > GATE_THRESHOLD
        except Exception as exc:
            log.debug("gate failed on %s (%s) — sending to OCR anyway", path, exc)
            return True

    def text(self, path: Path) -> str:
        if not self.available:
            return ""
        if self.gate and not self.looks_texty(path):
            return ""
        try:
            out = subprocess.run(
                [self.binary, str(path), "stdout", "--psm", "6"],
                capture_output=True,
                text=True,
                timeout=TIMEOUT_S,
                check=False,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            log.warning("ocr failed on %s: %s", path, exc)
            return ""
        if out.returncode != 0:
            log.debug("tesseract rc=%s on %s", out.returncode, path)
            return ""
        return " ".join(out.stdout.split())


def get_ocr(enabled: bool = True, gate: bool = True):
    return Tesseract(gate=gate) if enabled else NoOcr()
