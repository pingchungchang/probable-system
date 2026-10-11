# SOS — PoC A

Answers one question: **on my own files, does semantic search find what I meant?**

See [PLAN.md](PLAN.md). Two stages, ~12 hours each:

1. **inotify** — watch a directory, index with CLIP on CPU, search. The verdict lands here.
2. **FUSE** — swap the event source for `passthrough_hp.cc`; probe `FUSE_PASSTHROUGH`.

Retrieval is **hybrid**: CLIP dense vectors fused by RRF with a lexical channel over filenames
and OCR text. Dense-only was the original scope, and it failed immediately on real screenshots —
CLIP cannot read, so a screenshot of code scored a green meadow above it. See "Why hybrid" below.

Still excluded: NPU, VLM captions, negation, counting, the reconciler.

```
poc/
├── PLAN.md              # the plan; read this first
├── requirements.txt
├── sospoc/
│   ├── schema.sql       # subset of ../proposal.md §5, same names and types
│   ├── store.py         # SQLite + sqlite-vec + FTS5/BM25 + tokenisation
│   ├── embed.py         # CLIP-ViT-B/16 on CPU → becomes the CpuNpu backend
│   ├── ocr.py           # tesseract + the cheap text-detector gate (§4.5)
│   ├── index.py         # event source, debounce, ingest loop
│   └── search.py        # dense kNN + BM25 fused by RRF (§6.4)
├── bin/{sos-index,sos-search}
├── eval/
│   ├── queries.txt      # fill in BEFORE searching anything
│   └── results.md       # scores + the §8 baseline numbers
├── corpus/              # local only; do not commit images
└── fuse/                # stage 2: vendored passthrough_hp.cc + fork
```

## Status

Specs and config only — `sospoc/*.py` and `bin/*` are not written yet. They are tasks 1.2–1.5 in
PLAN.md §2. The quick start below describes the target, not the current state.

## Quick start (NixOS)

```sh
nix develop                        # python, torch (CPU), open-clip, sqlite-vec
sos-index  ~/Pictures --db sos.db --once   # backfill and exit
sos-index  ~/Pictures --db sos.db          # backfill, then keep watching (Ctrl-C to stop)
sos-search "a rainy street at night" --timings
sos-search --repl                  # model loaded once — use this for the 20 queries
python eval/score.py --db sos.db   # guided scoring (PLAN.md §6.3)
```

The devShell puts `bin/` on `PATH` and exports `SQLITE_VEC_PATH`. `nix run .#index` and
`nix run .#search` work without entering the shell.

<details>
<summary>Non-Nix</summary>

```sh
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
bin/sos-index ~/Pictures
```
</details>

## Why hybrid

CLIP embeds images by appearance, not content. On a Screenshots folder that fails hard:

| Query | Dense only | With lexical + OCR |
| :--- | :--- | :--- |
| `code` | `terminal.jpg` — and a green field ranked 2nd, 0.236 vs 0.238 | `code.jpg` `[ds]` |
| `reciprocal rank fusion` | arbitrary, all scores in a 0.23–0.29 band | `code.jpg` `[ds]` |
| `a black circle on white` | `shape.jpg`, 0.322 — a real margin | unchanged |

Dense works for visual concepts and collapses to noise on anything text-bearing. Two channels
fix different halves: **filenames** (free) and **OCR** (tesseract, gated). `[ds]` in the output
says which channels found a hit.

## OCR

Off the critical path by default via a cheap edge-density gate (proposal §4.5): text scores
0.032–0.12, photographs 0.000–0.009, threshold 0.015. The margin is deliberately asymmetric —
a false positive wastes a second, a false negative makes a document permanently unfindable.

```sh
sos-index ~/Pictures --db sos.db --once            # gate on (default)
sos-index ~/Pictures --db sos.db --once --ocr-all  # OCR everything, no gate
sos-index ~/Pictures --db sos.db --once --no-ocr   # skip OCR
```

The gate is tuned on synthetic fixtures, which are cleaner than real photographs. Re-check it
on a real corpus; `--ocr-all` sidesteps it.

## Vector backend

Two, chosen at runtime, behind one interface — the same `VectorIndex` split the real system
needs:

| Backend | When | Notes |
| :--- | :--- | :--- |
| `sqlite-vec` | `SQLITE_VEC_PATH` set (the flake does this) or the pip package is importable | `vec0` virtual table, cosine distance |
| numpy brute force | otherwise | Fine to ~50k vectors. Not a degraded mode at PoC scale. |

Vectors serialise as raw little-endian float32 — sqlite-vec's own blob format — so the
extension works as a bare `vec0.so` with no Python package involved.

## Tests

```sh
nix develop .#test                          # fast shell: numpy/pillow/watchdog, no torch
python -m unittest discover -s tests -v     # 72 tests
SOS_FORCE_NUMPY=1 python -m unittest discover -s tests   # the other backend
nix flake check -L                          # runs the suite sandboxed, both backends
SOS_TEST_CLIP=1 nix develop -c python -m unittest tests.test_embed   # real weights, ~600 MB
```

Tests use `FakeEncoder` — deterministic vectors keyed on the filename stem — so the suite
runs in seconds with no model download. It verifies plumbing, not retrieval quality; quality
is what the real run in [PLAN.md](PLAN.md) §6 is for.

| Env var | Effect |
| :--- | :--- |
| `SOS_FAKE_ENCODER=1` | Substitute `FakeEncoder` for CLIP. Test and CI only. |
| `SOS_FORCE_NUMPY=1` | Ignore sqlite-vec and use the brute-force backend. |
| `SOS_DEBOUNCE=0.05` | Shorten the 3 s debounce window. |
| `SOS_TEST_CLIP=1` | Un-skip the real-model tests. |
| `SQLITE_VEC_PATH` | Directory holding `vec0.so`. The devShell sets it. |

## CI

`.github/workflows/ci.yml` — `tests` (both backends), `endpoints` (the CLI driven as a user
would), `nix` (flake check + devShell), `lint` (ruff).
`.github/workflows/clip.yml` — real weights, weekly and on demand.

## Before you start

Write all twenty queries in `eval/queries.txt` first. Writing them after seeing results measures
nothing — it is the single most likely way to get a false positive out of this exercise.
