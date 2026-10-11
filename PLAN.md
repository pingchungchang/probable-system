# PoC A — Implementation Plan

**The question this answers:** *on my own files, does semantic search actually find what I meant?*

Nothing else. Not whether the NPU works, not whether FUSE is fast, not whether the index
converges after a crash. Those are real problems with known shapes and the main plan handles
them. This one is different: it is the risk that cannot be engineered away, because the answer
lives in CLIP's retrieval quality rather than in anything we write (`../proposal.md` §12.2).

If the answer is no, every other estimate in `../IMPLEMENTATION_PLAN.md` is moot, and two days
is a cheap price for finding out.

---

## 1. Scope

**In**

- Watch a directory, index images as they appear, search them in natural language.
- CLIP-ViT-B/16 on host CPU. Dense retrieval only.
- Two commands, one SQLite file, one process each. No daemon, no gRPC, no IPC.

**Out** — deliberately, and none of it is hard to add later

| Excluded | Why it isn't needed to answer the question |
| :--- | :--- |
| NPU / HailoRT | The question is about retrieval quality, not where the matmul runs |
| ~~FTS5, RRF~~ | **Added after Stage 1.** Dense-only failed on the first real corpus: CLIP cannot read, so screenshots of code were unfindable. §6.3 predicted text-bearing failures would prove the lexical and OCR paths load-bearing; they did, on day one. |
| ~~OCR~~ | **Added after Stage 1**, same reason. tesseract stands in for proposal §4.5's detector + recognizer. |
| Negation, VLM captions | Downstream features; meaningless if dense retrieval is poor |
| Reconciler, event ring, backpressure | Correctness-under-stress, not quality |
| Rust CLI, protobuf, systemd | Packaging |

**Stage 2 adds FUSE and nothing else.** Keep the split — mixing "is the idea good" with "does
the mount work" means a disappointing result has two possible causes.

---

## 2. Stage 1 — inotify, ~12 hours over 2 days

No FUSE. `watchdog` gives the same `(path, closed)` signal for a fraction of the setup cost, and
interception is not what we are testing.

| | Task | Hours |
| :-- | :--- | ---: |
| 1.1 | `nix develop`; confirm the devShell resolves torch/open-clip and that `vec0` accepts `float[512]` | 1 |
| 1.2 | `store.py` — open DB, apply `schema.sql`, insert/lookup/delete | 2 |
| 1.3 | `embed.py` — load CLIP-ViT-B/16, `encode_image` / `encode_text`, L2-normalise | 2 |
| 1.4 | `index.py` — watchdog observer, 3 s debounce by path, MIME check, ingest loop, progress to stderr | 3 |
| 1.5 | `search.py` + `bin/sos-search` — encode query, kNN, print ranked paths with scores | 2 |
| 1.6 | Point it at a real photo directory. Wait for the backfill. Start searching. | 2 |

**Day 2 afternoon is the deliverable** — not working code, but *the answer*. Sit with it and run
twenty queries you actually care about.

### Choices that matter

- **PyTorch CPU now, ONNX Runtime later.** PyTorch is zero-friction for iteration. But when this
  becomes the §8 benchmark baseline, it **must** be ONNX Runtime — a naive PyTorch baseline makes
  the NPU look good by strawmanning the alternative, and that is exactly what a technical judge
  will catch. Budget half a day for the swap at that point, not now.
- **ViT-B/16, not B/32.** B/32 is faster to iterate with, but B/16 is the target model, so B/16
  is what tells you about the real system — and keeps the baseline honest. Only drop to B/32 if
  CPU encode makes the backfill genuinely unbearable, and record that you did.
- **Column names identical to `../proposal.md` §5.** The PoC database should be readable by the
  real daemon. Costs nothing now, saves a migration later.
- **Delete handling included** (it is four lines), **rename handling excluded** (it is a design
  question, answered in proposal §6.1, and irrelevant to retrieval quality).

---

## 3. Stage 2 — swap in FUSE, ~12 hours over 2 days

Only start this once Stage 1 has given a verdict.

| | Task | Hours |
| :-- | :--- | ---: |
| 2.1 | Vendor `passthrough_hp.cc` at a pinned commit; build; mount unmodified; `pjdfstest` | 3 |
| 2.2 | Add a `release` hook writing `path\n` to a FIFO. Crude on purpose — no ring, no protobuf | 3 |
| 2.3 | `index.py` reads the FIFO instead of watchdog. One function swapped | 1 |
| 2.4 | Probe `FUSE_PASSTHROUGH` availability on this kernel and libfuse; record the answer | 2 |
| 2.5 | Measure: `cp` throughput through the mount vs direct to the backing directory | 3 |

2.1 is the one to hold the line on: **upstream must pass `pjdfstest` before any of our code is
added.** An afternoon's discipline that permanently separates their bugs from ours.

2.4 is the highest-information hour in the whole PoC. `FUSE_PASSTHROUGH` is what the <3% host-CPU
claim rests on, and it is a yes/no the kernel answers immediately.

---

## 4. Architecture

```
bin/sos-index  ──►  watchdog (stage 1) or FIFO (stage 2)
                      │  3s debounce
                      ▼
                    embed.py ── CLIP-ViT-B/16, CPU ──► 512-d normalised
                      │
                      ▼
                    store.py ──► sos.db  (files + file_vectors)
                                   ▲
bin/sos-search ────────────────────┘  read-only connection
```

Two processes, no IPC: the searcher opens the same SQLite file read-only. WAL mode makes
concurrent read-while-indexing work without ceremony.

---

## 5. Schema

`sospoc/schema.sql` — a strict subset of `../proposal.md` §5, same names and types, so the real
daemon can later open a PoC database unchanged. `float[512]` here rather than `int8[512]`:
quantisation is an optimisation, and measuring unquantised recall first gives the baseline that
§12.3's overfetch tuning will eventually be judged against.

---

## 6. Test plan

This is the point of the exercise, so it gets specified rather than improvised.

### 6.1 Corpus

Your own photo directory. Not COCO, not Open Images — a public benchmark tells you how CLIP
performs on a public benchmark, which is already published. Real holiday photos, screenshots,
scanned receipts, and blurry accidents are what the product has to survive.

Target 2,000–10,000 images. Record the count.

### 6.2 Queries

Write **20 queries before searching anything**, in a text file, phrased the way you would
actually think of them. Writing them afterwards means writing them to fit what the system
already does, which measures nothing.

Cover all four kinds, roughly five each:

| Kind | Example |
| :--- | :--- |
| Concrete object | "a bicycle" |
| Scene / atmosphere | "a rainy street at night" |
| Fine-grained visual | "a person wearing a red jacket" |
| Text-bearing | "a screenshot of a terminal" |

### 6.3 Scoring

For each query, you judge the top 5. Record a hit if at least one result is something you would
have accepted had you been searching manually.

| Outcome | Reading |
| :--- | :--- |
| **≥ 16/20** | The premise holds. Proceed to Stage 2 and the full plan with confidence. |
| **10–15/20** | Workable but not magical. Hybrid retrieval (FTS5 + RRF) is doing more work than the main plan assumes — move RRF tuning earlier and lower the recall@10 target before it is on a slide. |
| **< 10/20** | Stop and reconsider. Either the corpus is adversarial, or §12.2 has arrived early. Check B/16 vs B/32 and normalisation before concluding anything. |

Note **which kinds** fail, not just the count. Failures concentrated in one row are informative:
text-bearing failures confirm the OCR pipeline is load-bearing rather than a nice-to-have, and
fine-grained failures predict where negation will struggle.

### 6.4 Numbers to record

These are the §8 baseline, so capture them even though the PoC is about feel:

| Measurement | How |
| :--- | :--- |
| CPU encode time per image | mean over the backfill, note core count |
| Host CPU during backfill | `/usr/bin/time -v`, or `cpu.stat` if cgroups are handy |
| Search latency | wall time of `sos-search`, p50 over the 20 queries, minus process startup |
| Index size per file | `sos.db` bytes ÷ file count |
| Backfill wall time | start to finish, for the throughput table in proposal §6.7 |

---

## 7. What this becomes

Not throwaway:

- `embed.py` is the **`CpuNpu` backend** behind the interface in `../IMPLEMENTATION_PLAN.md`
  §1.7 — the same role `MockNpu` plays, with a real model.
- The measurements in §6.4 are the **§8 benchmark baseline**, which has to exist anyway. Every
  headline claim is "versus the same thing on host CPU."
- `sospoc/schema.sql` is the first draft of `daemon/sosd/store/schema.sql`.
- Stage 2's vendored `passthrough_hp.cc` is the actual driver base for weeks 3–4.

The only genuinely discardable parts are the watchdog listener and the FIFO.

---

## 8. Risks specific to the PoC

| | Risk | Response |
| :-- | :--- | :--- |
| P1 | `sqlite-vec` unavailable or `vec0` rejects the declaration | Already handled: `store.py` falls back to numpy brute force automatically. Verified on NixOS — `pkgs.sqlite-vec` 0.1.6 ships `lib/vec0.so`, which the flake exports as `SQLITE_VEC_PATH`. |
| P5 | NixOS: pip wheels for torch will not link without nix-ld or an FHS env | Use `nix develop`. `requirements.txt` is the non-Nix path only. |
| P2 | CPU backfill is slow enough to stall iteration | Expected — it is the baseline being measured. Index 500 images first and start searching while the rest runs. |
| P3 | Tempting to add FTS5 "since it's easy" | Don't. Dense-only is what isolates the measurement. Adding sparse retrieval before scoring makes the result uninterpretable. |
| P4 | The twenty queries get rewritten to fit the results | The single most likely way to get a false positive. Commit the query file before the first search. |
