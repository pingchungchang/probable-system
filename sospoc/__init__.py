"""SOS PoC A — does semantic search find what I meant, on my own files?

See ../PLAN.md. CPU only, dense retrieval only, no NPU.
"""

DIM = 512
# OpenAI's CLIP weights were trained with QuickGELU. Pairing them with a plain
# "ViT-B-16" config silently degrades every embedding — open_clip only warns.
# For a PoC whose whole purpose is measuring retrieval quality, that would bias
# the §6 verdict downward.
MODEL = "ViT-B-16-quickgelu"
PRETRAINED = "openai"
