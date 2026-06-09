"""
One-time script: download and cache the tiktoken GPT-2 tokenizer to
./tokenizer_cache/ so compute nodes (no internet) can load it offline.

Run once on a login node before submitting Slurm jobs:
    python save_tokenizer.py
"""
import os
import pathlib

CACHE_DIR = pathlib.Path(__file__).parent / "tokenizer_cache"
CACHE_DIR.mkdir(exist_ok=True)

os.environ["TIKTOKEN_CACHE_DIR"] = str(CACHE_DIR)

import tiktoken  # noqa: E402 — import after env var is set

enc = tiktoken.get_encoding("gpt2")

# Smoke-test
hello = enc.encode("hello world")
assert enc.decode(hello) == "hello world"

print(f"Tokenizer saved to: {CACHE_DIR}")
print(f"Files: {list(CACHE_DIR.iterdir())}")
print(f"Vocab size: {enc.n_vocab}")
