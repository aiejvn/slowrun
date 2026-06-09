"""
Thin wrapper around tiktoken that prefers the local ./tokenizer_cache/
directory.  Import this instead of calling tiktoken.get_encoding() directly.

Usage:
    from tokenizer import get_encoding
    enc = get_encoding("gpt2")
"""
import os
import pathlib

_CACHE_DIR = pathlib.Path(__file__).parent / "tokenizer_cache"

if _CACHE_DIR.exists() and any(_CACHE_DIR.iterdir()):
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(_CACHE_DIR))

import tiktoken  


def get_encoding(name: str = "gpt2") -> tiktoken.Encoding:
    return tiktoken.get_encoding(name)
