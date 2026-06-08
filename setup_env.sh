#!/bin/bash
# Sets up the Python venv and pre-caches tokenizer data.
# Run on the login node (which has internet); compute nodes do not.

VENV="${1:-.venv}"

if [ ! -f "$VENV/bin/activate" ]; then
    echo "Creating venv at $VENV ..."
    python3.13 -m venv "$VENV"
    "$VENV/bin/pip" install -q -r requirements.txt
else
    echo "Venv already exists at $VENV, skipping install."
fi

echo "Pre-caching tiktoken gpt2 encoding..."
"$VENV/bin/python" -c "import tiktoken; tiktoken.get_encoding('gpt2')"

echo "Checking CUDA availability..."
"$VENV/bin/python" - <<'PYEOF'
import torch
if not torch.cuda.is_available():
    print("WARNING: CUDA is not available. Training will be slow or may fail.")
    exit(1)
print(f"CUDA OK — {torch.cuda.device_count()} device(s), {torch.version.cuda}")
PYEOF
