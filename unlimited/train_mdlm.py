"""
Masked Diffusion Language Model (MDLM) for the slowrun competition.

Bidirectional transformer with log-linear noise schedule, AdaLN timestep
conditioning, and discrete absorbing-mask ELBO. Adapted from the parameter-golf
MDLM (v5) to use the GPT-2 tokenizer and slowrun .pt data format.

Intended role: train this model once, then use it as a bidirectional distillation
teacher for the autoregressive ensemble in unlimited/train.py.  The teacher has
full bidirectional context (sees all 2048 positions) when producing logits,
unlike any AR teacher which only sees left context.

At teacher time: call model.teacher_logits(x) which runs a forward pass at
sigma≈0 (nearly clean) and returns (B, T, vocab_size) logits — drop-in
compatible with teacher_models[0].forward_logits(x) in unlimited/train.py.

Token layout:
    0–50256 : real GPT-2 tokens (50257 total)
    50256   : <|endoftext|> = document boundary; used as BOS in slowrun;
              NEVER masked — serves as visible structural anchor
    50257   : MASK absorbing state (new token, not in GPT-2 vocab)
    50258   : PAD — fills tail chunks shorter than SEQ_LEN; excluded from loss

Usage:
    torchrun --standalone --nproc_per_node=8 unlimited/train_mdlm.py
"""

import os
from dotenv import load_dotenv
load_dotenv()
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc
import math
import time
import json
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import wandb
import tiktoken

# =============================================================================
# CLI arguments
# =============================================================================

parser = argparse.ArgumentParser(description="Train MDLM diffusion LM")
parser.add_argument("--n-layer", type=int, default=24)
parser.add_argument("--n-embd", type=int, default=1024)
parser.add_argument("--n-head", type=int, default=16)
parser.add_argument("--device-batch-size", type=int, default=8)
parser.add_argument("--grad-accum", type=int, default=4)
parser.add_argument("--num-epochs", type=int, default=32)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--weight-decay", type=float, default=0.1)
parser.add_argument("--noise-eps", type=float, default=1e-3)
parser.add_argument("--input-bin", type=str, default=None)
parser.add_argument("--input-val-bin", type=str, default=None)
parser.add_argument("--checkpoint-path", type=str, default="mdlm_checkpoint.pt")
parser.add_argument("--run", type=str, default=None)
parser.add_argument("--wandb-group",   type=str, default=None)
parser.add_argument("--wandb-offline", action="store_true")
parser.add_argument("--eval-elbo-steps", type=int, default=128,
                    help="Riemann steps for discrete ELBO eval (higher = tighter bound)")
parser.add_argument("--eval-elbo-seqs", type=int, default=256,
                    help="Number of val sequences to evaluate ELBO on")
parser.add_argument("--resume", type=str, default=None,
                    help="Path to checkpoint to resume from")
args = parser.parse_args()

# =============================================================================
# Constants
# =============================================================================

VOCAB_SIZE   = 50257   # GPT-2 real tokens (indices 0–50256)
EOS_ID       = 50256   # <|endoftext|> = doc boundary; never masked
MASK_ID      = 50257   # absorbing MASK state
PAD_ID       = 50258   # structural padding; excluded from loss
TOTAL_VOCAB  = 50258   # model predicts indices 0–50257 (real tokens + MASK)
PADDED_VOCAB = 50304   # embedding table size (next multiple of 64 above 50259)

SEQ_LEN  = 2048
DATA_DIR = "fineweb_data"
NEG_INF  = -1e6

# =============================================================================
# Distributed helpers
# =============================================================================

def get_dist_info():
    if all(k in os.environ for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE")):
        return True, int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    return False, 0, 0, 1

def print0(s="", **kwargs):
    if int(os.environ.get("RANK", 0)) == 0:
        print(s, **kwargs)

class DummyWandb:
    def __init__(self): self.summary = {}
    def log(self, *a, **kw): pass
    def finish(self): pass

# =============================================================================
# Noise schedule
# =============================================================================

def log_linear_noise(t):
    """Log-linear schedule: alpha(t) = 1 - (1-eps)*t, sigma = -log(alpha)."""
    alpha = 1 - (1 - args.noise_eps) * t
    sigma = -torch.log(alpha.clamp(min=1e-8))
    return sigma, alpha

# =============================================================================
# Model
# =============================================================================

def rms_norm(x):
    return F.rms_norm(x, (x.size(-1),))

def apply_rotary(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], dim=3)


class TimestepEmbedder(nn.Module):
    """Sinusoidal sigma embedding → MLP → cond_dim conditioning vector."""
    def __init__(self, cond_dim=128):
        super().__init__()
        half = cond_dim // 2
        self.register_buffer("freqs",
            torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half))
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_dim * 4), nn.SiLU(), nn.Linear(cond_dim * 4, cond_dim)
        )

    def forward(self, sigma):
        emb = sigma[:, None] * self.freqs[None, :]
        return self.mlp(torch.cat([emb.sin(), emb.cos()], dim=-1))


class Attention(nn.Module):
    """Bidirectional (non-causal) multi-head attention with RoPE and Q/K RMS norm."""
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.hd = dim // n_heads
        self.c_q = nn.Linear(dim, dim, bias=False)
        self.c_k = nn.Linear(dim, dim, bias=False)
        self.c_v = nn.Linear(dim, dim, bias=False)
        self.c_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q = self.c_q(x).view(B, T, self.n_heads, self.hd)
        k = self.c_k(x).view(B, T, self.n_heads, self.hd)
        v = self.c_v(x).view(B, T, self.n_heads, self.hd)
        q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
        q, k = rms_norm(q), rms_norm(k)
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=False
        )
        return self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C))


class AdaLN(nn.Module):
    """Adaptive layer norm: scale + shift conditioned on the timestep embedding."""
    def __init__(self, dim, cond_dim=128):
        super().__init__()
        self.proj = nn.Linear(cond_dim, 2 * dim, bias=True)

    def forward(self, x, c):
        s, sh = self.proj(c).unsqueeze(1).chunk(2, dim=-1)
        return rms_norm(x) * (1 + s) + sh


class Block(nn.Module):
    def __init__(self, dim, n_heads, cond_dim=128):
        super().__init__()
        self.attn = Attention(dim, n_heads)
        self.adaln_attn = AdaLN(dim, cond_dim)
        self.adaln_mlp = AdaLN(dim, cond_dim)
        hidden = 256 * ((4 * dim + 255) // 256)  # SwiGLU, round to multiple of 256
        self.mlp_gate = nn.Linear(dim, hidden, bias=False)
        self.mlp_fc   = nn.Linear(dim, hidden, bias=False)
        self.mlp_proj = nn.Linear(hidden, dim, bias=False)

    def forward(self, x, cos, sin, c):
        x = x + self.attn(self.adaln_attn(x, c), cos, sin)
        h = self.adaln_mlp(x, c)
        x = x + self.mlp_proj(F.silu(self.mlp_gate(h)) * self.mlp_fc(h))
        return x


class DiffusionLM(nn.Module):
    def __init__(self, n_layer, n_embd, n_head, cond_dim=128):
        super().__init__()
        self.n_embd = n_embd
        self.wte = nn.Embedding(PADDED_VOCAB, n_embd)
        self.sigma_map = TimestepEmbedder(cond_dim)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, cond_dim) for _ in range(n_layer)])
        self.head = nn.Linear(n_embd, PADDED_VOCAB, bias=False)
        hd = n_embd // n_head
        inv_freq = 1.0 / (10000 ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))
        freqs = torch.outer(torch.arange(SEQ_LEN * 2, dtype=torch.float32), inv_freq)
        self.register_buffer("cos", freqs.cos()[None, :, None, :])
        self.register_buffer("sin", freqs.sin()[None, :, None, :])

    def forward(self, xt, sigma):
        """Forward pass — DDP-compatible entry point. Returns raw float logits (B, T, TOTAL_VOCAB)."""
        B, T = xt.shape
        x = self.wte(xt)
        c = F.silu(self.sigma_map(sigma)).to(dtype=x.dtype)
        cos, sin = self.cos[:, :T], self.sin[:, :T]
        for block in self.blocks:
            x = block(x, cos, sin, c)
        return self.head(rms_norm(x))[..., :TOTAL_VOCAB].float()

    def subs_log_probs(self, xt, sigma):
        """MDLM substitution log probs: visible tokens frozen to identity, masked positions predicted.

        Used during eval (called on raw/uncompiled model outside DDP).
        """
        logits = self.forward(xt, sigma)
        logits[:, :, MASK_ID] = NEG_INF  # never predict MASK
        # PAD_ID == TOTAL_VOCAB == 50258, already outside logit range — no clamping needed
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        # Freeze visible (non-MASK) tokens to one-hot log probs
        frozen = torch.full_like(logits, NEG_INF)
        frozen.scatter_(-1, xt.clamp(max=TOTAL_VOCAB - 1).unsqueeze(-1), 0.0)
        return torch.where((xt != MASK_ID).unsqueeze(-1), frozen, logits)

    def teacher_logits(self, x):
        """Bidirectional logits at t→0. Drop-in for forward_logits(x) in unlimited/train.py.

        Runs the model with sigma≈0 (nearly clean input, no masking applied).
        Every token position sees the full bidirectional context.
        Returns (B, T, VOCAB_SIZE) — does NOT include MASK or PAD columns.
        """
        B = x.shape[0]
        sigma = torch.full((B,), 0.01, device=x.device, dtype=torch.float32)
        logits = self.forward(x, sigma)
        logits[:, :, MASK_ID] = NEG_INF
        return logits[..., :VOCAB_SIZE]

    def get_device(self):
        return self.wte.weight.device

# =============================================================================
# MDLM training loss
# =============================================================================

def mdlm_loss(model, x0):
    """Continuous-time absorbing-mask ELBO loss.

    model: the (possibly DDP-wrapped) DiffusionLM — called via model(xt, sigma)
           so DDP gradient all-reduce fires correctly during backward.
    x0:    (B, T) int64 batch — real tokens, EOS_ID, and PAD_ID.
    """
    B = x0.shape[0]
    # Antithetic sampling: pair (t, 1-t) to reduce gradient variance
    t_half = torch.rand(B // 2 + 1, device=x0.device)
    t = torch.cat([t_half, 1 - t_half])[:B].clamp(1e-5, 1 - 1e-5)

    sigma, alpha = log_linear_noise(t)

    # EOS and PAD never enter diffusion — always visible
    is_special = (x0 == EOS_ID) | (x0 == PAD_ID)
    move = (torch.rand_like(x0.float()) < (1 - alpha)[:, None]) & ~is_special
    xt = torch.where(move, MASK_ID, x0)

    # Forward through DDP wrapper
    logits = model(xt, sigma)                        # (B, T, TOTAL_VOCAB)

    # --- inline subs_log_probs ---
    logits[:, :, MASK_ID] = NEG_INF
    logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    frozen = torch.full_like(logits, NEG_INF)
    frozen.scatter_(-1, xt.clamp(max=TOTAL_VOCAB - 1).unsqueeze(-1), 0.0)
    log_probs = torch.where((xt != MASK_ID).unsqueeze(-1), frozen, logits)
    # ------------------------------

    x0_safe = x0.masked_fill(x0 >= TOTAL_VOCAB, 0)  # clamp PAD for safe gather
    log_p_x0 = torch.gather(log_probs, -1, x0_safe.unsqueeze(-1)).squeeze(-1)

    dsigma = (1 - args.noise_eps) / alpha
    is_masked    = (xt == MASK_ID).float()
    content_mask = (x0 != PAD_ID).float()

    n_content = content_mask.sum().clamp(min=1)
    return (dsigma[:, None] * (-log_p_x0) * is_masked * content_mask).sum() / n_content

# =============================================================================
# Discrete ELBO evaluation
# =============================================================================

@torch.no_grad()
def evaluate_elbo_bpb(raw_model, chunk_tensors, token_bytes_np, device,
                      n_seqs=256, n_steps=128):
    """Estimate BPB via discrete absorbing-mask ELBO on random val chunks.

    raw_model: unwrapped DiffusionLM (not DDP, not compiled).
    token_bytes_np: np.int32 array of length VOCAB_SIZE mapping token → byte count.
    """
    raw_model.eval()
    total_bits  = 0.0
    total_bytes = 0
    rng = np.random.default_rng(0)
    indices = rng.choice(len(chunk_tensors), size=min(n_seqs, len(chunk_tensors)), replace=False)

    t_grid = torch.arange(1, n_steps + 1, device=device, dtype=torch.float32) / n_steps
    sigma_grid, alpha_grid = log_linear_noise(t_grid)

    for idx in indices:
        x0 = chunk_tensors[idx].unsqueeze(0).to(device)  # (1, T)
        is_special   = (x0 == EOS_ID) | (x0 == PAD_ID)
        content_mask = (x0 != PAD_ID).float()
        x0_safe      = x0.masked_fill(x0 >= TOTAL_VOCAB, 0)

        # Terminal KL contribution
        alpha_T   = float(alpha_grid[-1])
        n_content = int(content_mask.sum().item())
        seq_bits  = n_content * alpha_T * math.log(VOCAB_SIZE) / math.log(2)

        alpha_prev = 1.0
        for step in range(n_steps):
            alpha_curr = float(alpha_grid[step])
            sigma_curr = sigma_grid[step:step + 1].expand(1)
            move = (torch.rand_like(x0.float()) < (1 - alpha_curr)) & ~is_special
            xt   = torch.where(move, MASK_ID, x0)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                log_probs = raw_model.subs_log_probs(xt, sigma_curr)
            log_p_x0  = torch.gather(log_probs.float(), -1, x0_safe.unsqueeze(-1)).squeeze(-1)
            reveal_prob = (alpha_prev - alpha_curr) / max(1.0 - alpha_curr, 1e-12)
            is_masked   = (xt == MASK_ID).float()
            seq_bits   += (reveal_prob * (-log_p_x0) * is_masked * content_mask / math.log(2)).sum().item()
            alpha_prev  = alpha_curr

        # Accumulate bytes (use token_bytes_np LUT on content tokens)
        y_flat   = x0_safe.view(-1).cpu().numpy()
        mask_cpu = content_mask.view(-1).bool().cpu().numpy()
        total_bytes += token_bytes_np[y_flat[mask_cpu]].sum()
        total_bits  += seq_bits

    raw_model.train()
    return total_bits / max(total_bytes, 1)

# =============================================================================
# DataLoader (slowrun .pt format, doc-aware chunking)
# =============================================================================

def load_chunks(filepath):
    """Load a slowrun .pt file and build doc-aware padded chunk tensors.

    Each document runs from doc_starts[k] (the BOS=50256 token) to
    doc_starts[k+1]-1. Documents are split into seq_len chunks; tail
    chunks shorter than SEQ_LEN are right-padded with PAD_ID.
    """
    data       = torch.load(filepath, weights_only=True)
    tokens_np  = data["tokens"].numpy().astype(np.int64)
    doc_starts = data["doc_starts"].numpy().astype(np.int64)

    # Sentinel: append len(tokens) so the last doc has a defined end
    doc_ends = np.concatenate([doc_starts[1:], [len(tokens_np)]])

    chunk_tensors = []
    for k in range(len(doc_starts) - 1):
        start = int(doc_starts[k])          # position of BOS token (inclusive)
        end   = int(doc_ends[k])            # exclusive end of this doc

        pos = start
        while pos < end:
            chunk_end = min(pos + SEQ_LEN, end)
            length    = chunk_end - pos
            buf       = np.full(SEQ_LEN, PAD_ID, dtype=np.int64)
            buf[:length] = tokens_np[pos:chunk_end]
            chunk_tensors.append(torch.from_numpy(buf))
            pos = chunk_end

    return chunk_tensors


def sample_batch(chunk_tensors, batch_size, device, rng):
    indices = rng.integers(0, len(chunk_tensors), size=batch_size)
    return torch.stack([chunk_tensors[i] for i in indices]).to(device)

# =============================================================================
# LR schedule
# =============================================================================

def get_lr(step, num_iterations):
    warmup   = round(0.05 * num_iterations)
    warmdown = round(0.20 * num_iterations)
    if step < warmup:
        return args.lr * (step + 1) / max(warmup, 1)
    elif step <= num_iterations - warmdown:
        return args.lr
    else:
        progress = (num_iterations - step) / max(warmdown, 1)
        return args.lr * progress  # linear decay to 0

# =============================================================================
# Main
# =============================================================================

def main():
    ddp, rank, local_rank, world_size = get_dist_info()

    if ddp and torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(42 + rank)
    np.random.seed(42 + rank)

    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    # --- Wandb (rank 0 only) ---
    run_name  = args.run if args.run else f"mdlm_{time.strftime('%Y%m%d_%H%M%S')}"
    _wandb_kw = {"project": "slowrun", "name": run_name}
    if args.wandb_group:
        _wandb_kw["group"] = args.wandb_group
    if args.wandb_offline:
        _wandb_kw["mode"] = "offline"
    wandb_run = DummyWandb() if rank != 0 else wandb.init(**_wandb_kw)

    # --- Token bytes LUT for BPB eval ---
    encoder        = tiktoken.get_encoding("gpt2")
    eot_id         = encoder._special_tokens["<|endoftext|>"]
    token_bytes_np = np.array([
        0 if i == eot_id else len(encoder.decode_single_token_bytes(i))
        for i in range(VOCAB_SIZE)
    ], dtype=np.int32)

    # --- Data ---
    train_path = args.input_bin     if args.input_bin     else os.path.join(DATA_DIR, "fineweb_train.pt")
    val_path   = args.input_val_bin if args.input_val_bin else os.path.join(DATA_DIR, "fineweb_val.pt")
    print0(f"Loading data...")
    train_chunks = load_chunks(train_path)
    val_chunks   = load_chunks(val_path)
    print0(f"  Train: {len(train_chunks):,} chunks  |  Val: {len(val_chunks):,} chunks")

    rng = np.random.default_rng(42 + rank)

    # --- Model ---
    print0(f"\n{'='*60}")
    print0(f"MDLM  {args.n_layer}L  {args.n_embd}d  {args.n_head}h")
    print0(f"{'='*60}")
    raw_model = DiffusionLM(args.n_layer, args.n_embd, args.n_head).to(device)
    n_params  = sum(p.numel() for p in raw_model.parameters())
    print0(f"Parameters: {n_params:,}")
    print0(f"EOS_ID={EOS_ID} (never masked)  MASK_ID={MASK_ID}  PAD_ID={PAD_ID}")

    compiled_model = torch.compile(raw_model, dynamic=False)

    if ddp:
        compiled_model = torch.nn.parallel.DistributedDataParallel(
            compiled_model, device_ids=[local_rank]
        )

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        compiled_model.parameters(),
        lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay, fused=True
    )

    # --- Training schedule ---
    tokens_per_step = args.device_batch_size * args.grad_accum * world_size * SEQ_LEN
    total_tokens    = len(train_chunks) * SEQ_LEN
    num_iterations  = round(total_tokens * args.num_epochs / tokens_per_step)
    print0(f"Steps: {num_iterations}  |  Tokens/step: {tokens_per_step:,}  |  Epochs: {args.num_epochs}")
    print0(f"{'='*60}")

    # --- Resume ---
    start_step = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        raw_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt.get("step", 0)
        print0(f"Resumed from step {start_step}")

    # --- Training loop ---
    compiled_model.train()
    smooth_loss = 0.0
    t0 = time.time()
    gc.collect()
    gc.freeze()
    gc.disable()

    for step in range(start_step, num_iterations):
        lr = get_lr(step, num_iterations)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for _ in range(args.grad_accum):
            batch = sample_batch(train_chunks, args.device_batch_size, device, rng)
            with autocast_ctx:
                loss = mdlm_loss(compiled_model, batch) / args.grad_accum
            loss.backward()
            accum_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(compiled_model.parameters(), 1.0)
        optimizer.step()

        ema_beta   = 0.9
        smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * accum_loss
        debiased    = smooth_loss / (1 - ema_beta ** (step - start_step + 1))

        if step % 100 == 0 or step == start_step:
            elapsed = time.time() - t0
            tok_s   = max(step - start_step, 1) * tokens_per_step / elapsed
            pct     = 100 * step / num_iterations
            print0(f"  step {step:05d} ({pct:.1f}%) | loss: {debiased:.4f} | lr: {lr:.1e} | {tok_s/1e3:.0f}K tok/s")
            wandb_run.log({"step": step, "train_loss": debiased, "lr": lr, "tok_s": tok_s})

        # ELBO val eval every 500 steps (rank 0 only, uses raw_model)
        if step > start_step and step % 500 == 0 and rank == 0:
            val_bpb = evaluate_elbo_bpb(
                raw_model, val_chunks, token_bytes_np, device,
                n_seqs=args.eval_elbo_seqs, n_steps=args.eval_elbo_steps,
            )
            print0(f"  [step {step}] Val ELBO BPB: {val_bpb:.4f}")
            wandb_run.log({"step": step, "val_elbo_bpb": val_bpb})

            torch.save({
                "model":        raw_model.state_dict(),
                "optimizer":    optimizer.state_dict(),
                "step":         step,
                "val_elbo_bpb": val_bpb,
                "config": {
                    "n_layer": args.n_layer,
                    "n_embd":  args.n_embd,
                    "n_head":  args.n_head,
                },
            }, args.checkpoint_path)
            print0(f"  Checkpoint → {args.checkpoint_path}")

    # --- Final checkpoint ---
    if rank == 0:
        torch.save({
            "model":     raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step":      num_iterations,
            "config": {
                "n_layer": args.n_layer,
                "n_embd":  args.n_embd,
                "n_head":  args.n_head,
            },
        }, args.checkpoint_path)
        print0(f"\nFinal checkpoint → {args.checkpoint_path}")

    wandb_run.finish()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
