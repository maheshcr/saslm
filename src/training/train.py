#!/usr/bin/env python3
"""
Main Training Script for SASLM

Orchestrates the complete training pipeline:
- Configuration loading
- Data loading with weighted sampling
- Model creation (from scratch or fine-tune)
- Training with checkpointing and grokking detection
- Evaluation and sample generation

Usage:
    # Train from scratch (Option A)
    python src/training/train.py --config configs/exp_a1_prose_only.yaml

    # Fine-tune GPT-2 (Option B)
    python src/training/train.py --config configs/exp_b1_prose_only_finetune.yaml

    # Resume from checkpoint
    python src/training/train.py --config configs/exp_a1_prose_only.yaml --resume

    # Override settings
    python src/training/train.py --config configs/exp_a1_prose_only.yaml --max-steps 50000

For Colab:
    1. Mount Google Drive
    2. Run this script
    3. It will auto-resume if disconnected
"""

import os
import sys
import math
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.config import load_config, ExperimentConfig, get_device, print_config
from src.training.checkpoint_manager import CheckpointManager
from src.training.grokking_detector import GrokkingDetector


def compute_loss(model, input_ids, labels):
    """
    Compute loss for both custom GPT and HuggingFace models.

    Custom GPT: returns (logits, loss) with positional args
    HuggingFace: returns CausalLMOutput with .loss attribute
    """
    # Check if it's a HuggingFace model
    if hasattr(model, 'config') and hasattr(model.config, 'model_type'):
        # HuggingFace model - use keyword arguments
        outputs = model(input_ids=input_ids, labels=labels)
        return outputs.logits, outputs.loss
    else:
        # Custom GPT model - use positional arguments
        return model(input_ids, labels)
from src.training.metrics_logger import MetricsLogger


# =============================================================================
# Model Definitions
# =============================================================================

class LayerNorm(nn.Module):
    """LayerNorm with optional bias."""
    def __init__(self, ndim: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.1, bias: bool = False):
        super().__init__()
        assert n_embd % n_head == 0

        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout

        # Key, query, value projections
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        # Output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)

        # Regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size))
                             .view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.size()

        # Calculate query, key, values
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Attention with Flash Attention if available
        if hasattr(F, 'scaled_dot_product_attention'):
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            # Manual attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """Feed-forward network."""
    def __init__(self, n_embd: int, dropout: float = 0.1, bias: bool = False):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Transformer block."""
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.1, bias: bool = False):
        super().__init__()
        self.ln_1 = LayerNorm(n_embd, bias=bias)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout, bias)
        self.ln_2 = LayerNorm(n_embd, bias=bias)
        self.mlp = MLP(n_embd, dropout, bias)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """GPT Language Model."""

    def __init__(
        self,
        vocab_size: int,
        block_size: int = 512,
        n_layer: int = 6,
        n_head: int = 6,
        n_embd: int = 384,
        dropout: float = 0.1,
        bias: bool = False,
    ):
        super().__init__()
        self.config = {
            'vocab_size': vocab_size,
            'block_size': block_size,
            'n_layer': n_layer,
            'n_head': n_head,
            'n_embd': n_embd,
            'dropout': dropout,
            'bias': bias,
        }

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(vocab_size, n_embd),
            wpe=nn.Embedding(block_size, n_embd),
            drop=nn.Dropout(dropout),
            h=nn.ModuleList([Block(n_embd, n_head, block_size, dropout, bias) for _ in range(n_layer)]),
            ln_f=LayerNorm(n_embd, bias=bias),
        ))
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Report number of parameters
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Model parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config['block_size'], f"Sequence length {t} > block size {self.config['block_size']}"

        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # Forward pass
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # Calculate loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Generate tokens autoregressively."""
        for _ in range(max_new_tokens):
            # Crop to block size
            idx_cond = idx if idx.size(1) <= self.config['block_size'] else idx[:, -self.config['block_size']:]

            # Forward pass
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Optional top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


# =============================================================================
# Data Loading
# =============================================================================

def create_dataloaders(config: ExperimentConfig, tokenizer, device: str):
    """Create train and validation dataloaders."""
    from src.data.weighted_sampler import WeightedCorpusSampler
    from src.data.data_loader import SASLMDataset
    from torch.utils.data import DataLoader

    print("\nLoading corpus...")

    # Create sampler
    sampler = WeightedCorpusSampler(
        corpus_dir=config.data.corpus_path,
        period_weights=config.data.period_weights,
        importance_weights={
            'core': 3.0,
            'supplementary': 1.0,
            'reference': 0.3,
        },
        chunk_size=config.model.block_size * 4,  # Larger chunks for variety
    )

    include_poetry = 'poetry' in config.content
    sampler.load(include_prose=True, include_poetry=include_poetry, verbose=True)
    sampler.print_statistics()

    # Split into train/val
    train_chunks, val_chunks = sampler.get_train_val_split(
        val_ratio=config.data.val_split,
        seed=config.data.seed,
    )

    print(f"\nCreating datasets...")
    print(f"  Train chunks: {len(train_chunks):,}")
    print(f"  Val chunks: {len(val_chunks):,}")

    # Create datasets
    train_dataset = SASLMDataset(
        chunks=train_chunks,
        tokenizer=tokenizer,
        seq_length=config.model.block_size,
    )

    val_dataset = SASLMDataset(
        chunks=val_chunks,
        tokenizer=tokenizer,
        seq_length=config.model.block_size,
    )

    print(f"  Train sequences: {len(train_dataset):,}")
    print(f"  Val sequences: {len(val_dataset):,}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.hardware.num_workers,
        pin_memory=config.hardware.pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.hardware.num_workers,
        pin_memory=config.hardware.pin_memory,
    )

    return train_loader, val_loader


def load_tokenizer(config: ExperimentConfig):
    """Load tokenizer based on config.

    For fine-tuning (approach='finetune'), uses HuggingFace tokenizer
    to ensure compatibility with pretrained model vocabulary.

    For from-scratch training, uses custom tokenizer.
    """
    if config.approach == 'finetune':
        # Use HuggingFace tokenizer for fine-tuning
        from transformers import AutoTokenizer

        base_model = config.model.base_model
        print(f"Loading HuggingFace tokenizer for {base_model}...")

        hf_tokenizer = AutoTokenizer.from_pretrained(base_model)

        # Wrap in a compatibility class that has same interface as tokenizers.Tokenizer
        class HFTokenizerWrapper:
            def __init__(self, hf_tok):
                self.hf_tokenizer = hf_tok
                # Set pad token if not set
                if self.hf_tokenizer.pad_token is None:
                    self.hf_tokenizer.pad_token = self.hf_tokenizer.eos_token

            def get_vocab_size(self):
                return len(self.hf_tokenizer)

            def encode(self, text):
                # Return object with .ids attribute for compatibility
                class Encoding:
                    def __init__(self, ids):
                        self.ids = ids
                encoded = self.hf_tokenizer.encode(text, add_special_tokens=False)
                return Encoding(encoded)

            def decode(self, ids):
                return self.hf_tokenizer.decode(ids)

            def token_to_id(self, token):
                return self.hf_tokenizer.convert_tokens_to_ids(token)

        tokenizer = HFTokenizerWrapper(hf_tokenizer)
        print(f"Loaded tokenizer: {tokenizer.get_vocab_size():,} tokens")
        return tokenizer

    else:
        # Use custom tokenizer for from-scratch training
        from tokenizers import Tokenizer

        tokenizer_path = Path(config.tokenizer.tokenizer_path)

        if tokenizer_path.is_dir():
            tokenizer_file = tokenizer_path / "tokenizer.json"
        else:
            tokenizer_file = tokenizer_path

        if not tokenizer_file.exists():
            raise FileNotFoundError(
                f"Tokenizer not found at {tokenizer_file}. "
                f"Run: python src/data/train_tokenizer.py --corpus {config.data.corpus_path}"
            )

        tokenizer = Tokenizer.from_file(str(tokenizer_file))
        print(f"Loaded tokenizer: {tokenizer.get_vocab_size():,} tokens")

        return tokenizer


# =============================================================================
# Training Functions
# =============================================================================

def create_model(config: ExperimentConfig, vocab_size: int, device: str) -> nn.Module:
    """Create model based on config."""

    if config.approach == 'from_scratch':
        print("\nCreating model from scratch...")
        model = GPT(
            vocab_size=vocab_size,
            block_size=config.model.block_size,
            n_layer=config.model.n_layers,
            n_head=config.model.n_heads,
            n_embd=config.model.n_embd,
            dropout=config.model.dropout,
            bias=config.model.bias,
        )

    elif config.approach == 'finetune':
        print(f"\nLoading pretrained model: {config.model.base_model}...")
        from transformers import GPT2LMHeadModel, GPT2Config

        # Load pretrained
        model = GPT2LMHeadModel.from_pretrained(config.model.base_model)

        # Resize embeddings if vocab changed
        if model.config.vocab_size != vocab_size:
            print(f"  Resizing embeddings: {model.config.vocab_size} -> {vocab_size}")
            model.resize_token_embeddings(vocab_size)

        # Freeze layers if specified
        if config.model.freeze_layers > 0:
            print(f"  Freezing first {config.model.freeze_layers} layers")
            for i, block in enumerate(model.transformer.h[:config.model.freeze_layers]):
                for param in block.parameters():
                    param.requires_grad = False

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Trainable parameters: {n_params:,}")

    else:
        raise ValueError(f"Unknown approach: {config.approach}")

    model = model.to(device)
    return model


def create_optimizer(model: nn.Module, config: ExperimentConfig) -> torch.optim.Optimizer:
    """Create optimizer with weight decay."""
    # Separate parameters that should and shouldn't have weight decay
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'bias' in name or 'ln' in name or 'layernorm' in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optim_groups = [
        {'params': decay_params, 'weight_decay': config.training.weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ]

    # Use fused AdamW if available (faster on CUDA)
    use_fused = torch.cuda.is_available()
    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=config.training.learning_rate,
        betas=(config.training.beta1, config.training.beta2),
        fused=use_fused,
    )

    return optimizer


def create_scheduler(optimizer, config: ExperimentConfig, num_training_steps: int):
    """Create learning rate scheduler."""
    warmup_steps = config.training.warmup_steps

    if config.training.lr_scheduler == 'cosine':
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / (num_training_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    elif config.training.lr_scheduler == 'linear':
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / (num_training_steps - warmup_steps)
            return 1.0 - progress

    elif config.training.lr_scheduler == 'constant':
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            return 1.0

    else:
        raise ValueError(f"Unknown scheduler: {config.training.lr_scheduler}")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(model, val_loader, device, max_batches: int = None) -> Tuple[float, float]:
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for i, batch in enumerate(val_loader):
        if max_batches and i >= max_batches:
            break

        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        with autocast(enabled=device != 'cpu'):
            _, loss = compute_loss(model, input_ids, labels)

        total_loss += loss.item() * input_ids.size(0)
        total_tokens += input_ids.size(0)

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf')

    model.train()
    return avg_loss, perplexity


@torch.no_grad()
def generate_samples(model, tokenizer, prompts: list, device: str, max_tokens: int = 100) -> list:
    """Generate text samples from prompts."""
    model.eval()
    samples = []

    # Check if it's a HuggingFace model for proper generation settings
    is_hf_model = hasattr(model, 'config') and hasattr(model.config, 'model_type')

    for prompt in prompts:
        # Encode prompt
        encoded = tokenizer.encode(prompt)
        input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)

        # Generate with appropriate settings
        if is_hf_model:
            # HuggingFace model - set pad_token_id to avoid warnings
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=0.8,
                top_k=50,
                do_sample=True,
                pad_token_id=model.config.eos_token_id,
            )
        else:
            # Custom GPT model
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=0.8,
                top_k=50,
            )

        # Decode
        generated = tokenizer.decode(output_ids[0].tolist())
        samples.append(generated)

    model.train()
    return samples


# =============================================================================
# Main Training Loop
# =============================================================================

def train(config: ExperimentConfig, resume: bool = False):
    """Main training function."""

    # Setup device
    device = get_device(config.hardware)
    print(f"\nUsing device: {device}")

    # Load tokenizer
    tokenizer = load_tokenizer(config)
    vocab_size = tokenizer.get_vocab_size()

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config, tokenizer, device)

    # Create model
    model = create_model(config, vocab_size, device)

    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config, config.training.max_steps)

    # Setup training tools
    checkpoint_mgr = CheckpointManager(
        experiment_name=config.name,
        base_path=config.hardware.drive_path,
        save_every_n_steps=config.training.save_interval,
    )

    grokking_detector = GrokkingDetector(
        window_size=config.grokking.detection_window,
        drop_threshold=config.grokking.detection_threshold,
    ) if config.grokking.enabled else None

    logger = MetricsLogger(
        experiment_name=config.name,
        log_dir='./logs',
        console_interval=100,
    )

    # Mixed precision scaler
    scaler = GradScaler() if device == 'cuda' else None

    # Resume from checkpoint if requested
    start_step = 0
    if resume:
        start_step, last_metrics = checkpoint_mgr.load(model, optimizer, scheduler, device)
        if start_step > 0:
            print(f"\nResumed from step {start_step:,}")

    # Log config
    logger.log_config(config.to_dict())
    logger.log_model_summary(model)

    # Sample prompts for generation
    sample_prompts = [
        "The Supermind is",
        "The psychic being",
        "In the process of spiritual evolution,",
        "The goal of Integral Yoga is",
        "Consciousness in its nature is",
    ]

    # Training loop
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)

    model.train()
    train_iter = iter(train_loader)
    accumulation_steps = config.training.gradient_accumulation

    step = start_step
    accumulated_loss = 0.0
    start_time = time.time()

    while step < config.training.max_steps:
        optimizer.zero_grad()

        # Accumulate gradients
        for micro_step in range(accumulation_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass with mixed precision
            if scaler:
                with autocast():
                    _, loss = compute_loss(model, input_ids, labels)
                    loss = loss / accumulation_steps
                scaler.scale(loss).backward()
            else:
                _, loss = compute_loss(model, input_ids, labels)
                loss = loss / accumulation_steps
                loss.backward()

            accumulated_loss += loss.item()

        # Gradient clipping
        if scaler:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)

        # Optimizer step
        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        scheduler.step()

        step += 1

        # Logging
        if step % 100 == 0:
            lr = scheduler.get_last_lr()[0]
            logger.log(step, {
                'train_loss': accumulated_loss,
                'learning_rate': lr,
            })

        accumulated_loss = 0.0

        # Evaluation
        if step % config.training.eval_interval == 0:
            val_loss, perplexity = evaluate(model, val_loader, device, max_batches=50)

            metrics = {
                'val_loss': val_loss,
                'perplexity': perplexity,
            }

            logger.log(step, metrics, eval_metrics=True)

            # Grokking detection
            if grokking_detector:
                # Get recent train loss average
                train_losses = logger.get_metric('train_loss')
                recent_train_loss = sum(train_losses[-10:]) / min(10, len(train_losses)) if train_losses else 0

                event = grokking_detector.update(step, recent_train_loss, val_loss)
                if event:
                    print(f"\n{'='*60}")
                    print(f"GROKKING DETECTED at step {event.step:,}!")
                    print(f"Val loss: {event.val_loss_before:.4f} -> {event.val_loss_after:.4f}")
                    print(f"{'='*60}\n")

            # Save checkpoint
            checkpoint_mgr.save(
                model, optimizer, step,
                metrics={'train_loss': accumulated_loss, 'val_loss': val_loss},
                scheduler=scheduler,
                config=config.to_dict(),
            )

            # Check for best model
            checkpoint_mgr.save_best(model, step, val_loss)

        # Generate samples
        if step % config.training.sample_interval == 0:
            print("\n--- Generated Samples ---")
            samples = generate_samples(model, tokenizer, sample_prompts[:3], device)
            for prompt, sample in zip(sample_prompts[:3], samples):
                print(f"\nPrompt: {prompt}")
                print(f"Output: {sample[:300]}...")

            checkpoint_mgr.save_samples(step, samples, sample_prompts[:3])

    # Final evaluation
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)

    val_loss, perplexity = evaluate(model, val_loader, device)
    print(f"\nFinal val loss: {val_loss:.4f}")
    print(f"Final perplexity: {perplexity:.2f}")

    # Save final checkpoint
    checkpoint_mgr.save(
        model, optimizer, step,
        metrics={'val_loss': val_loss, 'perplexity': perplexity},
        scheduler=scheduler,
    )

    # Log summary
    logger.print_summary()
    logger.save_summary()

    # Grokking analysis
    if grokking_detector:
        analysis = grokking_detector.get_analysis()
        print("\nGrokking analysis:")
        for k, v in analysis.items():
            print(f"  {k}: {v}")

    return model


# =============================================================================
# Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train SASLM')
    parser.add_argument('--config', required=True, help='Path to config YAML')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--max-steps', type=int, default=None, help='Override max steps')
    parser.add_argument('--batch-size', type=int, default=None, help='Override batch size')
    parser.add_argument('--lr', type=float, default=None, help='Override learning rate')

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Apply overrides
    if args.max_steps:
        config.training.max_steps = args.max_steps
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.lr:
        config.training.learning_rate = args.lr

    # Print config
    print_config(config)

    # Train
    train(config, resume=args.resume)


if __name__ == '__main__':
    main()
