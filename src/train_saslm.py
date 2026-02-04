import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
import numpy as np
from tqdm.auto import tqdm
import os
import glob
from tokenizers import Tokenizer as HFTokenizer

# ==========================================
# 1. Config & Hyperparameters
# ==========================================
@dataclass
class GPTConfig:
    block_size: int = 1024  # Increased for longer context
    vocab_size: int = 32000 # Increased to 32k
    n_layer: int = 12      # Increased to GPT-2 Small scale
    n_head: int = 12       # 768 / 12 = 64 dim per head
    n_embd: int = 768      # GPT-2 Small embedding size
    dropout: float = 0.1
    bias: bool = True

BATCH_SIZE = 64 # A100 40GB/80GB can easily handle this with mixed precision
MAX_ITERS = 5000
EVAL_INTERVAL = 500
LEARNING_RATE = 3e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' # Auto-detect
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps" # For Mac M1/M2/M3

print(f"Using device: {DEVICE}")

# ==========================================
# 2. Tokenizer & Data Loader (MODIFIED)
# ==========================================
def load_data_and_tokenizer():
    # Load our custom tokenizer
    tokenizer = HFTokenizer.from_file("saslm_tokenizer.json")
    print(f"Loaded tokenizer with vocab size: {tokenizer.get_vocab_size()}")

    # Load Text Data
    txt_files = glob.glob("processed_text/*.txt")
    full_text = ""
    for f in tqdm(txt_files, desc="Loading Text Files"):
        with open(f, 'r', encoding='utf-8') as file:
            full_text += file.read() + "\n<|endoftext|>\n"
    
    print(f"Total corpus length: {len(full_text)} characters")

    # Encode
    # We use encode_batch for speed if we had list of lines, but here one big blob
    # Actually better to process file by file to avoid RAM spike?
    # For <100MB corpus, in-memory is fine.
    # Note: Our tokenizer is ByteLevel BPE.
    
    encoded = tokenizer.encode(full_text)
    data = torch.tensor(encoded.ids, dtype=torch.long)
    print(f"Total tokens: {len(data)}")

    # Train/Val Split
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    return train_data, val_data, tokenizer

# ==========================================
# 3. Model Architecture (Original GPT)
# ==========================================
class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.flash = hasattr(F, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                       .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.flash:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.attn_dropout.p if self.training else 0.0, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = LayerNorm(config.n_embd, config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln2 = LayerNorm(config.n_embd, config.bias)
        self.mlp = MLP(config)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight 

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            return logits, loss
        else:
            logits = self.lm_head(x[:, [-1], :])
            return logits, None
            
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# ==========================================
# 4. Training Loop
# ==========================================
def get_batch(split, train_data, val_data, block_size):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (BATCH_SIZE,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y

@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(200) # average over 200 batches
        for k in range(200):
            X, Y = get_batch(split, train_data, val_data, block_size)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def train():
    # Load Data
    train_data, val_data, tokenizer = load_data_and_tokenizer()
    vocab_size = tokenizer.get_vocab_size()

    # Initialize Model
    config = GPTConfig(vocab_size=vocab_size, block_size=512)
    model = GPT(config)
    model.to(DEVICE)
    
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Mixed Precision Scaler
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == 'cuda'))

    for iter in range(MAX_ITERS):
        # Evaluation
        if iter % EVAL_INTERVAL == 0 or iter == MAX_ITERS - 1:
            losses = estimate_loss(model, train_data, val_data, config.block_size)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            
            # Generate Sample
            context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
            print("--- Generated Sample ---")
            print(tokenizer.decode(model.generate(context, max_new_tokens=50)[0].tolist()))
            print("------------------------")
            
            # Save Checkpoint
            torch.save(model.state_dict(), "saslm_checkpoint.pth")
            print("Checkpoint saved to saslm_checkpoint.pth")

        # Training Step
        xb, yb = get_batch('train', train_data, val_data, config.block_size)
        
        # Mixed Precision Forward Pass
        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=(DEVICE == 'cuda')):
            logits, loss = model(xb, yb)
        
        # Scaling
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
    # Save Model
    torch.save(model.state_dict(), "saslm_model-v1.pth")
    print("Model saved to saslm_model.pth")

if __name__ == "__main__":
    train()
