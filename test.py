# SOLUTION: Direct dataset loading for Google Colab
# The dataset downloaded successfully, we just need to access it properly

import os
from datasets import load_dataset, Dataset
import tiktoken
import numpy as np
from tqdm.auto import tqdm

print("=== GOOGLE COLAB TINYSTORIES FIX ===")

# Method 1: Force reload with download_mode
def load_dataset_colab_fix():
    """Load dataset with Colab-specific fixes"""
    try:
        print("Attempting Method 1: Force redownload...")
        ds = load_dataset(
            "roneneldan/TinyStories",
            download_mode="force_redownload",
            verification_mode="no_checks"
        )
        print("✅ Method 1 successful!")
        return ds
    except Exception as e:
        print(f"❌ Method 1 failed: {e}")
        
        try:
            print("Attempting Method 2: Load from cache with ignore_verifications...")
            ds = load_dataset(
                "roneneldan/TinyStories",
                ignore_verifications=True
            )
            print("✅ Method 2 successful!")
            return ds
        except Exception as e2:
            print(f"❌ Method 2 failed: {e2}")
            
            try:
                print("Attempting Method 3: Load with trust_remote_code...")
                ds = load_dataset(
                    "roneneldan/TinyStories",
                    trust_remote_code=True,
                    verification_mode="no_checks"
                )
                print("✅ Method 3 successful!")
                return ds
            except Exception as e3:
                print(f"❌ Method 3 failed: {e3}")
                return None

# Try loading the dataset
ds = load_dataset_colab_fix()

if ds is not None:
    print(f"\n📊 Dataset loaded successfully!")
    print(f"Available splits: {list(ds.keys())}")
    for split in ds.keys():
        print(f"  - {split}: {len(ds[split]):,} examples")
        
    # Show sample data
    print(f"\n📝 Sample from train split:")
    print(f"Text preview: {ds['train'][0]['text'][:200]}...")
else:
    print("\n🔄 All methods failed. Trying manual approach...")
    
    # Manual approach - create dataset from scratch if needed
    print("Creating minimal dataset for testing...")
    
    # Create a small sample dataset for testing
    sample_texts = [
        "Once upon a time, there was a little girl who loved to read books.",
        "The sun was shining brightly in the blue sky.",
        "A cat sat on the mat and looked around curiously.",
        "Children played happily in the garden all day long.",
        "The old tree had many branches and green leaves."
    ] * 1000  # Repeat to create more samples
    
    # Create datasets
    train_data = sample_texts[:4000]
    val_data = sample_texts[4000:5000]
    
    ds = {
        'train': Dataset.from_dict({'text': train_data}),
        'validation': Dataset.from_dict({'text': val_data})
    }
    
    print(f"✅ Created sample dataset:")
    print(f"  - train: {len(ds['train']):,} examples")
    print(f"  - validation: {len(ds['validation']):,} examples")

# Now proceed with tokenization
print("\n🔤 Starting tokenization...")

enc = tiktoken.get_encoding("gpt2")

def process(example):
    """Process text examples into token IDs"""
    ids = enc.encode_ordinary(example['text'])
    return {'ids': ids, 'len': len(ids)}

# Check if binary files already exist
if not os.path.exists("train.bin"):
    print("📦 Creating binary files...")
    
    # Tokenize the dataset
    tokenized = {}
    for split_name, split_data in ds.items():
        print(f"Tokenizing {split_name} split...")
        tokenized[split_name] = split_data.map(
            process,
            remove_columns=['text'],
            desc=f"Tokenizing {split_name}",
            num_proc=1,  # Use single process for stability in Colab
        )
    
    # Create binary files
    for split, dset in tokenized.items():
        print(f"Creating {split}.bin...")
        
        # Calculate total length
        lengths = dset['len']
        arr_len = sum(lengths)
        
        filename = f'{split}.bin'
        dtype = np.uint16
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        
        # Write data in batches for memory efficiency
        total_batches = min(1024, len(dset))  # Adjust batch count for smaller datasets
        
        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'Writing {filename}'):
            # Calculate batch indices
            start_idx = (batch_idx * len(dset)) // total_batches
            end_idx = ((batch_idx + 1) * len(dset)) // total_batches
            
            if start_idx < end_idx:
                batch_data = dset[start_idx:end_idx]
                
                # Concatenate all token IDs in this batch
                batch_tokens = []
                for token_list in batch_data['ids']:
                    batch_tokens.extend(token_list)
                
                if batch_tokens:  # Only write if we have tokens
                    arr_batch = np.array(batch_tokens, dtype=dtype)
                    arr[idx:idx + len(arr_batch)] = arr_batch
                    idx += len(arr_batch)
        
        arr.flush()
        print(f"✅ {filename} created with {idx:,} tokens")
    
    print("🎉 Tokenization completed!")
else:
    print("✅ Binary files already exist!")

# Verify the files were created
for split in ['train', 'validation']:
    filename = f'{split}.bin'
    if os.path.exists(filename):
        file_size = os.path.getsize(filename)
        print(f"📁 {filename}: {file_size:,} bytes")
    else:
        print(f"❌ {filename}: NOT FOUND")

print("\n🚀 Ready to proceed with model training!")


import torch

def get_batch(split):
    if split == 'train':
        data = np.memmap('train.bin', dtype=np.unit16, mode='r')
    else:
        data = np.memmap('validation.bin', dtype=np.uint16, mode='r')

    ix = torch.randint(len(data) -block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x,y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x,y = x.to(device), y.to(device)
    return x,y

# Complete GPT Training Code - Fixed for Google Colab
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
import numpy as np
from tqdm.auto import tqdm
from contextlib import nullcontext

# Remove the problematic import DS - it's not needed

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
            # Fixed typo: torch.tril not torch.trill
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
        
    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.flash:
            # Fixed typo: is_causal not is_casual
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

@dataclass
class GPTConfig:
    block_size: int
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float = 0.0
    bias: bool = True

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
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

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
        assert t <= self.config.block_size
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
                logits[logits < v[:, [-1]]] = -float('inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# Training configuration
learning_rate = 1e-4
max_iters = 5000  # Reduced for faster training in Colab
warmup_steps = 500  # Adjusted proportionally
min_lr = 5e-5
eval_iters = 100
batch_size = 16  # Reduced for memory efficiency
block_size = 128
gradient_accumulation_steps = 8  # Reduced proportionally

device = "cuda" if torch.cuda.is_available() else "cpu"
device_type = 'cuda' if 'cuda' in device else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]

ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

print(f"🚀 Training on: {device} ({device_type})")
print(f"📊 Using dtype: {dtype}")

# Set device and seed
torch.cuda.set_device(0) if device_type == 'cuda' else None
torch.manual_seed(42)
if device_type == 'cuda':
    torch.cuda.manual_seed(42)

def get_batch(split):
    """Get a batch of data for training/validation"""
    if split == 'train':
        data = np.memmap('train.bin', dtype=np.uint16, mode='r')
    else:
        data = np.memmap('validation.bin', dtype=np.uint16, mode='r')

    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# Initialize model
config = GPTConfig(
    vocab_size=50257,
    block_size=128,
    n_layer=6,
    n_head=6,
    n_embd=384,
    dropout=0.1,
    bias=True
)

model = GPT(config).to(device)
print(f"📦 Model parameters: {sum(p.numel() for p in model.parameters()):,}")

def estimate_loss(model):
    """Estimate loss on train and validation sets"""
    out = {}
    model.eval()
    with torch.inference_mode():
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                with ctx:
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
    model.train()
    return out

# Setup optimizer and scheduler
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR

# Fixed: Remove 'epochs' parameter from AdamW (it doesn't exist)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.1)

scheduler_warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
scheduler_decay = CosineAnnealingLR(optimizer, T_max=max_iters - warmup_steps, eta_min=min_lr)
scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_decay], milestones=[warmup_steps])

# Fixed: enabled not enambled
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# Training loop
best_val_loss = float('inf')
best_model_params_path = "best_model_params.pt"
train_loss_list, validation_loss_list = [], []

print("🎯 Starting training...")

for epoch in tqdm(range(max_iters), desc="Training"):
    # Evaluation
    if epoch % eval_iters == 0:
        losses = estimate_loss(model)
        print(f"\nEpoch {epoch}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        train_loss_list.append(losses['train'].item())
        validation_loss_list.append(losses['val'].item())

        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            torch.save(model.state_dict(), best_model_params_path)
            print(f"💾 New best model saved! Val loss: {best_val_loss:.4f}")
        
    # Training step
    X, y = get_batch("train")

    with ctx:
        logits, loss = model(X, y)
        loss = loss / gradient_accumulation_steps

    scaler.scale(loss).backward()

    if ((epoch + 1) % gradient_accumulation_steps == 0) or (epoch + 1 == max_iters):
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    scheduler.step()

print("🎉 Training completed!")

# Test generation
import tiktoken
enc = tiktoken.get_encoding("gpt2")

# Load best model
model.load_state_dict(torch.load(best_model_params_path))
model.eval()

def generate_text(prompt, max_tokens=100):
    """Generate text from a prompt"""
    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    with torch.no_grad():
        generated = model.generate(tokens, max_tokens, temperature=0.8, top_k=50)
    
    return enc.decode(generated[0].tolist())

# Test generation
test_prompt = "Once upon a time"
generated_text = generate_text(test_prompt, max_tokens=50)
print(f"\n📝 Generated text:")
print(f"Prompt: '{test_prompt}'")
print(f"Generated: {generated_text}")

print(f"\n📈 Final Results:")
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Training completed successfully! 🚀")