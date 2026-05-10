# Transformers

Transformer attention mechanisms implemented from scratch in PyTorch. Every function is heavily commented — the focus is on understanding *why* each step works, not just *what* it does.

## Contents

### `attention.py` — Scaled Dot-Product Attention
Implementation of the core attention formula:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Covers:
- Q/K/V tensor layout `(batch, heads, seq_len, d_k)`
- Attention weight matrix computation
- Masking with `masked_fill(..., -inf)`
- Causal (look-ahead) mask demo

### `multihead_attention.py` — Multi-Head Attention
Full `nn.Module` implementation:
- Projection matrices `W_Q`, `W_K`, `W_V`, `W_O`
- `split_heads` / `combine_heads` with `.view()` and `.transpose()`
- All 8 heads computed in a single parallel `matmul`
- Parameter count breakdown (~1M params for d_model=512, h=8)

### `masks.py` — Padding & Causal Masks
- `make_padding_mask(seq, pad_idx)` — masks PAD tokens so they don't contribute to attention
- `make_causal_mask(seq_len)` — lower-triangular mask so token `i` can only attend to positions `0..i`

### `positional_encoding.py` — Sinusoidal Positional Encoding
Fixed (non-learnable) positional encoding from *Attention Is All You Need*:

$$PE_{(pos,\,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{model}}}\right), \quad PE_{(pos,\,2i+1)} = \cos(\cdots)$$

Implemented as an `nn.Module` with pre-computed encoding buffer.

### `why_sqrt_dk.py` — Intuition Behind the `1/√d_k` Scaling
Step-by-step numerical experiment showing:
- Without scaling: `Var(q·k) ≈ d_k` → scores blow up → softmax saturates → gradients vanish
- With scaling by `1/√d_k`: variance stays ≈ 1 → stable softmax → healthy gradients

## Key shapes

```
Input:          (batch, seq_len, d_model)
After split:    (batch, h, seq_len, d_k)     d_k = d_model / h
Attention:      (batch, h, seq_len, seq_len)
Output:         (batch, seq_len, d_model)
```

## Requirements

```bash
pip install torch
```

## Reference

Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762), 2017.
