# Designing LLM Architectures That Survive KV Cache Quantization

**TL;DR:** If your model has `n_embd / n_head < head_dim`, every KV cache
quantization method will fail on it. This is a mathematical property of
your architecture, not a bug in the quantizer. Design for the post-market
pipeline or your users will hit a wall you can't patch.

**Project:** LeanKV — TurboQuant KV cache quantization for LLM inference
**Date:** 2026-04-12
**Based on:** Empirical validation across 5 models, 4 architectures, 4 quantization
levels (2.5–4.5 bits/element), Metal GPU + AVX2 + ARM NEON

---

## The Case Study: Qwen3-4B

We discovered that Qwen3-4B degrades **catastrophically** under TQ3 KV cache
quantization (+48% perplexity increase), while every other model we tested —
including the architecturally identical Qwen3-8B — shows only +3–6%.

The root cause is a single architectural decision:

| | Qwen3-4B | Qwen3-8B |
|---|---|---|
| n_embd | 2560 | 4096 |
| n_head | 32 | 32 |
| **n_embd / n_head** | **80** | **128** |
| head_dim (K/V) | 128 | 128 |
| **Q→KV ratio** | **0.625** | **1.0** |
| TQ3 PPL delta | **+48%** | +6% |

Qwen3-4B projects 80-dimensional Q vectors into 128-dimensional KV space via
W_K. The learned KV representations occupy a **rank-80 subspace** of 128-dim
space. Quantization adds noise in all 128 dimensions — including the 48-dim
orthogonal complement the model never saw during training. The attention
mechanism cannot compensate because it was trained exclusively within the
rank-80 subspace.

**No codebook, rotation, or outlier treatment can fix this.** It is a
mathematical property of the rank-deficient projection, not an implementation
bug. We verified this across three quantization methods (TQ2, TQ3, TQ4) and
with/without Hadamard rotation pre-conditioning.

---

## Cross-Model Validation

| Model | n_embd/n_head | head_dim | Q→KV ratio | TQ3 sensitivity |
|-------|:---:|:---:|:---:|---|
| **Qwen3-4B** | 80 | 128 | **0.625** | **Catastrophic (+48%)** |
| Qwen3-8B | 128 | 128 | 1.0 | Normal (+6%) |
| Llama 3-8B | 128 | 128 | 1.0 | Normal |
| Mistral 7B | 128 | 128 | 1.0 | Normal (+3%) |
| Gemma 3-4B | 320 | 256 | 1.25 | **Robust (improves PPL)** |

Gemma 3-4B is the perfect control: same parameter count as Qwen3-4B, but uses
8 heads with n_embd/n_head = 320 projected to head_dim = 256. The KV space is
fully utilized (ratio 1.25). Result: TQ3 actually **improves** perplexity — the
quantization noise acts as regularization.

---

## The Six Rules for Quantization-Friendly Architectures

### Rule 1: Q-dim to KV-dim ratio must be >= 1.0

```
n_embd / n_head >= n_embd_head_k
```

This is the most important rule. If you want to save parameters, reduce n_head
(use more GQA), don't shrink the Q input dimension below the KV dimension.
A rank-deficient KV subspace cannot be fixed post-training.

### Rule 2: head_dim must be power-of-2

Hadamard rotation is the single most impactful pre-processing step for KV
quantization — it converts arbitrary distributions to near-Gaussian at zero
runtime cost. But it requires power-of-2 dimensions.

Our results: without Hadamard, TQ3 PPL is **130–143 on every model**
(completely broken). With Hadamard, it's +3–6% on well-designed architectures.

Choose 64, 128, 256 — never 80, 96, 160.

### Rule 3: head_dim should align with quantization block sizes

Quantization formats pack elements in blocks of 32, 64, or 128. If head_dim
isn't a clean multiple, blocks span head boundaries and corrupt data.

head_dim = 128 is the current sweet spot — it aligns with every standard
quantization block format.

### Rule 4: W_K/W_V projections must be full-rank

Even if the dimensions technically match (ratio = 1.0), training procedures
or initialization schemes that produce effectively low-rank W_K matrices will
cause the same rank-deficiency problem. Quantization noise in near-null
directions creates spurious attention patterns.

Model makers should verify that their W_K/W_V weight matrices have full
effective rank and that KV activations use the full head_dim space uniformly.

### Rule 5: GQA ratio is a KV memory knob — but has a quantization tradeoff

Fewer KV heads (higher GQA ratio) directly reduces KV memory before any
quantization. But each surviving KV head carries more load, so quantization
error per head has a larger impact on output.

| Strategy | KV memory | Quantization robustness |
|----------|-----------|------------------------|
| Many KV heads + aggressive quant | Low | Risk of per-head error accumulation |
| Few KV heads + mild quant | Low | Each head has more redundancy |
| Few KV heads + large head_dim | Very low | Best of both — Hadamard works better at higher d |

Qwen3.5-9B chose 4 KV heads with head_dim = 256 — aggressive GQA but large
head_dim. This works well: more dimensions = better Hadamard convergence
(concentration of measure makes post-rotation distributions more Gaussian).

### Rule 6: Design for the post-market pipeline

The model maker's job doesn't end at training. The deployment pipeline includes:

- Weight quantization (GPTQ, AWQ, Q4_K_M, etc.)
- KV cache quantization (TurboQuant, KIVI, etc.)
- Speculative decoding
- Pruning and distillation

Each of these assumes certain properties of the architecture. A model designed
purely to minimize training loss without considering these downstream constraints
may be "optimal" on paper but unusable in practice.

Qwen3-4B likely achieves marginally better training loss from its 80→128
projection — but it trades away compatibility with **every general-purpose KV
cache quantization method**.

---

## The Broader Principle

**The architecture IS the API for post-training optimization.**

Designing a model with `n_embd/n_head < head_dim` is like shipping a library
with a broken interface — technically functional, but hostile to every
downstream consumer. Model makers must treat quantization compatibility as a
first-class design constraint, not an afterthought.

The cost of getting this right is zero. Choosing head_dim = n_embd/n_head
(or smaller) instead of an arbitrary larger value doesn't reduce model capacity.
It simply ensures that the KV space is fully utilized — which is better for
training AND for post-training quantization.

---

## What We Implemented

To protect users from silently broken configurations, LeanKV auto-detects the
Q/KV dimension mismatch at model load and downgrades aggressive quantization
types (TQ3, TQ2) to the maximum safe level (TQ4):

```
llama_init_from_model: this model has Q-dim (80) < KV head-dim (128) —
  rank-deficient KV subspace makes aggressive quantization unreliable.
  Downgrading K-cache from tq3_0 to tq4_0
```

This is a guardrail, not a fix. The right fix is in the model architecture.

---

## References

1. Zandieh et al. "TurboQuant: Online Vector Quantization with Near-optimal
   Distortion Rate." arXiv:2504.19874 (2025). Google Research.
2. LeanKV test results: `docs/TQ2-METAL-RESULTS.md`, `docs/RESULTS.md`
3. Source: `src/llama.cpp` — auto-detection at `llama_init_from_model()`
