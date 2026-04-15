# Phase 6: Runtime Calibration for KV Cache Quantization

**Status:** Design (2026-04-15)
**Predecessor:** Phase 3.5 shipped V1 adaptive policy but Phase 5 CUDA
validation showed V1's W_K-variance heuristic fails on Llama 3-8B and
Qwen3-8B. Runtime calibration is the proposed fix.

**Core insight:** Static analysis of W_K weights is insufficient to predict
KV quantization sensitivity on all architectures. The sensitivity sometimes
lives in the **actual runtime K values** (which depend on inputs), not in
the weight matrix structure. We need a dynamic signal.

---

## 1. The Llama Mystery (Context)

From Phase 5 CUDA batch validation (Section 19.3 of RESULTS.md):

- **Llama 3-8B spectrum skew**: max/median = 2.78× → LOW (classified as flat)
- **Llama 3-8B V1 behavior**: 30 of 32 layers classified as "no outliers", downgraded to TQ2_0
- **Llama 3-8B TQ2_0 quality**: +71% PPL (dramatically worse than uniform TQ2_1 at +35%)
- **Llama 3-8B V1 adaptive quality**: +65% PPL (almost as bad as uniform TQ2_0)

W_K variance says "flat, safe to downgrade" but actual quantization behavior
is dramatically different. The sensitivity mechanism is **invisible to static
analysis**.

Two falsifiable hypotheses for what's going on:

1. **Input-dependent outliers** — K values during real inference have
   elevated variance on specific channels, even though W_K weights don't.
   Runtime measurement would catch this.

2. **Non-local sensitivity** — attention head rotation, position encoding
   interaction, or cross-layer error amplification. Runtime variance might
   look flat too, in which case we need a different signal entirely
   (Q-side analysis, attention score perturbation tests, etc).

**Phase 6a addresses hypothesis 1 directly.** If the runtime variance is
also flat on Llama, we rule out hypothesis 1 and know to investigate
hypothesis 2 in Phase 7+.

---

## 2. Phase 6a — Runtime Post-Hadamard Variance Instrumentation

**Goal:** Measure actual K cache variance during inference warmup and use
it as the adaptive detection signal instead of W_K row norms.

### 2.1 Design

Add a CLI flag `--kv-calibrate-runtime N` that enables a **warmup
calibration pass**:

1. Load the model with the user's requested K cache type (e.g., `-ctk tq2_0`)
2. **Temporarily override to TQ3_0 or F16** for the first N tokens of inference
   (calibration phase — use higher precision to avoid contaminating the
   measurement with quantization noise)
3. For each attention layer, after Hadamard rotation is applied but before
   quantization, compute per-channel statistics:
   - Online mean
   - Online variance (Welford's algorithm for numerical stability)
   - Max absolute value seen
4. After N tokens, compute per-layer:
   - `sorted_var = sort(channel_var, desc)`
   - `runtime_max_over_median = sorted_var[0] / sorted_var[d/2]`
   - `runtime_n_moderate = count where sorted_var > T × median`
5. Log the runtime spectrum summary per layer
6. **Rebuild the K cache with the runtime-derived per-layer type selection**
   (or simply log the recommendation and require user to re-run)

### 2.2 Implementation Sketch

**File**: `src/llama-runtime-calibrate.cpp` (new file)

```cpp
// Forward declaration
struct runtime_calib_state {
    int n_layer;
    int n_embd_head_k;
    int tokens_processed;
    int calib_window;           // N from --kv-calibrate-runtime N

    // Per-layer Welford accumulators
    std::vector<std::array<double, TQ_OUTLIER_MAX_DIM>> mean;
    std::vector<std::array<double, TQ_OUTLIER_MAX_DIM>> m2;   // sum of squared deviations
    std::vector<int64_t> count_per_layer;
};

// Called from llama_decode before each forward pass
void runtime_calib_observe_k(runtime_calib_state * st,
                              int layer_idx,
                              const ggml_tensor * k_rotated);  // post-Hadamard K

// Called after calib_window tokens — produces report + type_k_l update
void runtime_calib_finalize(runtime_calib_state * st,
                             llama_kv_cache * kv_self);
```

### 2.3 Hook Point

The Hadamard rotation currently happens inside the attention graph builder
(`llama-build-context.cpp`) before the K is written to the cache. We need
to capture the **post-Hadamard K** just before it's stored.

Two options for where to add the observation:

**Option A — Custom ggml op**: Add a `ggml_map_custom1` node in the
attention graph that captures K statistics as a side effect without
affecting the tensor flow. Clean but requires graph modification.

**Option B — Post-computation callback**: After `ggml_graph_compute`
returns, inspect the K cache tensors directly and extract recent additions.
Less clean but simpler to prototype.

**Recommendation**: Start with **Option B** for the initial prototype
(2-3 days), move to Option A if we need per-step observation instead of
per-graph-compute.

### 2.4 Warmup Calibration Protocol

1. User runs: `llama-cli ... -ctk tq2_0 --kv-calibrate-runtime 128 -p "..."`
2. LeanKV allocates K cache with F16 tensors temporarily
3. Processes first 128 tokens normally, observing K values after Hadamard
4. At token 128:
   - Computes per-layer runtime statistics
   - Prints report
   - (Optionally) reallocates K cache with tq2_0/tq2_1/tq3_0 mix based on runtime data
5. Continues processing with the calibrated cache

**Calibration prompt choice**: For reliable results, use a chunk of
WikiText-2 or similar representative text. The baked-in calibration data
question is deferred to Phase 6b.

### 2.5 Expected Output

```
runtime calibration: observed 128 tokens, analyzing K cache distributions...

layer  0: runtime max/med=2.15x, moderate=4, proposed=TQ2_0
layer  1: runtime max/med=2.37x, moderate=6, proposed=TQ2_0
layer  2: runtime max/med=5.82x, moderate=22, proposed=TQ2_1 (promoted!)
layer  3: runtime max/med=3.41x, moderate=14, proposed=TQ2_1 (promoted!)
...
layer 31: runtime max/med=2.04x, moderate=3, proposed=TQ2_0

runtime calibration summary: 0%=18 layers, 12.5%=10, 25%=4, 50%=0
runtime calibration vs W_K-based V1: 4 layers promoted that V1 missed
recommended: rebuild with --kv-outlier-frac -1 ... (uses runtime data)
```

### 2.6 What Counts as Success

**Success case** — runtime signal differs from W_K signal on Llama 3-8B:

- Llama 3-8B runtime max/median > 5× on some layers (vs W_K's 2.78×)
- Runtime V1 promotes layers that W_K-based V1 missed
- Re-running with runtime-derived config beats uniform TQ2_1

This would vindicate the hypothesis and give us a shippable Phase 6a with
a real benefit.

**Failure case** — runtime signal is also flat on Llama 3-8B:

- Llama 3-8B runtime max/median still ~2-3×, no significant channels flagged
- Runtime-derived V1 ≈ W_K-derived V1 ≈ uniform TQ2_0
- PPL is the same

This rules out Phase 6a as a solution to the Llama mystery but still has
value:
- We confirm that Llama's sensitivity is not channel-local
- We move to Phase 7 investigation (W_Q analysis, attention perturbation)
- The runtime instrumentation is still useful for other models

**Either outcome is a publishable finding and advances the project.**

### 2.7 Scope and Timeline

- Prototype (Option B, post-hoc analysis): 2-3 days
- Clean implementation (Option A, online observation): 1 extra day
- Testing on 6 models (Mistral, Qwen3-8B, Gemma 3-4B, Llama 3-8B, Qwen3-4B, Qwen 3.5-9B): 1 day of wall time
- Integration with existing V1 policy: 1 day

**Total: ~1 week of focused work**

### 2.8 Dependencies and Risks

**Dependencies:**
- Nothing blocking. Builds on existing Phase 3.5 adaptive mechanism.

**Risks:**
- The calibration prompt might not be representative — different text could
  produce different runtime distributions. Mitigation: run calibration on
  1K-token chunks of WikiText-2 and average.
- If runtime variance changes over time (e.g., start of generation vs late
  in a long context), a single 128-token calibration might be misleading.
  Mitigation: sample statistics across multiple non-contiguous windows.
- Runtime calibration adds ~seconds to first generation. Acceptable for
  interactive use, might be noticeable in batch serving. Mitigation: run
  calibration once, cache result (Phase 6b).

---

## 3. Phase 6b — Fingerprint Cache (Deferred)

**Status:** Sketched, not yet needed pending Phase 6a results.

### 3.1 Core Idea

Runtime calibration costs seconds per model load. That's fine for
interactive use but wasteful when you load the same model 100 times. Solution:
cache the calibration result per model fingerprint.

### 3.2 Fingerprint Scheme

SHA256 hash of the first 64 MB of the GGUF file (the header + weight
tensors — enough to disambiguate different quantizations of the same model).
Full-file hash is slower and unnecessary.

### 3.3 Cache Location

`~/.cache/leankv/fingerprints.json`:

```json
{
  "version": 1,
  "entries": {
    "sha256:abc123...": {
      "model_name": "Mistral-7B-Instruct-v0.3-Q4_K_M",
      "model_fingerprint_scheme": "first_64mb_sha256",
      "calibration_date": "2026-04-15T01:45:00Z",
      "calibration_method": "runtime_post_hadamard",
      "calibration_tokens": 128,
      "calibration_prompt": "wikitext2_random_chunk_seed_42",

      "recommendation": {
        "metric": 0,
        "threshold": 1.5,
        "per_layer_types": ["tq2_0", "tq2_0", ..., "tq2_1", "tq3_0"],
        "expected_ppl_vs_tq2_1": 0.016
      },

      "v1_heuristic_comparison": {
        "w_k_based": {"threshold": 1.5, "layers_promoted": 21},
        "runtime_based": {"threshold": 1.5, "layers_promoted": 23},
        "agreement": 0.85
      }
    }
  }
}
```

### 3.4 Workflow

1. **First load of new model**: compute fingerprint, check cache, miss.
   Run Phase 6a calibration, store result, apply.
2. **Subsequent loads**: compute fingerprint (fast, ~100ms), cache hit,
   apply cached per-layer types instantly. Zero runtime overhead.
3. **Force recalibration**: `--kv-recalibrate` flag overrides cache.

### 3.5 Timeline

- Fingerprint computation: 0.5 day
- Cache read/write: 0.5 day
- Integration with Phase 6a: 0.5 day
- Testing: 0.5 day

**Total: 2 days of focused work** (after Phase 6a is done)

### 3.6 When to build it

**Only after Phase 6a proves runtime calibration is worth the cost.** If
Phase 6a shows runtime variance ≈ W_K variance on all models, there's no
reason to build 6b because the static V1 policy is already as good as
runtime can do. Phase 6b adds value only when runtime calibration produces
different (and better) recommendations than static analysis.

---

## 4. Phase 6c — Mini-PPL Validator (Deferred)

**Status:** Sketched.

After Phase 6a produces a per-layer type recommendation, we still don't
know for sure that it beats uniform TQ2_1 or TQ3_0 on the target model.
Phase 6c adds a final validation step:

1. Apply the recommended per-layer config
2. Run a mini-PPL test on a baked-in calibration dataset (say 16 chunks)
3. Compare to uniform TQ2_1 and uniform TQ3_0 on the same dataset
4. Store the PPL delta in the fingerprint cache
5. If the recommendation is worse than uniform TQ2_1, fall back to uniform
   TQ2_1 and record the failure

This prevents the Phase 3.5 mistake of shipping a heuristic that *looks*
good but actually degrades quality. Every shipped recommendation is
PPL-validated on a reference dataset.

### 4.1 Dataset question

Baking 1 MB of text into the binary is small enough to be acceptable.
Alternative: download on first calibration, cache separately.

### 4.2 Timeline

- Calibration dataset packaging: 0.5 day
- Mini-PPL runner: 1 day
- Cache integration: 0.5 day

**Total: 2 days**

---

## 5. Dependency Graph and Order

```
Phase 6a (runtime instrumentation)     — 1 week
     │
     ├── Phase 6b (fingerprint cache)  — 2 days [conditional on 6a success]
     │
     └── Phase 6c (mini-PPL validator) — 2 days [conditional on 6a+6b]
              │
              └── Phase 7? (further investigation if Llama mystery persists)
```

**Decision tree**:
1. Build Phase 6a
2. Run on Llama 3-8B + 5 other models
3. **If runtime variance solves the Llama mystery** → build 6b + 6c, ship
4. **If runtime variance is flat on Llama** → mystery requires different approach (Phase 7: W_Q analysis, attention perturbation, or direct PPL-based search over policies)

---

## 6. Open Questions for Implementation

1. **Observation timing**: per-token (online) vs per-batch (offline) vs
   post-graph (batched). Trade-offs in implementation complexity vs
   observation granularity.

2. **Calibration prompt**: hardcoded representative text vs user-provided
   vs randomized from a bundled corpus. Affects reproducibility.

3. **Statistic choice**: variance (second moment) vs IQR (robust) vs full
   histogram (heavy but informative). Welford is fastest; IQR is more
   robust to outliers in the input itself.

4. **Multi-window sampling**: 1 window of 128 tokens vs 4 windows of 32
   tokens from different positions in a long context. Latter is more
   representative but adds complexity.

5. **Handling failed calibration**: what if the user interrupts during
   calibration? Partial data should not corrupt the cache.

6. **Versioning the cache**: if we ship Phase 6a v2 with a different
   statistic, how do we invalidate old cache entries? Probably a
   `calibration_version` field in the cache JSON.

---

## 7. Success Metrics for Phase 6a

- **Primary metric**: Does runtime-derived V1 outperform W_K-derived V1
  on Llama 3-8B? (Measured as 160-chunk PPL delta vs uniform TQ2_1.)
- **Secondary metric**: Does the runtime signal change the recommendation
  on at least 2 of 6 tested models? If all 6 give the same answer as
  W_K, the runtime instrumentation is unnecessary.
- **Tertiary metric**: Calibration overhead — must be < 10 seconds on
  typical hardware for 128-token calibration to be acceptable for
  interactive use.

---

## 8. Files Likely to Change

**New files:**
- `src/llama-runtime-calibrate.cpp` — Welford accumulators + variance tracker
- `src/llama-runtime-calibrate.h` — public API
- `src/llama-fingerprint.cpp` — SHA256 + cache (Phase 6b, later)
- `docs/phase6-runtime-calibration-results.md` — findings after running

**Modified files:**
- `src/llama-build-context.cpp` — hook point for K observation
- `src/llama.cpp` — CLI parsing, integration, reporting
- `common/common.cpp` — `--kv-calibrate-runtime` flag
- `common/common.h` — `int kv_calibrate_runtime_tokens` field

---

## 9. Why This Matters

Phase 3.5 shipped a V1 policy that works on Mistral and is **harmful** on
Llama 3 and Qwen3-8B. That's a real problem for any user who doesn't know
which category their model falls into. Runtime calibration is the only
approach we've proposed that could:

1. Detect Llama-class sensitivity dynamically
2. Produce per-model recommendations automatically
3. Avoid the "V1 is only good on Mistral" ceiling

If Phase 6a works, we can upgrade the shipping recommendation from
"TQ3_0 default, V1 opt-in" to "TQ3_0 default, V1 with runtime
calibration for users who want aggressive compression." That's a
meaningful user-facing improvement.

If Phase 6a doesn't work, we learn something important about Llama's
sensitivity and can focus Phase 7 on the right investigation.

Either way, Phase 6a is the next honest step in the LeanKV research
program.
