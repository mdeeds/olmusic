# Real-Time MIDI LLM Copilot: Architecture & Design Document

## 1. Project Overview

This project implements a real-time, AI-driven musical co-pilot using a localized Large Language Model (LLM). The system listens to multi-channel MIDI input from live instruments (e.g., Juno 106, Yamaha DX7, MPC) and generates contextual accompaniment in real-time on Channel 0.

The architecture is split into two distinct domains:

-   **Training (Python/PyTorch):** Continuous pre-training and LoRA fine-tuning of an English-language base LLM to understand a custom MIDI vocabulary.
-   **Inference (Rust/Candle):** A low-latency, deterministic execution environment handling concurrent MIDI I/O and manual KV-cache management.

## 2. Core Model Architecture

-   **Base Model:** `allenai/OLMo-2-0425-1B` (or equivalent ~1B parameter causal LM).
-   **Precision:** 16-bit (`bfloat16`) or quantized (8-bit/4-bit `QLoRA`) for the frozen core blocks to fit within 16GB VRAM.
-   **Context Window:** Small sliding window (starting at `~256` tokens, to be empirically scaled based on hardware latency constraints).

## 3. Data Representation & Tokenization

Standard text-based BPE tokenizers are replaced with a custom Byte-Level BPE tokenizer trained on pre-processed MIDI event text.

### 3.1 Custom Tokenizer Setup

-   **Library:** Hugging Face `tokenizers` (`ByteLevel` pre-tokenizer).
-   **Vocabulary Size:** `~1024` to `8192` tokens.
-   **Constraint:** The model's embedding matrix (`embed_tokens`) and output head (`lm_head`) MUST be resized.

### 3.2 Event Serialization & Sorting

LLMs require strict sequential ordering. Simultaneous MIDI events (occurring at the same time delta) MUST be sorted deterministically before tokenization:

1.  **Time Delta** (Always first)
2.  **Event Type** (Note-Offs BEFORE Note-Ons to prevent hanging notes)
3.  **Channel** (Channel 0 precedes Channel 1, etc.)
4.  **Pitch** (Lowest to highest)

### 3.3 Hierarchical Time Encoding (Grid-Offset)

To preserve the small context window, continuous clock pulses (`0xF8`) and unary Run-Length Encoding are discarded.

Time is encoded using a structural grid token followed by a syncopation/offset token.

-   **Example:** A 14-tick delay is encoded as `P_12 P_2` (where `P_12` represents an eighth-note grid anchor, and `P_2` is the humanized offset).
-   **Requirement:** Ensure spaces between grid and offset tokens during BPE training to prevent the algorithm from merging them into a single token (e.g., `P_12P_2`).

### 3.4 Data Filtering

-   **Continuous Controllers (CC):** Pitch bends, mod wheels, and all other non-essential CC messages MUST be stripped from both training data and the real-time inference loop to prevent context window exhaustion.

## 4. Training Pipeline (Python)

Training is conducted in Python using `transformers` and `peft`. Because the model is shifting from English text to MIDI tokens, a two-stage Continuous Pre-training strategy is required.

### 4.1 Two-Stage Training

-   **Phase 1: Vocabulary Alignment (Pre-training).** Train on a massive generic multi-track MIDI corpus (e.g., Lakh MIDI). This teaches the newly initialized embedding layer fundamental musical grammar.
-   **Phase 2: Style Transfer (Fine-tuning).** Train on the target dataset (e.g., 1000 Yamaha 12-bar blues styles, multi-part, transposed across all 12 keys). Context windows should begin with structural metadata tokens (e.g., `[GENRE_BLUES]`, `[PART_PIANO]`).

### 4.2 LoRA Configuration

The 1B transformer blocks are frozen. Only the new vocabulary embeddings and attention adapters are trained.

-   **Fully Unfrozen Modules:** `embed_tokens`, `lm_head` (CRITICAL: Required because vocabulary changed).
-   **LoRA Target Modules:** `q_proj`, `k_proj`, `v_proj`, `o_proj`.
-   **Export:** Weights are merged and exported strictly as `.safetensors` for Rust compatibility.

## 5. Inference Engine (Rust)

The live loop is written in Rust to avoid Python's GIL and Garbage Collection latency spikes.

### 5.1 Dependencies

-   `candle-core`, `candle-transformers` (Hugging Face Rust ML framework).
-   `tokenizers` (Native Rust execution of the custom `tokenizer.json`).
-   `midir` (Real-time concurrent MIDI I/O).

### 5.2 Context Window & KV Cache Management

-   **CONSTRAINT:** Do NOT use a continuously rolling sliding window with standard KV cache. Standard Rotary Position Embeddings (RoPE) break if absolute token positions shift leftward while retaining old cached Key tensors.
-   **Solution: Periodic Cache Flushing**
    1.  Generate tokens using `use_cache=True` until the context limit (e.g., 512 tokens) is reached.
    2.  Truncate the `input_ids` array (e.g., drop the oldest 50%).
    3.  Discard the KV Cache (`past_key_values = None`).
    4.  Run a complete forward pass (prefill) on the truncated sequence to rebuild the KV cache with correct `RoPE` alignments.
    5.  Resume generation.

### 5.3 Multi-Channel Routing

-   **Input:** The Rust app listens to external hardware (Channels 1-15), intercepts the MIDI, tokenizes it, and injects it into the active context window.
-   **Output:** The LLM consistently generates its accompaniment explicitly tagged for Channel 0.

## 6. Variables for Empirical Tuning

During development, the following trade-offs will require experimentation:

-   **Context Window Size vs. Inference Latency:** A larger window captures chord progressions better but increases the prefill latency during Periodic Cache Flushing.
-   **Time Encoding Granularity:** Testing the balance between vocabulary bloat (absolute time shifts) vs. context window consumption (Grid-Offset encoding requires 2 tokens per time shift).
