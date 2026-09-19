# On-Device Model Compression Pipeline

> "A 150 MB model runs at 200 ms/frame. You need 30 FPS on a phone. What do you change?"

A from-scratch PyTorch implementation of the three levers used to answer that
question in practice — **pruning**, **quantization**, and **knowledge
distillation** — measured end to end on a real CNN trained on Fashion-MNIST,
composed into a single pipeline that stops as soon as a size/latency/accuracy
budget is met.

## Architecture

The single entry point `scaffold.py` calls `main()`, which walks the same
five-part pipeline built in `model.py`, step by step.

```mermaid
graph TD
    load_fashion_mnist --> SmallCNN
    SmallCNN --> train_classifier
    train_classifier --> model_size_mb
    train_classifier --> measure_latency_ms
    measure_latency_ms --> fps_from_latency

    train_classifier --> global_magnitude_prune
    global_magnitude_prune --> fine_tune_pruned
    fine_tune_pruned --> apply_masks
    global_magnitude_prune --> prune_schedule
    prune_schedule --> iterative_prune
    fine_tune_pruned --> iterative_prune
    iterative_prune --> weight_sparsity
    iterative_prune --> sparse_storage_mb
    sparse_storage_mb --> pruning_breakeven_sparsity

    train_classifier --> quantize_symmetric
    quantize_symmetric --> dequantize
    dequantize --> fake_quantize
    train_classifier --> quantize_per_channel
    quantize_per_channel --> fake_quantize_per_channel
    fake_quantize --> quantization_mse
    fake_quantize --> quantize_model
    fake_quantize_per_channel --> quantize_model
    quantize_model --> quantized_size_mb
    fake_quantize --> quantization_snr_db
    fake_quantize_per_channel --> quantization_snr_db
    quantization_snr_db --> snr_per_bit
    snr_per_bit --> db_gain_per_bit
    train_classifier --> activation_ranges
    activation_ranges --> activation_scales

    train_classifier --> distillation_loss
    distillation_loss --> soft_targets
    distillation_loss --> train_student
    train_student --> compare_students
    train_classifier --> compare_students

    train_classifier --> compression_report
    sparse_storage_mb --> compression_report
    measure_latency_ms --> compression_report
    compression_report --> meets_budget
    iterative_prune --> compress_for_budget
    quantize_model --> compress_for_budget
    train_student --> compress_for_budget
    meets_budget --> compress_for_budget
    compress_for_budget --> generate_report[final budget report]
```

Five stages, each building on the last:

1. **Baseline** — load real Fashion-MNIST, train a teacher `SmallCNN`, measure
   size and latency
2. **Pruning** — global magnitude pruning with masked fine-tuning on a cubic
   sparsity schedule, plus the sparse-storage break-even math
3. **Quantization** — symmetric and per-channel int-N weight quantization,
   calibrated activation ranges, measured SNR per bit width
4. **Distillation** — a narrower student trained on the teacher's soft labels,
   compared against the same student trained on hard labels alone
5. **The pipeline** — `compress_for_budget` composes all three levers,
   scoring each stage against the budget and stopping at the first pass

## How compression works

Conceptually, each lever attacks a different part of what makes a model
expensive: how *many* parameters it has, how many *bits* each one costs, and
how *wide* the architecture is.

```mermaid
graph LR
    T["Teacher model<br/>0.42 MB · fp32<br/>80.7% accuracy"]

    T --> P1["Pruning<br/>zero out smallest-<br/>magnitude weights"]
    P1 --> P2["Masked fine-tuning<br/>keep zeros fixed,<br/>recover accuracy"]
    P2 --> P3["Sparse model<br/>80% sparsity"]

    T --> Q1["Quantization<br/>float32 → int8/int4<br/>per-channel scales"]
    Q1 --> Q2["Calibrate activations<br/>+ measure SNR"]
    Q2 --> Q3["Quantized model<br/>4-8x smaller weights"]

    T --> D1["Distillation<br/>train a narrower<br/>student network"]
    D1 --> D2["Soft labels<br/>from teacher logits"]
    D2 --> D3["Small model<br/>4x fewer params"]

    P3 --> C["Compose:<br/>distill → prune → quantize"]
    Q3 --> C
    D3 --> C
    C --> F["Compressed model<br/>0.016 MB · int8, 80% sparse<br/>76.3% accuracy"]
    F --> B{"Meets budget?<br/>≤0.03 MB, ≤33ms, ≥70.7% acc"}
    B -->|yes| Ship["Ship to phone"]
    B -->|no| More["Compress further<br/>or relax budget"]

    style T fill:#2d2d2d,stroke:#888,color:#fff
    style F fill:#1b4332,stroke:#52b788,color:#fff
    style Ship fill:#1b4332,stroke:#52b788,color:#fff
    style B fill:#333,stroke:#888,color:#fff
```

Pruning shrinks *parameter count* (but only pays off in storage once sparsity
clears the break-even point for your index format). Quantization shrinks
*bits per parameter* (at the cost of precision — SNR drops ~6 dB per bit
removed). Distillation shrinks the *architecture itself* by training a
smaller network to mimic the larger one's soft output distribution, not
just its hard labels. None of them alone gets you to budget here —
composing all three does.

## Results

Teacher CNN: 105,866 params, 0.42 MB (fp32), 80.7% accuracy.

Budget: ≤ 0.03 MB, ≤ 33.3 ms/frame (30 FPS), ≥ 70.7% accuracy.

| Stage     | Size (MB) | Latency (ms) | Accuracy | Sparsity | Budget met |
|-----------|----------:|-------------:|---------:|---------:|:----------:|
| teacher   | 0.4235    | 0.316         | 0.807    | 0.00     | ✗ |
| distilled | 0.1068    | 0.245         | 0.753    | 0.00     | ✗ |
| pruned    | 0.0322    | 0.249         | 0.762    | 0.80     | ✗ |
| quantized | 0.0160    | 0.248         | 0.763    | 0.80     | ✓ |

**Budget met at the `quantized` stage** — distill for a smaller architecture,
prune what's left, quantize what survives. Each lever attacks a different
bottleneck; composing them is what actually closes a 26x size gap.

*(Latency is measured on CPU in this environment, so absolute numbers won't
match a phone — but the relative story per stage, and the methodology for
measuring it, transfers directly.)*

## What each lever bought

**Pruning** — global magnitude pruning, masked fine-tuning, cubic sparsity
schedule:
- One-shot pruning to 90% sparsity: accuracy collapses to 58.6%
- The same target reached iteratively (schedule `[0.633, 0.867, 0.9]`) with
  fine-tuning between rounds: 81.9% — *above* the original teacher
- Sparse storage only pays off past ~33% sparsity at 16-bit indices — a
  90%-sparse tensor stored densely is still full size; this is the
  "zeros still cost bytes" trap naive pruning walks into

**Quantization** — symmetric and per-channel, calibrated activation ranges,
measured SNR:
- 8-bit and 4-bit per-channel: accuracy holds (80.9%, 81.4%)
- 2-bit: falls off a cliff (26.6%)
- fc1 SNR drops 41.2 dB → 16.0 dB → 1.4 dB across 8/4/2-bit
- Measured gain per bit: 5.99 dB, matching the ~6.02 dB theoretical prediction
  for a uniform quantizer

**Distillation** — soft-label training into a narrower student (8, 16, 32):
- Distilled: 75.3% vs. plain hard-label training: 74.0% (+1.3 pts)
- Modest at this data/model scale, but consistent and in the right direction

## Project structure

- `model.py` — every building block: data loading, `SmallCNN`, training,
  pruning, quantization, distillation, measurement, and the composed
  `compress_for_budget` pipeline
- `scaffold.py` — runs the full story end to end and prints the report above

## Running it

```bash
pip install torch numpy
python scaffold.py
```

Downloads Fashion-MNIST (idx files, cached in the system temp dir) on first
run, trains a teacher, and walks through all five stages, printing size,
latency, and accuracy after each.

## Why this exists

Most "model compression" writeups either explain the theory or show a single
before/after number. This project measures every stage of every lever
independently first (so you can see what pruning costs before quantization
even enters the picture), then composes them — which is closer to how the
decision actually gets made in an on-device deployment: given a budget, which
combination of techniques gets you there with the least accuracy lost.
