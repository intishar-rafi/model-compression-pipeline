"""
On-Device Model Compression Pipeline scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""On-Device Model Compression Pipeline.

Story: train a teacher CNN on Fashion-MNIST, state a budget in megabytes,
milliseconds and accuracy, then apply pruning, quantization and distillation,
measuring size, latency and accuracy after every stage, and finish with the
pipeline that composes the levers until the budget is met.
"""
import copy
import torch


def main() -> None:
    torch.manual_seed(0)
    data = load_fashion_mnist(n_train=6000, n_test=1000)
    Xtr, ytr, Xte, yte = data["X_train"], data["y_train"], data["X_test"], data["y_test"]

    # ---- 1. The teacher and its numbers ----
    teacher = SmallCNN()
    train_classifier(teacher, Xtr, ytr, epochs=3)
    base = compression_report(teacher, Xte, yte)
    print(f"teacher: {base['params']:,} params, {base['size_mb']} MB fp32, {base['latency_ms']} ms/frame ({fps_from_latency(base['latency_ms'])} FPS), accuracy {base['accuracy']}")

    # ---- 2. Pruning: one shot vs iterative, and what sparse storage costs ----
    one_shot = copy.deepcopy(teacher)
    masks_1 = global_magnitude_prune(one_shot, 0.9)
    acc_one_shot = accuracy(one_shot, Xte, yte)
    iterative = copy.deepcopy(teacher)
    masks_it, hist = iterative_prune(iterative, Xtr, ytr, target=0.9, n_rounds=3)
    print(f"\npruning to 90% sparsity: one-shot accuracy {acc_one_shot:.3f} vs iterative {accuracy(iterative, Xte, yte):.3f}  (schedule {[h[0] for h in hist]})")
    print(f"  sparse storage {sparse_storage_mb(iterative, masks_it)} MB vs dense {model_size_mb(iterative)} MB; break-even sparsity {pruning_breakeven_sparsity()}")

    # ---- 3. Quantization: accuracy and SNR per bit width ----
    print("\nweight quantization of the teacher:")
    table = snr_per_bit(teacher.fc1.weight.detach(), [2, 4, 8])
    for bits in (8, 4, 2):
        q, _ = quantize_model(teacher, bits)
        print(f"  {bits:2d}-bit per-channel: {quantized_size_mb(teacher, bits)} MB, accuracy {accuracy(q, Xte, yte):.3f}, fc1 SNR {quantization_snr_db(teacher.fc1.weight.detach(), bits, per_channel=True)} dB")
    print(f"  measured gain per bit on fc1 (per-tensor): {db_gain_per_bit(table)} dB (theory: 6.02)")
    ranges = activation_ranges(teacher, Xtr[:512])
    print("  calibrated activation absmax: " + ", ".join(f"{k} {v:.2f}" for k, v in ranges.items()))

    # ---- 4. Distillation vs plain training of the same student ----
    cmp = compare_students(teacher, Xtr, ytr, Xte, yte, epochs=2)
    print(f"\nstudent (8,16,32): {cmp['student_kd']['params']:,} params, {cmp['student_kd']['size_mb']} MB")
    print(f"  distilled accuracy {cmp['student_kd']['accuracy']} vs plain {cmp['student_plain']['accuracy']} (gain {cmp['kd_gain']:+.3f}); teacher {cmp['teacher']['accuracy']}")
    print("  (toy scale: the gain moves by a point or two across seeds)")

    # ---- 5. The pipeline against a budget ----
    budget = {"max_mb": 0.03, "max_latency_ms": 1000.0 / 30, "min_accuracy": round(base["accuracy"] - 0.10, 3)}
    print(f"\nbudget: <= {budget['max_mb']} MB, <= {budget['max_latency_ms']:.1f} ms per frame (30 FPS), accuracy >= {budget['min_accuracy']}")
    print("  (latency on this CPU is far under the 30 FPS line, so size and accuracy decide; on a phone profiler this same table is what you would show)")
    result = compress_for_budget(teacher, Xtr, ytr, Xte, yte, budget, epochs=2)
    for name, rep, checks in result["stages"]:
        flags = "".join("Y" if checks[k] else "n" for k in ("size_ok", "latency_ok", "accuracy_ok"))
        print(f"  {name:9s} {rep['sparse_mb']:.4f} MB  {rep['latency_ms']:.3f} ms  acc {rep['accuracy']:.3f}  sparsity {rep['sparsity']:.2f}  [size/latency/acc {flags}]")
    print(f"budget met: {result['ok']} at stage '{result['final_stage']}'")


if __name__ == "__main__":
    main()

