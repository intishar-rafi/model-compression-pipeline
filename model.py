"""
On-Device Model Compression Pipeline

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_fashion_mnist
import os
import gzip
import tempfile
import urllib.request
import numpy as np
import torch

BASE_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def _download(fname):
    dest = os.path.join(tempfile.gettempdir(), fname)
    if not os.path.exists(dest):
        urllib.request.urlretrieve(BASE_URL + fname, dest)
    return dest


def _read_images(path):
    with gzip.open(path, "rb") as f:
        buf = f.read()
    arr = np.frombuffer(buf, dtype=np.uint8, offset=16)
    return arr.reshape(-1, 1, 28, 28)


def _read_labels(path):
    with gzip.open(path, "rb") as f:
        buf = f.read()
    return np.frombuffer(buf, dtype=np.uint8, offset=8)


def load_fashion_mnist(n_train=6000, n_test=1000):
    train_images_path = _download(FILES["train_images"])
    train_labels_path = _download(FILES["train_labels"])
    test_images_path = _download(FILES["test_images"])
    test_labels_path = _download(FILES["test_labels"])

    X_train_full = _read_images(train_images_path)
    y_train_full = _read_labels(train_labels_path)
    X_test_full = _read_images(test_images_path)
    y_test_full = _read_labels(test_labels_path)

    X_train = X_train_full[:n_train].astype(np.float32) / 255.0
    y_train = y_train_full[:n_train].astype(np.int64)
    X_test = X_test_full[:n_test].astype(np.float32) / 255.0
    y_test = y_test_full[:n_test].astype(np.int64)

    return {
        "X_train": torch.from_numpy(X_train).to(torch.float32),
        "y_train": torch.from_numpy(y_train).to(torch.int64),
        "X_test": torch.from_numpy(X_test).to(torch.float32),
        "y_test": torch.from_numpy(y_test).to(torch.int64),
    }

# Step 2 - SmallCNN
import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallCNN(nn.Module):
    def __init__(self, c1=16, c2=32, hidden=64, n_classes=10):
        super().__init__()
        # TODO: conv1, conv2 (3x3, padding 1), fc1 (c2*7*7 -> hidden), fc2 (hidden -> n_classes); store c1, c2, hidden
        self.c1, self.c2, self.hidden = c1, c2, hidden
        self.conv1 = nn.Conv2d(1, c1, 3, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, 3, padding=1)
        self.fc1 = nn.Linear(c2 * 7 * 7, hidden)
        self.fc2 = nn.Linear(hidden, n_classes)

    def forward(self, x):
        # TODO: conv-relu-pool, conv-relu-pool, flatten, fc1-relu, fc2
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Step 3 - train_classifier
import torch
import torch.nn.functional as F

def train_classifier(model, X, y, epochs=1, lr=1e-3, batch_size=64, seed=0):
    # TODO: Adam; seeded torch.randperm per epoch; cross-entropy; return mean epoch losses
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed)
    n = X.shape[0]
    epoch_losses = []

    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n, generator=generator)
        total_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb = X[idx], y[idx]

            optimizer.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        epoch_losses.append(total_loss / n_batches)

    return epoch_losses


def accuracy(model, X, y, batch_size=256):
    # TODO: eval + no_grad; fraction of argmax == y as a float
    model.eval()
    correct = 0
    n = X.shape[0]
    with torch.no_grad():
        for start in range(0, n, batch_size):
            xb = X[start:start + batch_size]
            yb = y[start:start + batch_size]
            preds = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
    return correct / n

# Step 4 - model_size_mb
import torch

def count_params(model):
    # TODO: (total, nonzero) as ints
    total = 0
    nonzero = 0
    for p in model.parameters():
        total += p.numel()
        nonzero += torch.count_nonzero(p).item()
    return total, nonzero


def model_size_mb(model, bits=32):
    # TODO: n_params * bits / 8 / 1e6, rounded to 4 decimals
    total, _ = count_params(model)
    size = total * bits / 8 / 1e6
    return round(size, 4)


def sparsity(model):
    # TODO: 1 - nonzero / total, rounded to 4 decimals
    total, nonzero = count_params(model)
    return round(1 - nonzero / total, 4)

# Step 5 - measure_latency_ms
import time
import torch

def measure_latency_ms(model, input_shape=(1, 1, 28, 28), iters=20, warmup=5):
    # TODO: eval + no_grad; warmup passes; time iters passes with perf_counter; return the median ms
    model.eval()
    x = torch.randn(input_shape)

    with torch.no_grad():
        for _ in range(warmup):
            model(x)

        times = []
        for _ in range(iters):
            start = time.perf_counter()
            model(x)
            end = time.perf_counter()
            times.append((end - start) * 1000)

    times.sort()
    n = len(times)
    mid = n // 2
    if n % 2 == 0:
        median = (times[mid - 1] + times[mid]) / 2
    else:
        median = times[mid]

    return median


def fps_from_latency(latency_ms):
    # TODO: 1000 / latency_ms rounded to 1 decimal
    return round(1000 / latency_ms, 1)

# Step 6 - global_magnitude_prune
import torch
import torch.nn as nn

def global_magnitude_prune(model, target_sparsity):
    # TODO: one threshold across all Conv2d/Linear weights via torch.kthvalue; zero in place; return {name: bool mask}
    weight_names = []
    weight_tensors = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            weight_names.append(f"{name}.weight")
            weight_tensors.append(module.weight)

    all_mags = torch.cat([w.detach().abs().flatten() for w in weight_tensors])
    total_weights = all_mags.numel()
    k = int(target_sparsity * total_weights)  # floor for non-negative values

    if k == 0:
        threshold = None
    else:
        threshold = torch.kthvalue(all_mags, k).values

    masks = {}
    with torch.no_grad():
        for name, w in zip(weight_names, weight_tensors):
            if threshold is None:
                mask = torch.ones_like(w, dtype=torch.bool)
            else:
                mask = w.abs() > threshold
                w.mul_(mask)
            masks[name] = mask

    return masks


def weight_sparsity(masks):
    # TODO: fraction of False entries over all masks, rounded to 4 decimals
    total = 0
    false_count = 0
    for mask in masks.values():
        total += mask.numel()
        false_count += (~mask).sum().item()
    return round(false_count / total, 4)

# Step 7 - fine_tune_pruned
import torch
import torch.nn.functional as F

def apply_masks(model, masks):
    # TODO: for each named weight in masks, multiply in place by the mask (no_grad)
    named = dict(model.named_parameters())
    with torch.no_grad():
        for name, mask in masks.items():
            named[name].mul_(mask)


def fine_tune_pruned(model, masks, X, y, epochs=1, lr=5e-4, batch_size=64, seed=0):
    # TODO: Adam training loop; apply_masks after every optimizer step; return mean epoch losses
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed)
    n = X.shape[0]
    epoch_losses = []

    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n, generator=generator)
        total_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb = X[idx], y[idx]

            optimizer.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            optimizer.step()
            apply_masks(model, masks)

            total_loss += loss.item()
            n_batches += 1

        epoch_losses.append(total_loss / n_batches)

    return epoch_losses

# Step 8 - iterative_prune
import torch

def prune_schedule(target, n_rounds):
    # TODO: [target * (1 - (1 - k/n)**3) for k in 1..n], rounded to 4 decimals
    return [round(target * (1 - (1 - k / n_rounds) ** 3), 4) for k in range(1, n_rounds + 1)]


def iterative_prune(model, X, y, target, n_rounds, epochs_per_round=1, lr=5e-4, seed=0):
    # TODO: for each scheduled sparsity: global_magnitude_prune, fine_tune_pruned; return (masks, history)
    schedule = prune_schedule(target, n_rounds)
    history = []
    masks = None

    for round_idx, s in enumerate(schedule):
        masks = global_magnitude_prune(model, s)
        losses = fine_tune_pruned(
            model, masks, X, y,
            epochs=epochs_per_round, lr=lr, seed=seed + round_idx
        )
        history.append((weight_sparsity(masks), losses[-1]))

    return masks, history

# Step 9 - sparse_storage_mb
import torch

def sparse_storage_mb(model, masks, value_bits=32, index_bits=16):
    # TODO: masked weights: nonzero * (value_bits + index_bits); other params: numel * value_bits; / 8e6, 4 decimals
    total_bits = 0
    for name, param in model.named_parameters():
        if name in masks:
            nonzero = masks[name].sum().item()
            total_bits += nonzero * (value_bits + index_bits)
        else:
            total_bits += param.numel() * value_bits
    return round(total_bits / 8e6, 4)


def pruning_breakeven_sparsity(value_bits=32, index_bits=16):
    # TODO: 1 - value_bits / (value_bits + index_bits), 4 decimals
    return round(1 - value_bits / (value_bits + index_bits), 4)

# Step 10 - quantize_symmetric
import torch

def quantize_symmetric(w, bits):
    # TODO: qmax = 2**(bits-1) - 1; scale = absmax / qmax (1.0 if zero); q = clamp(round(w/scale)) int32; return (q, float scale)
    qmax = 2 ** (bits - 1) - 1
    absmax = w.abs().max().item()
    scale = absmax / qmax if absmax != 0 else 1.0
    q = torch.clamp(torch.round(w / scale), -qmax, qmax).to(torch.int32)
    return q, float(scale)


def dequantize(q, scale):
    # TODO: q.float() * scale
    return q.float() * scale


def fake_quantize(w, bits):
    # TODO: dequantize(*quantize_symmetric(w, bits))
    return dequantize(*quantize_symmetric(w, bits))

# Step 11 - quantize_per_channel
import torch

def quantize_per_channel(w, bits):
    # TODO: per output channel (dim 0) absmax scales; return (int32 q, float scales of shape (out,))
    qmax = 2 ** (bits - 1) - 1
    n_out = w.shape[0]
    flat = w.reshape(n_out, -1)
    absmax = flat.abs().max(dim=1).values
    scales = torch.where(absmax != 0, absmax / qmax, torch.ones_like(absmax))

    scale_shape = (n_out,) + (1,) * (w.dim() - 1)
    scale_broadcast = scales.reshape(scale_shape)
    q = torch.clamp(torch.round(w / scale_broadcast), -qmax, qmax).to(torch.int32)

    return q, scales


def fake_quantize_per_channel(w, bits):
    # TODO: dequantized float tensor
    q, scales = quantize_per_channel(w, bits)
    scale_shape = (w.shape[0],) + (1,) * (w.dim() - 1)
    scale_broadcast = scales.reshape(scale_shape)
    return q.float() * scale_broadcast


def quantization_mse(w, w_hat):
    # TODO: mean squared error as a float
    return ((w - w_hat) ** 2).mean().item()

# Step 12 - quantize_model
import copy
import torch
import torch.nn as nn

def quantize_model(model, bits, per_channel=True):
    # TODO: deepcopy; fake-quantize each conv/linear weight in place (per channel or per tensor); return (copy, scales dict)
    model_copy = copy.deepcopy(model)
    scales = {}

    with torch.no_grad():
        for name, module in model_copy.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                w_name = f"{name}.weight"
                w = module.weight
                if per_channel:
                    q, s = quantize_per_channel(w, bits)
                    scale_shape = (w.shape[0],) + (1,) * (w.dim() - 1)
                    w_hat = q.float() * s.reshape(scale_shape)
                else:
                    q, s = quantize_symmetric(w, bits)
                    w_hat = q.float() * s
                module.weight.copy_(w_hat)
                scales[w_name] = s

    return model_copy, scales


def quantized_size_mb(model, bits):
    # TODO: weights at bits, other params at 32 bits; MB rounded to 4 decimals
    total_bits = 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            total_bits += module.weight.numel() * bits
            if module.bias is not None:
                total_bits += module.bias.numel() * 32
    return round(total_bits / 8e6, 4)

# Step 13 - quantization_snr_db
import math
import torch

def quantization_snr_db(w, bits, per_channel=False):
    # TODO: 10 log10(mean(w^2) / mean((w - w_hat)^2)), 2 decimals
    if per_channel:
        w_hat = fake_quantize_per_channel(w, bits)
    else:
        w_hat = fake_quantize(w, bits)
    signal = (w ** 2).mean().item()
    noise = ((w - w_hat) ** 2).mean().item()
    snr = 10 * math.log10(signal / noise)
    return round(snr, 2)


def snr_per_bit(w, bits_list):
    # TODO: {bits: snr_db}
    return {bits: quantization_snr_db(w, bits) for bits in bits_list}


def db_gain_per_bit(snr_table):
    # TODO: (snr at max bits - snr at min bits) / (max bits - min bits), 2 decimals
    bits_sorted = sorted(snr_table.keys())
    min_bits, max_bits = bits_sorted[0], bits_sorted[-1]
    gain = (snr_table[max_bits] - snr_table[min_bits]) / (max_bits - min_bits)
    return round(gain, 2)

# Step 14 - activation_ranges
import torch
import torch.nn as nn

def activation_ranges(model, X, batch_size=256):
    # TODO: forward hooks on Conv2d/Linear recording running absmax of outputs; run X in eval/no_grad; remove hooks; return {name: float}
    ranges = {}

    def make_hook(name):
        def hook(module, input, output):
            batch_max = output.abs().max().item()
            if name not in ranges or batch_max > ranges[name]:
                ranges[name] = batch_max
        return hook

    handles = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            handles.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    n = X.shape[0]
    with torch.no_grad():
        for start in range(0, n, batch_size):
            xb = X[start:start + batch_size]
            model(xb)

    for h in handles:
        h.remove()

    return ranges


def activation_scales(ranges, bits):
    # TODO: {name: absmax / (2**(bits-1) - 1)}
    qmax = 2 ** (bits - 1) - 1
    return {name: absmax / qmax for name, absmax in ranges.items()}

# Step 15 - distillation_loss
import torch
import torch.nn.functional as F

def soft_targets(logits, T):
    # TODO: softmax(logits / T, dim=1)
    return F.softmax(logits / T, dim=1)


def distillation_loss(student_logits, teacher_logits, y, T=4.0, alpha=0.7):
    # TODO: alpha * T^2 * KL(teacher_soft || student_soft) + (1 - alpha) * CE(student_logits, y)
    student_log_soft = F.log_softmax(student_logits / T, dim=1)
    teacher_log_soft = F.log_softmax(teacher_logits / T, dim=1)

    kl = F.kl_div(student_log_soft, teacher_log_soft, reduction='batchmean', log_target=True)
    ce = F.cross_entropy(student_logits, y)

    return alpha * T ** 2 * kl + (1 - alpha) * ce

# Step 16 - train_student
import torch

def train_student(student, teacher, X, y, epochs=1, lr=1e-3, batch_size=64, T=4.0, alpha=0.7, seed=0):
    # TODO: teacher eval + no_grad logits per batch; student Adam on distillation_loss; return mean epoch losses
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed)
    n = X.shape[0]
    epoch_losses = []

    teacher.eval()
    student.train()
    for epoch in range(epochs):
        perm = torch.randperm(n, generator=generator)
        total_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb = X[idx], y[idx]

            with torch.no_grad():
                teacher_logits = teacher(xb)

            optimizer.zero_grad()
            student_logits = student(xb)
            loss = distillation_loss(student_logits, teacher_logits, yb, T=T, alpha=alpha)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        epoch_losses.append(total_loss / n_batches)

    return epoch_losses

# Step 17 - compare_students
import torch

def compare_students(teacher, X_train, y_train, X_test, y_test, c1=8, c2=16, hidden=32, epochs=2, seed=0):
    # TODO: two identically seeded students; one distilled, one plain; report accuracy/params/size for all three and kd_gain
    torch.manual_seed(seed)
    student_kd = SmallCNN(c1=c1, c2=c2, hidden=hidden)

    torch.manual_seed(seed)
    student_plain = SmallCNN(c1=c1, c2=c2, hidden=hidden)

    train_student(student_kd, teacher, X_train, y_train, epochs=epochs, seed=seed)
    train_classifier(student_plain, X_train, y_train, epochs=epochs, seed=seed)

    def report(model):
        total, _ = count_params(model)
        return {
            'accuracy': round(accuracy(model, X_test, y_test), 4),
            'params': total,
            'size_mb': model_size_mb(model),
        }

    result = {
        'teacher': report(teacher),
        'student_kd': report(student_kd),
        'student_plain': report(student_plain),
    }
    result['kd_gain'] = round(
        result['student_kd']['accuracy'] - result['student_plain']['accuracy'], 4
    )

    return result

# Step 18 - compression_report
import torch

def compression_report(model, X_test, y_test, bits=32, masks=None, latency_iters=20):
    # TODO: params, sparsity, size_mb (quantized_size_mb), sparse_mb (sparse_storage_mb when masks), accuracy, latency_ms
    total, _ = count_params(model)

    size_mb = quantized_size_mb(model, bits)
    if masks is not None:
        sparse_mb = sparse_storage_mb(model, masks, value_bits=bits)
    else:
        sparse_mb = size_mb

    return {
        'params': total,
        'sparsity': sparsity(model),
        'size_mb': size_mb,
        'sparse_mb': sparse_mb,
        'accuracy': round(accuracy(model, X_test, y_test), 4),
        'latency_ms': round(measure_latency_ms(model, iters=latency_iters), 3),
    }


def meets_budget(report, budget):
    # TODO: size_ok (sparse_mb <= max_mb), latency_ok, accuracy_ok, ok
    size_ok = report['sparse_mb'] <= budget['max_mb']
    latency_ok = report['latency_ms'] <= budget['max_latency_ms']
    accuracy_ok = report['accuracy'] >= budget['min_accuracy']

    return {
        'size_ok': size_ok,
        'latency_ok': latency_ok,
        'accuracy_ok': accuracy_ok,
        'ok': size_ok and latency_ok and accuracy_ok,
    }

# Step 19 - compress_for_budget
import torch

def compress_for_budget(teacher, X_train, y_train, X_test, y_test, budget, student_widths=(8, 16, 32), prune_target=0.8, prune_rounds=2, bits=8, epochs=2, seed=0):
    # TODO: stages teacher -> distilled student -> iteratively pruned -> quantized; report + meets_budget after each; stop at first ok
    stages = []

    # Stage 1: teacher, fp32, no masks
    report = compression_report(teacher, X_test, y_test, bits=32, masks=None)
    checks = meets_budget(report, budget)
    stages.append(('teacher', report, checks))
    if checks['ok']:
        return {'stages': stages, 'final_stage': 'teacher', 'ok': True, 'model': teacher}

    # Stage 2: distilled student
    c1, c2, hidden = student_widths
    torch.manual_seed(seed)
    student = SmallCNN(c1=c1, c2=c2, hidden=hidden)
    train_student(student, teacher, X_train, y_train, epochs=epochs, seed=seed)

    report = compression_report(student, X_test, y_test, bits=32, masks=None)
    checks = meets_budget(report, budget)
    stages.append(('distilled', report, checks))
    if checks['ok']:
        return {'stages': stages, 'final_stage': 'distilled', 'ok': True, 'model': student}

    # Stage 3: pruned student
    masks, _ = iterative_prune(
        student, X_train, y_train, prune_target, prune_rounds,
        epochs_per_round=1, seed=seed
    )

    report = compression_report(student, X_test, y_test, bits=32, masks=masks)
    checks = meets_budget(report, budget)
    stages.append(('pruned', report, checks))
    if checks['ok']:
        return {'stages': stages, 'final_stage': 'pruned', 'ok': True, 'model': student}

    # Stage 4: quantized (pruned) student
    quantized_student, _ = quantize_model(student, bits)

    report = compression_report(quantized_student, X_test, y_test, bits=bits, masks=masks)
    checks = meets_budget(report, budget)
    stages.append(('quantized', report, checks))
    if checks['ok']:
        return {'stages': stages, 'final_stage': 'quantized', 'ok': True, 'model': quantized_student}

    return {'stages': stages, 'final_stage': 'quantized', 'ok': False, 'model': quantized_student}

