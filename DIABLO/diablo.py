"""
DIABLO: Death-Induced Apoptotic Biologically-Learned Optimization

A CNN that learns exclusively by pruning connections in a randomly initialized
network — no backpropagation, no gradient computation, no weight updates.

Biological analogy:
    DIABLO/Smac protein promotes neuronal apoptosis by inhibiting IAPs
    (Inhibitor of Apoptosis Proteins), releasing caspases for programmed cell death.

Algorithmic mapping:
    - Randomly initialized network  → dense neural tissue
    - IAP scores per connection     → protection/health of each synapse
    - Mitochondrial stress signal   → misclassification rate on forward passes
    - DIABLO release                → activity-based death scoring
    - Caspase cascade               → threshold-based pruning
    - Surviving subnetwork          → the learned model

Key insight: A sufficiently over-parameterized random network contains within it
a subnetwork that performs well (Ramanujan et al., 2020). DIABLO finds it
without ever computing a gradient.

Dependencies: numpy, (optional) torchvision for MNIST auto-download
              Falls back to manual gzip download if torchvision unavailable.
"""

import numpy as np
import struct
import gzip
import os
import time
import json
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional


# =============================================================================
# MNIST loader (multiple fallback strategies)
# =============================================================================

def load_mnist(path: str = "./mnist_data") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load MNIST. Tries torchvision first, falls back to manual download.
    Returns (train_images, train_labels, test_images, test_labels)
    as uint8 arrays: images (N, 28, 28), labels (N,).
    """
    # Strategy 1: torchvision (most reliable)
    try:
        from torchvision import datasets
        train = datasets.MNIST(path, train=True, download=True)
        test = datasets.MNIST(path, train=False, download=True)
        return (
            train.data.numpy(),
            train.targets.numpy(),
            test.data.numpy(),
            test.targets.numpy(),
        )
    except ImportError:
        pass

    # Strategy 2: manual download from multiple mirrors
    import urllib.request

    os.makedirs(path, exist_ok=True)

    mirrors = [
        "https://storage.googleapis.com/cvdf-datasets/mnist/",
        "http://yann.lecun.com/exdb/mnist/",
        "https://ossci-datasets.s3.amazonaws.com/mnist/",
    ]

    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images":  "t10k-images-idx3-ubyte.gz",
        "test_labels":  "t10k-labels-idx1-ubyte.gz",
    }

    def read_images(filepath):
        with gzip.open(filepath, 'rb') as f:
            _, n, rows, cols = struct.unpack('>IIII', f.read(16))
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows, cols)

    def read_labels(filepath):
        with gzip.open(filepath, 'rb') as f:
            struct.unpack('>II', f.read(8))
            return np.frombuffer(f.read(), dtype=np.uint8)

    for key, fname in files.items():
        fpath = os.path.join(path, fname)
        if not os.path.exists(fpath):
            downloaded = False
            for mirror in mirrors:
                try:
                    print(f"  Downloading {fname} from {mirror}...")
                    urllib.request.urlretrieve(mirror + fname, fpath)
                    downloaded = True
                    break
                except Exception as e:
                    print(f"    Failed ({e}), trying next mirror...")
            if not downloaded:
                raise RuntimeError(
                    f"Could not download {fname}. Please manually download MNIST "
                    f"and place .gz files in {path}/"
                )

    return (
        read_images(os.path.join(path, files["train_images"])),
        read_labels(os.path.join(path, files["train_labels"])),
        read_images(os.path.join(path, files["test_images"])),
        read_labels(os.path.join(path, files["test_labels"])),
    )


# =============================================================================
# Pure NumPy CNN layers (forward pass only — no backward ever)
# =============================================================================

def conv2d_forward(x: np.ndarray, W: np.ndarray, b: np.ndarray,
                   mask: np.ndarray, stride: int = 1) -> np.ndarray:
    """
    Masked convolution forward pass via im2col.
    x: (batch, C_in, H, W)
    W: (C_out, C_in, kH, kW)  — frozen random weights
    mask: same shape as W       — binary mask (the learnable structure)
    b: (C_out,)
    """
    W_eff = W * mask
    batch, C_in, H, W_in = x.shape
    C_out, _, kH, kW = W_eff.shape
    H_out = (H - kH) // stride + 1
    W_out = (W_in - kW) // stride + 1

    # im2col
    cols = np.zeros((batch, C_in, kH, kW, H_out, W_out), dtype=x.dtype)
    for i in range(kH):
        i_max = i + stride * H_out
        for j in range(kW):
            j_max = j + stride * W_out
            cols[:, :, i, j, :, :] = x[:, :, i:i_max:stride, j:j_max:stride]

    cols_reshaped = cols.transpose(0, 4, 5, 1, 2, 3).reshape(batch * H_out * W_out, -1)
    W_reshaped = W_eff.reshape(C_out, -1).T

    out = (cols_reshaped @ W_reshaped).reshape(batch, H_out, W_out, C_out)
    out = out.transpose(0, 3, 1, 2)
    out += b[np.newaxis, :, np.newaxis, np.newaxis]
    return out


def maxpool2d(x: np.ndarray, size: int = 2) -> np.ndarray:
    """2D max pooling."""
    B, C, H, W = x.shape
    H_out, W_out = H // size, W // size
    x_reshaped = x[:, :, :H_out * size, :W_out * size]
    x_reshaped = x_reshaped.reshape(B, C, H_out, size, W_out, size)
    return x_reshaped.max(axis=(3, 5))


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def dense_forward(x: np.ndarray, W: np.ndarray, b: np.ndarray,
                  mask: np.ndarray) -> np.ndarray:
    """Masked fully-connected forward pass."""
    return x @ (W * mask).T + b


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


# =============================================================================
# DIABLO Network
# =============================================================================

@dataclass
class DiabloLayer:
    """A single layer with frozen weights and a mutable binary mask."""
    name: str
    weights: np.ndarray          # Frozen random weights (never modified)
    bias: np.ndarray             # Frozen random bias
    mask: np.ndarray             # Binary mask (THIS is what learns)
    iap_scores: np.ndarray       # "Inhibitor of Apoptosis" protection scores
    layer_type: str              # 'conv' or 'dense'

    @property
    def alive_ratio(self) -> float:
        return self.mask.sum() / self.mask.size

    @property
    def total_params(self) -> int:
        return self.mask.size

    @property
    def alive_params(self) -> int:
        return int(self.mask.sum())


class DiabloCNN:
    """
    A CNN that learns by pruning, not by gradient descent.

    CRITICAL DESIGN CHOICE: The network is intentionally over-parameterized
    (~2M params). The lottery ticket hypothesis requires that a good subnetwork
    EXISTS within the random initialization. Wider layers = more lottery tickets.

    Architecture (MNIST, 28x28 input):
        Conv(1 → 128, 5x5)  → ReLU → MaxPool(2)    → (128, 12, 12)
        Conv(128 → 256, 3x3) → ReLU → MaxPool(2)    → (256, 5, 5)
        Dense(256*5*5=6400 → 512) → ReLU
        Dense(512 → 10)

    All weights are randomly initialized once and NEVER changed.
    Learning happens exclusively through the binary masks.
    """

    def __init__(self, seed: int = 42, init_scale: float = 1.0):
        self.rng = np.random.RandomState(seed)
        self.layers: List[DiabloLayer] = []
        self.init_scale = init_scale
        self._build()

    def _kaiming_init(self, shape: tuple, fan_in: int) -> np.ndarray:
        """Kaiming He initialization (important for signal propagation without training)."""
        std = self.init_scale * np.sqrt(2.0 / fan_in)
        return self.rng.randn(*shape).astype(np.float32) * std

    def _make_layer(self, name: str, shape: tuple, fan_in: int, ltype: str) -> DiabloLayer:
        return DiabloLayer(
            name=name,
            weights=self._kaiming_init(shape, fan_in),
            bias=np.zeros(shape[0], dtype=np.float32),
            mask=np.ones(shape, dtype=np.float32),
            iap_scores=np.ones(shape, dtype=np.float32),
            layer_type=ltype,
        )
    def _build_small(self):
        """~3.6M params — pure MLP, width-scaled."""
        # Dense1: 784 → 1541
        self.layers.append(self._make_layer("dense1", (1541, 784), 784, "dense"))
        # Dense2: 1541 → 1541
        self.layers.append(self._make_layer("dense2", (1541, 1541), 1541, "dense"))
        # Dense3: 1541 → 10
        self.layers.append(self._make_layer("dense3", (10, 1541), 1541, "dense"))

    def _build_medium(self):
        """~1B params — pure MLP, width-scaled."""
        # Dense1: 784 → 31228
        self.layers.append(self._make_layer("dense1", (31228, 784), 784, "dense"))
        # Dense2: 31228 → 31228
        self.layers.append(self._make_layer("dense2", (31228, 31228), 31228, "dense"))
        # Dense3: 31228 → 10
        self.layers.append(self._make_layer("dense3", (10, 31228), 31228, "dense"))

    def _build_large(self):
        """~100B params — pure MLP, width-scaled."""
        # Dense1: 784 → 315830
        self.layers.append(self._make_layer("dense1", (315830, 784), 784, "dense"))
        # Dense2: 315830 → 315830
        self.layers.append(self._make_layer("dense2", (315830, 315830), 315830, "dense"))
        # Dense3: 315830 → 10
        self.layers.append(self._make_layer("dense3", (10, 315830), 315830, "dense"))

    def _build_small_conv(self):
            """~3.6M params — width-scaled variant of shared topology."""
            # Conv1: 1 → 32, kernel 7x7, input 28x28 → 22x22 → pool → 11x11
            self.layers.append(self._make_layer("conv1", (32, 1, 7, 7), 1*7*7, "conv"))
            # Conv2: 32 → 64, kernel 5x5, 11x11 → 7x7 → pool → 3x3
            self.layers.append(self._make_layer("conv2", (64, 32, 5, 5), 32*5*5, "conv"))
            # Conv3: 64 → 128, kernel 3x3, 3x3 → 1x1 (no pool)
            self.layers.append(self._make_layer("conv3", (128, 64, 3, 3), 64*3*3, "conv"))
            # Dense1: 128*1*1=128 → 1795
            self.layers.append(self._make_layer("dense1", (1795, 128), 128, "dense"))
            # Dense2: 1795 → 1795
            self.layers.append(self._make_layer("dense2", (1795, 1795), 1795, "dense"))
            # Dense3: 1795 → 10
            self.layers.append(self._make_layer("dense3", (10, 1795), 1795, "dense"))

    def _build_medium_conv(self):
        """~1B params — width-scaled variant of shared topology."""
        # Conv1: 1 → 512, kernel 7x7, input 28x28 → 22x22 → pool → 11x11
        self.layers.append(self._make_layer("conv1", (512, 1, 7, 7), 1*7*7, "conv"))
        # Conv2: 512 → 1024, kernel 5x5, 11x11 → 7x7 → pool → 3x3
        self.layers.append(self._make_layer("conv2", (1024, 512, 5, 5), 512*5*5, "conv"))
        # Conv3: 1024 → 2048, kernel 3x3, 3x3 → 1x1 (no pool)
        self.layers.append(self._make_layer("conv3", (2048, 1024, 3, 3), 1024*3*3, "conv"))
        # Dense1: 2048*1*1=2048 → 30100
        self.layers.append(self._make_layer("dense1", (30100, 2048), 2048, "dense"))
        # Dense2: 30100 → 30100
        self.layers.append(self._make_layer("dense2", (30100, 30100), 30100, "dense"))
        # Dense3: 30100 → 10
        self.layers.append(self._make_layer("dense3", (10, 30100), 30100, "dense"))

    def _build_large_conv(self):
        """~100B params — width-scaled variant of shared topology."""
        # Conv1: 1 → 512, kernel 7x7, input 28x28 → 22x22 → pool → 11x11
        self.layers.append(self._make_layer("conv1", (512, 1, 7, 7), 1*7*7, "conv"))
        # Conv2: 512 → 1024, kernel 5x5, 11x11 → 7x7 → pool → 3x3
        self.layers.append(self._make_layer("conv2", (1024, 512, 5, 5), 512*5*5, "conv"))
        # Conv3: 1024 → 2048, kernel 3x3, 3x3 → 1x1 (no pool)
        self.layers.append(self._make_layer("conv3", (2048, 1024, 3, 3), 1024*3*3, "conv"))
        # Dense1: 2048*1*1=2048 → 315149
        self.layers.append(self._make_layer("dense1", (315149, 2048), 2048, "dense"))
        # Dense2: 315149 → 315149
        self.layers.append(self._make_layer("dense2", (315149, 315149), 315149, "dense"))
        # Dense3: 315149 → 10
        self.layers.append(self._make_layer("dense3", (10, 315149), 315149, "dense"))

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
            """
            Forward pass through masked network.
            x: (batch, 1, 28, 28) normalized images
            Returns: (logits, activations_dict)
            """
            activations = {}
            h = x

            # Detect if we have conv layers
            has_conv = any(layer.layer_type == "conv" for layer in self.layers)

            if not has_conv:
                # Pure MLP: flatten input
                batch = h.shape[0]
                h = h.reshape(batch, -1)
                activations['flat'] = h

            for layer in self.layers:
                if layer.layer_type == "conv":
                    h = conv2d_forward(h, layer.weights, layer.bias, layer.mask)
                    activations[f'{layer.name}_pre'] = h
                    h = relu(h)
                    activations[f'{layer.name}_post'] = h
                    h = maxpool2d(h, 2)
                    activations[f'{layer.name}_pool'] = h
                elif layer.layer_type == "dense":
                    # Flatten on first dense layer after conv
                    if has_conv and h.ndim == 4:
                        batch = h.shape[0]
                        h = h.reshape(batch, -1)
                        activations['flat'] = h

                    is_last = (layer is self.layers[-1])
                    h = dense_forward(h, layer.weights, layer.bias, layer.mask)

                    if is_last:
                        activations['logits'] = h
                    else:
                        activations[f'{layer.name}_pre'] = h
                        h = relu(h)
                        activations[f'{layer.name}_post'] = h

            return h, activations

    def predict(self, x: np.ndarray) -> np.ndarray:
        logits, _ = self.forward(x)
        return logits.argmax(axis=1)

    def evaluate(self, images: np.ndarray, labels: np.ndarray,
                 batch_size: int = 256) -> float:
        """Compute accuracy over a dataset."""
        correct = 0
        n = len(images)
        for i in range(0, n, batch_size):
            x = images[i:i + batch_size]
            y = labels[i:i + batch_size]
            preds = self.predict(x)
            correct += (preds == y).sum()
        return correct / n

    def sparsity_report(self) -> dict:
        """Report the current state of the network structure."""
        total_all = 0
        alive_all = 0
        report = {}
        for layer in self.layers:
            total_all += layer.total_params
            alive_all += layer.alive_params
            report[layer.name] = {
                "alive": layer.alive_params,
                "total": layer.total_params,
                "ratio": layer.alive_ratio,
            }
        report["global"] = {
            "alive": alive_all,
            "total": total_all,
            "ratio": alive_all / total_all if total_all > 0 else 0,
        }
        return report


# =============================================================================
# DIABLO Pruning Engine
# =============================================================================

class DiabloEngine:
    """
    The DIABLO pruning algorithm.

    Biological process modeled:
    1. STRESS DETECTION: Forward pass batches, measure per-connection
       contribution to correct vs incorrect classifications.
    2. IAP EROSION: Connections that don't contribute to correct output
       have their IAP (protection) scores decreased.
    3. DIABLO RELEASE: When global stress (error rate) is high,
       DIABLO signal strength increases, accelerating pruning.
    4. CASPASE CASCADE: Connections whose IAP falls below threshold
       undergo apoptosis (mask → 0).
    5. HOMEOSTASIS: After pruning, surviving connections have IAP
       partially restored (the network stabilizes).
    """

    def __init__(self,
                 model: DiabloCNN,
                 iap_decay: float = 0.15,
                 iap_recovery: float = 0.05,
                 diablo_base: float = 0.08,
                 diablo_stress_mult: float = 2.0,
                 min_survival: float = 0.03,
                 num_probe_batches: int = 20,
                 batch_size: int = 128,
                 seed: int = 123):

        self.model = model
        self.iap_decay = iap_decay
        self.iap_recovery = iap_recovery
        self.diablo_base = diablo_base
        self.diablo_stress_mult = diablo_stress_mult
        self.min_survival = min_survival
        self.num_probe_batches = num_probe_batches
        self.batch_size = batch_size
        self.rng = np.random.RandomState(seed)
        self.history: List[dict] = []

    def _compute_connection_health(self,
                                   images: np.ndarray,
                                   labels: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Probe the network to assess which connections contribute to correct
        classifications. This is the "mitochondrial stress assay."

        Health scoring is PER-WEIGHT, not just per-filter:
        - For conv layers: health[c_out, c_in, kh, kw] reflects how much
          that specific weight element participates in correct predictions.
          Computed as (input_patch_activation * weight_magnitude * output_correctness).
        - For dense layers: health[out, in] = |input_activation[in]| * |weight| *
          (correct_signal[out] - incorrect_signal[out]).

        This is class-conditional Hebbian: "fire together FOR THE RIGHT
        REASON, wire together."
        """
        # Accumulate per-weight health scores
        health_accum = {layer.name: np.zeros_like(layer.weights) for layer in self.model.layers}
        health_count = 0

        n = len(images)
        indices = self.rng.permutation(n)

        for batch_idx in range(self.num_probe_batches):
            start = (batch_idx * self.batch_size) % n
            idx = indices[start:start + self.batch_size]
            if len(idx) == 0:
                continue

            x = images[idx]
            y = labels[idx]

            logits, activations = self.model.forward(x)
            preds = logits.argmax(axis=1)
            correct_mask = (preds == y).astype(np.float32)          # (B,)
            incorrect_mask = 1.0 - correct_mask
            B = len(x)

            # ─── Conv layers: per-weight health ───
            for layer in self.model.layers:
                if layer.layer_type == 'conv':
                    post_key = f"{layer.name}_post"
                    if post_key not in activations:
                        continue

                    act = activations[post_key]                      # (B, C_out, H, W)
                    # Per-filter output magnitude, averaged spatially
                    filter_act = np.abs(act).mean(axis=(2, 3))       # (B, C_out)

                    # Correct-weighted vs incorrect-weighted per filter
                    correct_signal = (filter_act * correct_mask[:, None]).mean(axis=0)   # (C_out,)
                    incorrect_signal = (filter_act * incorrect_mask[:, None]).mean(axis=0)

                    # Per-filter health: how much it contributes to correct classification
                    filter_health = correct_signal - 0.5 * incorrect_signal   # (C_out,)

                    # Weight magnitude modulation: within each filter, weights with
                    # larger absolute value that are in healthy filters get higher scores.
                    # This gives per-weight granularity.
                    W_abs = np.abs(layer.weights)                    # (C_out, C_in, kH, kW)
                    W_norm = W_abs / (W_abs.max(axis=(1, 2, 3), keepdims=True) + 1e-8)

                    # Broadcast filter_health across weight dimensions
                    per_weight_health = filter_health[:, None, None, None] * W_norm
                    health_accum[layer.name] += per_weight_health

                elif layer.layer_type == 'dense':
                    post_key = f"{layer.name}_post"
                    pre_key = f"{layer.name}_pre"
                    # Use post-ReLU if available, else pre-activation
                    ref_key = post_key if post_key in activations else pre_key
                    if ref_key not in activations:
                        continue

                    out_act = np.abs(activations[ref_key])           # (B, out_features)

                    # For dense layers, we also want to know which inputs were active.
                    # Get the layer's input activation.
                    if layer.name == "dense1":
                        in_act = np.abs(activations.get('flat', np.ones((B, layer.weights.shape[1]))))
                    elif layer.name == "dense2":
                        in_act = np.abs(activations.get('dense1_post', np.ones((B, layer.weights.shape[1]))))
                    else:
                        in_act = np.ones((B, layer.weights.shape[1]), dtype=np.float32)

                    # Per-output-neuron correctness signal
                    correct_signal = (out_act * correct_mask[:, None]).mean(axis=0)     # (out,)
                    incorrect_signal = (out_act * incorrect_mask[:, None]).mean(axis=0)
                    neuron_health = correct_signal - 0.5 * incorrect_signal              # (out,)

                    # Per-input mean activation (measures input relevance)
                    in_relevance = in_act.mean(axis=0)                                   # (in,)
                    in_relevance = in_relevance / (in_relevance.max() + 1e-8)

                    # Per-weight health: outer product of neuron_health × input_relevance,
                    # modulated by weight magnitude
                    W_abs = np.abs(layer.weights)
                    W_norm = W_abs / (W_abs.max() + 1e-8)

                    per_weight_health = (
                        neuron_health[:, None] *
                        in_relevance[None, :] *
                        W_norm
                    )
                    health_accum[layer.name] += per_weight_health

            health_count += 1

        # Normalize
        health_scores = {}
        for name in health_accum:
            health_scores[name] = health_accum[name] / max(health_count, 1)

        return health_scores

    def _compute_stress(self, images: np.ndarray, labels: np.ndarray) -> float:
        """Global mitochondrial stress = error rate on a probe sample."""
        idx = self.rng.choice(len(images), min(2000, len(images)), replace=False)
        acc = self.model.evaluate(images[idx], labels[idx])
        return 1.0 - acc

    def pruning_round(self,
                      images: np.ndarray,
                      labels: np.ndarray,
                      round_num: int) -> dict:
        """
        Execute one full DIABLO cycle:
        stress assessment → health scoring → IAP update → caspase pruning → homeostasis
        """
        t0 = time.time()

        # === 1. STRESS DETECTION ===
        stress = self._compute_stress(images, labels)

        # === 2. CONNECTION HEALTH ASSAY ===
        health_scores = self._compute_connection_health(images, labels)

        # === 3. IAP SCORE UPDATE ===
        for layer in self.model.layers:
            h = health_scores[layer.name]
            # Normalize health to [0, 1] range within layer (only among alive connections)
            alive = layer.mask > 0
            if alive.sum() == 0:
                continue
            h_alive = h[alive]
            h_min, h_max = h_alive.min(), h_alive.max()
            if h_max > h_min:
                h_norm = (h - h_min) / (h_max - h_min)
            else:
                h_norm = np.ones_like(h) * 0.5

            # IAP dynamics: healthy connections recover, weak ones erode
            iap_delta = np.where(
                h_norm > 0.5,
                self.iap_recovery * h_norm,
                -self.iap_decay * (1.0 - h_norm)
            )
            layer.iap_scores = np.clip(layer.iap_scores + iap_delta, 0.0, 1.0)
            layer.iap_scores *= layer.mask  # dead connections stay dead

        # === 4. DIABLO RELEASE & CASPASE CASCADE ===
        diablo_strength = self.diablo_base * (1.0 + self.diablo_stress_mult * stress)

        pruned_counts = {}
        for layer in self.model.layers:
            alive_before = layer.alive_params
            alive_mask = layer.mask > 0
            n_alive = alive_mask.sum()

            if n_alive == 0:
                pruned_counts[layer.name] = 0
                continue

            alive_iap = layer.iap_scores[alive_mask]

            # Minimum survival floor per layer
            min_alive = max(int(self.min_survival * layer.total_params), 1)
            max_prune = max(0, int(n_alive) - min_alive)
            n_to_prune = min(int(diablo_strength * n_alive), max_prune)

            if n_to_prune > 0:
                # Prune the weakest IAP scores
                threshold = np.partition(alive_iap, n_to_prune)[n_to_prune]
                death_mask = (layer.iap_scores < threshold) & alive_mask

                # Stochastic survival: ~10% of condemned connections get a reprieve
                # (biological noise in apoptosis signaling)
                survival_noise = self.rng.random(death_mask.shape) < 0.1
                death_mask = death_mask & ~survival_noise

                layer.mask[death_mask] = 0.0
                layer.iap_scores[death_mask] = 0.0

            pruned_counts[layer.name] = alive_before - layer.alive_params

        # === 5. HOMEOSTASIS ===
        for layer in self.model.layers:
            alive = layer.mask > 0
            layer.iap_scores[alive] = np.clip(
                layer.iap_scores[alive] + 0.02, 0.0, 1.0
            )

        # === RECORD ===
        elapsed = time.time() - t0
        report = self.model.sparsity_report()

        result = {
            "round": round_num,
            "stress": stress,
            "diablo_strength": diablo_strength,
            "pruned": pruned_counts,
            "sparsity": report,
            "elapsed": elapsed,
        }
        self.history.append(result)
        return result


# =============================================================================
# Training loop
# =============================================================================

def run_diablo(num_rounds: int = 30, seed: int = 42):
    """Full DIABLO training run on MNIST."""

    print("=" * 72)
    print("  DIABLO: Death-Induced Apoptotic Biologically-Learned Optimization")
    print("  Learning by pruning — zero backpropagation, zero weight updates")
    print("=" * 72)

    # ── Load data ──
    print("\n[1/3] Loading MNIST...")
    train_x, train_y, test_x, test_y = load_mnist()

    train_x = train_x.astype(np.float32)[:, np.newaxis, :, :] / 255.0
    test_x  = test_x.astype(np.float32)[:, np.newaxis, :, :] / 255.0
    print(f"       Train: {len(train_x):,}  Test: {len(test_x):,}")

    # ── Build model ──
    print("[2/3] Initializing over-parameterized random network...")
    print("       (weights frozen at init — will NEVER be updated)")
    model = DiabloCNN(seed=seed, init_scale=1.5)

    report = model.sparsity_report()
    total_params = report['global']['total']
    print(f"       Architecture: Conv(1→128,5x5) → Conv(128→256,3x3) → Dense(6400→512) → Dense(512→10)")
    print(f"       Total connections: {total_params:,}")

    # ── Baseline ──
    print("\n[3/3] Evaluating unpruned random network...")
    baseline_acc = model.evaluate(test_x[:5000], test_y[:5000], batch_size=256)
    print(f"       Baseline accuracy (random): {baseline_acc:.2%}")
    print(f"       (Expected ~10% for random 10-class classifier)\n")

    # ── DIABLO engine ──
    engine = DiabloEngine(
        model=model,
        iap_decay=0.15,
        iap_recovery=0.05,
        diablo_base=0.06,
        diablo_stress_mult=2.0,
        min_survival=0.03,
        num_probe_batches=25,
        batch_size=200,
        seed=seed + 1,
    )

    # ── Pruning loop ──
    print("=" * 72)
    print("  Beginning DIABLO pruning cycles")
    print("  mitochondrial stress → DIABLO release → caspase cascade → apoptosis")
    print("=" * 72)
    print(f"\n{'Rnd':>4} │ {'Test Acc':>8} │ {'Alive':>10} │ {'Pruned%':>8} │ {'Stress':>7} │ {'DIABLO':>7} │ {'Time':>6}")
    print("─────┼──────────┼────────────┼──────────┼─────────┼─────────┼───────")

    best_acc = baseline_acc
    best_round = -1
    accuracies = [baseline_acc]
    sparsities = [1.0]

    for r in range(num_rounds):
        result = engine.pruning_round(train_x, train_y, r)

        # Full eval every 5 rounds, subsample otherwise for speed
        if r % 5 == 0 or r == num_rounds - 1:
            acc = model.evaluate(test_x, test_y, batch_size=256)
        else:
            acc = model.evaluate(test_x[:3000], test_y[:3000], batch_size=256)

        alive_ratio = result['sparsity']['global']['ratio']
        alive_count = result['sparsity']['global']['alive']

        if acc > best_acc:
            best_acc = acc
            best_round = r

        accuracies.append(acc)
        sparsities.append(alive_ratio)

        print(f"  {r:2d} │ {acc:7.2%} │ {alive_count:>10,} │ {(1-alive_ratio)*100:7.2f}% │ "
              f"{result['stress']:6.3f} │ {result['diablo_strength']:6.4f} │ {result['elapsed']:5.1f}s")

    # ── Final report ──
    print("\n" + "=" * 72)
    print("  DIABLO Pruning Complete — Final Report")
    print("=" * 72)

    final_acc = model.evaluate(test_x, test_y, batch_size=256)
    final_report = model.sparsity_report()

    print(f"\n  Baseline (random network):   {baseline_acc:.2%}")
    print(f"  Best accuracy:               {best_acc:.2%}  (round {best_round})")
    print(f"  Final accuracy:              {final_acc:.2%}")
    print(f"\n  Parameter survival by layer:")
    for layer in model.layers:
        lr = final_report[layer.name]
        bar_len = int(lr['ratio'] * 40)
        bar = "█" * bar_len + "░" * (40 - bar_len)
        print(f"    {layer.name:8s}: {bar} {lr['ratio']:6.2%}  ({lr['alive']:,}/{lr['total']:,})")

    global_r = final_report['global']
    print(f"\n    {'TOTAL':8s}: {global_r['ratio']:6.2%} alive ({global_r['alive']:,}/{global_r['total']:,})")
    if global_r['ratio'] > 0:
        print(f"    Compression: {1/global_r['ratio']:.1f}x")

    print(f"\n  Accuracy: {baseline_acc:.2%} → {best_acc:.2%}")
    print(f"  ZERO gradients computed. ZERO weight updates.")
    print(f"  The network structure IS the learned representation.\n")

    # ── Save results ──
    results = {
        "baseline_acc": float(baseline_acc),
        "best_acc": float(best_acc),
        "final_acc": float(final_acc),
        "best_round": int(best_round),
        "accuracies": [float(a) for a in accuracies],
        "sparsities": [float(s) for s in sparsities],
        "layer_report": {l.name: {k: (int(v) if isinstance(v, (int, np.integer)) else float(v))
                                   for k, v in final_report[l.name].items()}
                         for l in model.layers},
        "total_params": int(total_params),
    }

    results_path = "diablo_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {results_path}")

    return results, model


if __name__ == "__main__":
    results, model = run_diablo(num_rounds=30, seed=42)
