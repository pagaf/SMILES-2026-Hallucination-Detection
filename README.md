# 🔍 SMILES-2026 Hallucination Detection

Detect whether a small language model's answer is *hallucinated* (fabricated) or *truthful* using the model's internal hidden states.

The model is **[Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B)** — a 24‑layer causal LM with hidden size 896.

**Primary competition metric:** Accuracy on the held‑out `test.csv`. Internally, I also track AUROC as a more informative metric for the probe quality.

---

## 1. Reproducibility Instructions

### 1.1. Environment

Tested with:

- Python 3.10+
- PyTorch 2.x with CUDA (T4 / similar)
- `transformers`
- `numpy`, `pandas`, `scikit-learn`, `tqdm`

Install dependencies:

```bash
git clone https://github.com/pagaf/SMILES-2026-Hallucination-Detection.git
cd SMILES-2026-Hallucination-Detection

python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate.bat     # Windows

pip install -r requirements.txt
```

### 1.2. Run the full pipeline

This single command:

- loads `Qwen/Qwen2.5-0.5B`,
- extracts hidden‑state features for `dataset.csv`,
- trains and evaluates the probe,
- extracts features for `test.csv`,
- writes `results.json` and `predictions.csv`.

```bash
python solution.py
```

Artifacts:

- `results.json` — evaluation summary (baseline vs probe).
- `predictions.csv` — predictions for `data/test.csv` (0 = truthful, 1 = hallucinated).

The repository is self‑contained: I only modified:

1. `aggregation.py` — feature extraction from hidden states.
2. `probe.py` — the probe classifier.

`solution.py`, `model.py`, `evaluate.py`, `splitting.py` remain unchanged in terms of logic and interface.

---

## 2. Final Approach

### 2.1. High-level idea

The goal is to use the internal states of Qwen2.5‑0.5B as a rich representation of the (prompt, response) pair, and train a lightweight supervised probe on top. Recent work shows that:

- Hidden states alone can be highly predictive of hallucinations, even without external retrieval [LLMs’ Internal States Retain the Power of Hallucination Detection, Chen et al., ICLR 2024][web:101][web:127].
- Geometric properties of these states (representation drift between layers, spectral statistics / “EigenScore”) correlate strongly with hallucination risk [web:101][web:147].

My final solution therefore combines:

1. **A strong baseline embedding**: last real token of the final layer (dim = 896).
2. **A compact 6‑dimensional geometric feature vector** encoding:
   - L2 norms of the last token in the last and penultimate layers,
   - late‑layer representation drift (cosine similarities),
   - spectral entropy and max eigenvalue of the per‑sequence covariance (EigenScore‑style).

These features are concatenated into a 902‑dimensional vector and fed into a small but expressive MLP probe.

### 2.2. `aggregation.py` — layer selection and geometric features

**Base aggregation**

I keep the base aggregation intentionally simple and strong:

```python
def aggregate(hidden_states, attention_mask):
    last_layer = hidden_states[-1]              # (seq_len, hidden_dim)
    real_positions = attention_mask.nonzero(as_tuple=False)
    last_pos = int(real_positions[-1].item())
    return last_layer[last_pos]                # (hidden_dim,)
```

This is the standard “CLS/last token” probing setup widely used in probing literature and in hallucination‑detection work such as INSIDE / EigenScore [web:101][web:211]. It already performed well in my experiments, giving test AUROC around 72–73% with a good probe.

**Compact geometric features (6 dims)**

On top of this 896‑dimensional base vector, I append a **6‑dimensional geometric descriptor** inspired by:

- Representation drift / internal confidence [web:101][web:203].
- Spectral statistics of hidden states / EigenScore [web:101][web:147].

The implemented features:

1. **L2 Norm of the last token on the final layer**  
   Captures the activation magnitude at the point where the model finishes its answer.

2. **L2 Norm of the last token on the penultimate layer**  
   Provides a reference magnitude slightly earlier in the computation.

3. **Cosine similarity between last token at layers −1 and −2**  
   Measures how much the representation changes in the final step (“representation drift”). Large drift indicates instability or uncertainty about the answer.

4. **Cosine similarity between last token at layers −2 and −4**  
   Captures a slightly longer‑range drift in late layers.

5. **Spectral entropy of the per‑sequence Gram matrix (EigenScore proxy)**  
   I take all non‑padding token vectors from the last layer, center them, compute the Gram matrix, obtain its eigenvalues, and compute
   \[
   H = -\sum_i p_i \log p_i,\quad p_i = \lambda_i / \sum_j \lambda_j
   \]
   Higher entropy corresponds to more diffuse / uncertain representations, in line with EigenScore’s intuition [web:101][web:147].

6. **Maximum eigenvalue of the Gram matrix**  
   Captures the dominant variance direction; high values reflect highly concentrated variance, which may correspond to more “confident” states.

Code (simplified, matching the final implementation):

```python
def extract_geometric_features(hidden_states, attention_mask):
    n_layers, seq_len, hidden_dim = hidden_states.shape
    device = hidden_states.device

    real_positions = attention_mask.nonzero(as_tuple=False)
    last_pos = int(real_positions[-1].item())

    # 1–2: L2 norms
    l2_last = torch.norm(hidden_states[-1, last_pos], p=2).unsqueeze(0)
    l2_penultimate = torch.norm(hidden_states[-2, last_pos], p=2).unsqueeze(0)

    # 3–4: representation drift
    cos = torch.nn.CosineSimilarity(dim=0)
    drift_1 = cos(hidden_states[-1, last_pos],
                  hidden_states[-2, last_pos]).unsqueeze(0)
    if n_layers >= 4:
        drift_2 = cos(hidden_states[-2, last_pos],
                      hidden_states[-4, last_pos]).unsqueeze(0)
    else:
        drift_2 = torch.tensor([1.0], device=device)

    # 5–6: spectral features (EigenScore-style)
    mask_bool = attention_mask.bool()
    real_tokens = hidden_states[-1][mask_bool]
    n_real = real_tokens.shape

    if n_real > 1:
        centered = real_tokens - real_tokens.mean(dim=0, keepdim=True)
        gram = torch.mm(centered, centered.T) / (n_real - 1)
        eigvals = torch.linalg.eigvalsh(gram).float()
        eigvals = torch.clamp(eigvals, min=1e-8)

        max_eigval = eigvals.max().unsqueeze(0)

        eig_sum = eigvals.sum()
        probs = eigvals / eig_sum
        spectral_entropy = -(probs * torch.log(probs)).sum().unsqueeze(0)
    else:
        spectral_entropy = torch.tensor([0.0], device=device)
        max_eigval = torch.tensor([0.0], device=device)

    geometric_feats = torch.cat([
        l2_last,
        l2_penultimate,
        drift_1,
        drift_2,
        spectral_entropy,
        max_eigval,
    ], dim=0)

    return torch.log1p(geometric_feats)
```

The final feature vector is:

- **Dimensionality:** 896 (base) + 6 (geometric) = **902**.

The choice of *only 6* geometric dimensions is deliberate: in earlier experiments, adding dozens of geometric features (layer‑wise norms, full drift curves, multiple spectral statistics) inflated the feature dimensionality to 2600+ and caused severe overfitting on only 689 samples (train AUROC ≈ 1.0, test AUROC ≈ 0.52–0.55). By contrast, this compact set captures the key geometric signals identified in the literature [web:101][web:147][web:203] while remaining statistically stable.

### 2.3. `probe.py` — the hallucination classifier

The probe is a small MLP classifier trained on the 902‑dimensional feature vectors.

**Architecture**

- Input: 902‑dimensional vector (hidden states + geometry).
- Hidden layers: `902 → 512 → 128 → 1`.
- Activation: `ReLU`.
- Normalization: `BatchNorm1d` after each hidden layer.
- Regularization: `Dropout(p=0.3)` after each hidden layer.

Rationale:

- A 2‑layer MLP is expressive enough to model non‑linear relations between hidden‑state features and hallucination labels, as shown in INSIDE and related probe studies [web:101][web:217].
- `BatchNorm` + `Dropout` counteract overfitting, which is critical given the small dataset size.

**Training details**

- Preprocessing: `StandardScaler` over all features.
- Loss: `BCEWithLogitsLoss` with `pos_weight = n_neg / n_pos` to balance the skewed label distribution (≈70% hallucinated).
- Optimizer: `AdamW(lr=5e-4, weight_decay=1e-3)`.
- Mini‑batch training: batch size = 64.
- Early stopping:
  - Up to 100 epochs.
  - Stop if training loss does not improve for 12 epochs.

Code sketch (simplified, matching the final implementation):

```python
class HallucinationProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self._net = None
        self._scaler = StandardScaler()
        self._threshold = 0.5

    def _build_network(self, input_dim):
        self._net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        if self._net is None:
            raise RuntimeError("Call fit() before forward().")
        return self._net(x).squeeze(-1)

    def fit(self, X, y):
        X_scaled = self._scaler.fit_transform(X)
        if self._net is None:
            self._build_network(X_scaled.shape)[1]

        X_t = torch.from_numpy(X_scaled).float()
        y_t = torch.from_numpy(y.astype(np.float32))

        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=5e-4,
            weight_decay=1e-3,
        )

        self.train()
        max_epochs = 100
        batch_size = 64
        best_loss = float("inf")
        patience = 12
        patience_counter = 0
        n_samples = X_t.size(0)

        for _ in range(max_epochs):
            perm = torch.randperm(n_samples)
            epoch_loss = 0.0

            for start in range(0, n_samples, batch_size):
                idx = perm[start:start + batch_size]
                optimizer.zero_grad()
                logits = self(X_t[idx])
                loss = criterion(logits, y_t[idx])
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach()) * len(idx)

            epoch_loss /= n_samples
            if epoch_loss + 1e-4 < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        self.eval()
        return self
```

**Threshold tuning (`fit_hyperparameters`)**

On the validation split, I tune the decision threshold to maximize F1:

- Compute `probs = sigmoid(logits)`.
- Try thresholds from:
  - all unique predicted probabilities, and
  - a coarse grid `linspace(0, 1, 201)`.
- Pick the threshold with the best F1 on validation.
- Store it in `self._threshold` and use it in `predict`.

---

## 3. Experiments and Failed Attempts

### 3.1. Multi-layer concatenation (large feature vectors)

I first tried concatenating mean‑pooled representations from multiple upper layers, e.g.:

- last 3 or 4 transformer layers,
- sometimes both last token and mean‑pooled tokens per layer.

This increased the feature dimension from 896 up to ~2700. While the train AUROC often reached ≈100%, the test AUROC dropped to ~52–53%, barely above random. This matches the warning from hallucination‑detection surveys that high‑capacity probes with large feature spaces can easily overfit small labeled datasets [web:218].

**Outcome:** Discarded. The model overfitted strongly; I returned to a single‑layer base (last token of last layer) plus a small number of well‑motivated geometric features.

### 3.2. Heavy geometric feature sets (layer-wise norms, full drift curves)

I implemented richer geometric descriptors:

- Layer‑wise mean and std of token L2 norms across all layers.
- Full representation‑drift curves: cosine similarity between consecutive layers for the last token.
- Multiple spectral statistics beyond entropy and max eigenvalue.

This again pushed the feature dimensionality well above 2k. Although the approach was theoretically motivated by ICR‑style probing and EigenScore [web:101][web:147][web:203], in practice:

- Train AUROC was close to 1.0.
- Test AUROC degraded to ≈0.52–0.55.

**Outcome:** Discarded. The features were too many and too noisy for 689 examples. The final model keeps only 6 carefully chosen geometric scalars.

### 3.3. More complex probe architectures (residual blocks, label smoothing)

I experimented with:

- deeper MLPs with residual connections;
- more aggressive bottlenecks and label smoothing.

These architectures occasionally matched the 72–73% AUROC baseline, but were not consistently better and sometimes underperformed due to optimization instability on such a small dataset. In line with probing best practices [web:145], I opted for the simplest architecture that achieved the highest AUROC reliably.

**Outcome:** Discarded for the final submission. The 2‑layer MLP with BatchNorm + Dropout was both simpler and consistently strong.

---

## 4. Final Results

On the provided single stratified split, my final configuration (last token + 6 geometric features + 2‑layer MLP probe) achieves:

- **Test AUROC:** **75.56%**
- Test Accuracy: **73.08%**
- Test F1: **80.82%**
- Train AUROC: **98.02%**
- Feature dimensionality: **902**
- Total labelled samples: **689**

(The exact numbers above come from `results.json`:
`avg_test_auroc = 0.7556`, `avg_test_accuracy ≈ 0.7308`, `avg_test_f1 ≈ 0.8082`.)

The improvement from ~72–73% AUROC (strong baseline with only last token) to **75.56%** comes from a small number of theoretically motivated geometric features inspired by recent work on internal‑state‑based hallucination detection and spectral feature analysis [web:101][web:127][web:147][web:203].

---

## 5. Files Modified

I only modified:

1. **`aggregation.py`**
   - Base aggregation: last real token of the final layer (dim = 896).
   - Added a 6‑dimensional geometric feature extractor based on:
     - L2 norms at late layers,
     - inter‑layer cosine drift,
     - spectral entropy and max eigenvalue of the last‑layer Gram matrix.

2. **`probe.py`**
   - Implemented `HallucinationProbe` as a 2‑layer MLP with BatchNorm, Dropout, and AdamW.
   - Added threshold tuning on validation data (`fit_hyperparameters`).
   - Kept the required public interface (`fit`, `fit_hyperparameters`, `predict`, `predict_proba`).

All other infrastructure files (`model.py`, `evaluate.py`, `splitting.py`, `solution.py`) are used as provided in the starter kit, ensuring the solution is fully reproducible with a single `python solution.py` run.
