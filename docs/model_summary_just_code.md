# TrajCast Model Blueprint

This document distills the mathematical structure and implementation details of the TrajCast framework so that the complete solution—data processing, model components, training, and rollout—can be reconstructed without reverse‑engineering the source code.

---

## 1. Atomic Graph Representation

- **Graph definition**: Each simulation frame is converted into an atomic graph $ G = (V, E) $ where nodes $ i \in V $ represent atoms and edges $ (i, j) \in E $ connect atoms within a cutoff $ r_\text{cut} $.
- **Node attributes**: Positions $ \mathbf{r}_i $, velocities $ \mathbf{v}_i $, displacements to the next frame $ \Delta\mathbf{r}_i $, velocity updates $ \Delta\mathbf{v}_i $, atomic masses $ m_i $, element identifiers, and optional timestep scalars.
- **Edge attributes**: Relative vectors $\mathbf{r}_{ij} = \mathbf{r}_j - \mathbf{r}_i + \mathbf{s}_{ij}$ with periodic shift $ \mathbf{s}_{ij} $ from `torch_nl`. Their norms $ r_{ij} = \|\mathbf{r}_{ij}\| $ and spherical harmonics $ Y_{\ell m}(\hat{\mathbf{r}}_{ij}) $ up to order $ \ell_\text{max} $ are stored.
- **Neighbor list**: `AtomicGraph.update_edge_index()` recomputes `edge_index` and lattice shifts at every rollout step to keep the radius graph consistent as atoms move.
- **Type mapping**: `AtomicGraph.from_atoms_dict()` replaces atomic numbers with contiguous species IDs; masses are attached and `TOTAL_MASS_KEY` is precomputed for conservation steps.

---

## 2. Feature Engineering Pipeline

Let $ \mathcal{I} $ denote irreducible representations (irreps) of $ \mathrm{O}(3) $ using `e3nn`. The encoding stack (shared by `TrajCastModel` and `EfficientTrajCastModel`) produces steerable node and edge features:

1. **Target normalization**: `NormalizationLayer` standardizes each predicted field $ y \in \{\Delta\mathbf{r}, \Delta\mathbf{v}\} $ using dataset means $ \mu_y $ and RMS $ \sigma_y $. In training these normalized targets are learned; inference rescales predictions via stored buffers.
2. **Species embedding**: `OneHotAtomTypeEncoding` maps element IDs to $ \mathrm{Irrep}(0^+) $ channels (one per species).
3. **Radial basis for edges**: `EdgeLengthEncoding` computes $ \phi_r(r_{ij}) $ using a trainable Bessel basis (size `num_edge_rbf`) multiplied by a polynomial cutoff $ f_\text{cut}(r_{ij}) $.
4. **Velocity norm embedding**: `ElementBasedNormEncoding` encodes $ \|\mathbf{v}_i\| $ with a Gaussian radial basis (size `num_vel_rbf`, range $[0, v_\text{max}]$). A tensor product with the element one‑hot channels yields species-aware scalars $ \phi_v(\|\mathbf{v}_i\|, Z_i) $.
5. **Spherical harmonics**:
   - `SphericalHarmonicProjection` on edges yields $ Y_{\ell m}(\hat{\mathbf{r}}_{ij}) $ for $ 0 \le \ell \le \ell_\text{max} $.
   - Velocities are projected to $ Y_{\ell m}(\hat{\mathbf{v}}_i) $ (excluding the $\ell=0$ component when building node features).
6. **Initial node features**:
   $$
   \mathbf{h}_i^{(0)} = \big[ \text{onehot}(Z_i),\; \phi_v(\|\mathbf{v}_i\|, Z_i),\; Y_{\ell m}(\hat{\mathbf{v}}_i)_{\ell \ge 1} \big]
   $$
   which is mixed through `LinearTensorMixer` into
   $$
   \mathbf{h}_i^{(0)} \in \bigoplus_{\ell=0}^{\ell_\text{max}} n_\text{hid}\times \ell,
   $$
   where `num_hidden_channels` = $ n_\text{hid} $ is the multiplicity per rotation order.

---

## 3. Message Passing Core

TrajCast stacks `num_mp_layers` residual message passing blocks that are strictly equivariant:

### 3.1 Edge message formation
$$
\mathbf{m}_{ij} = \text{DTP}_\text{edge}\big(\mathbf{h}_j^{(\ell)},\; Y_{\ell m}(\hat{\mathbf{r}}_{ij});\; w_r(r_{ij})\big),
$$
where `DepthwiseTensorProduct` contracts node features with edge harmonics. The weights $ w_r = \mathrm{MLP}_r(\phi_r(r_{ij})) $ are shared across species and learned.

### 3.2 Neighborhood pooling
$$
\tilde{\mathbf{m}}_i = \frac{1}{\bar{N}} \sum_{j \in \mathcal{N}(i)} \mathbf{m}_{ij},
$$
with $\bar{N}$ (`avg_num_neighbors`) acting as a learned normalization constant for stability when neighbor counts fluctuate.

### 3.3 Velocity conditioning
Messages are further modulated by each atom’s instantaneous velocity orientation:
$$
\mathbf{c}_i = \text{DTP}_\text{vel}\big( \mathcal{L}(\tilde{\mathbf{m}}_i),\; Y_{\ell m}(\hat{\mathbf{v}}_i);\; w_v(\|\mathbf{v}_i\|)\big),
$$
where $ \mathcal{L} $ is a linear contraction reducing multiplicities and $ w_v = \mathrm{MLP}_v(\phi_v(\|\mathbf{v}_i\|, Z_i)) $. This step injects dynamical conditioning, enabling the network to distinguish configurations with identical positions but different velocities—key for autoregressive rollout.

### 3.4 Gated update + residual shortcut
- The conditioned tensor is mixed by `LinearTensorMixer`, split into scalars and higher-order parts, then passed through `GatedNonLinearity` (activation defaults: scalar even/odd → SiLU/Tanh, gates → SiLU). This enforces equivariant non-linearity per [Weiler et al., 2018].
- A residual shortcut `tp_resnet` multiplies the previous features with the one‑hot species embedding, enabling element-specific biases without breaking equivariance.
- Final update:
  $$
  \mathbf{h}_i^{(\ell+1)} = \frac{1}{\sqrt{2}} \left( \text{Gate}(\mathbf{c}_i) + \text{TP}_\text{res}\big(\mathbf{h}_i^{(\ell)}, \text{onehot}(Z_i)\big) \right).
  $$

`ConditionedMessagePassingLayer` (non-residual) follows the same equations but omits the shortcut and contraction; it is used for optional blocks in `FlexibleModel`.

---

## 4. Readout, Constraints, and Outputs

1. **Compression head**: After the last MP layer, features are linearly compressed (typically by factor 4) into mixed scalar/vector irreps.
2. **Target projection**: `LinearTensorMixer` maps compressed features to the concatenated target irrep
   $$
   \mathbf{y}_i = \big[\Delta\mathbf{r}_i,\; \Delta\mathbf{v}_i\big] \in \mathrm{Irrep}(1^-) \oplus \mathrm{Irrep}(1^-).
   $$
3. **Conservation layer**: `ConservationLayer` enforces physics before unnormalizing:
   - **Linear momentum**: Adjusts predicted velocities so that $ \sum_i m_i \mathbf{v}_i^\text{new} = \sum_i m_i \mathbf{v}_i^\text{ref} $ (or a user-specified target). If the target momentum is zero, displacements are also shifted to keep the center of mass fixed.
   - **Angular momentum (optional)**: Solves $ \mathbf{L} = \sum_i (\mathbf{r}_i - \mathbf{r}_\text{COM}) \times m_i \mathbf{v}_i $, computes inertia tensor $ \mathbf{I} $, and corrects angular velocities via $ \boldsymbol{\omega} = \mathbf{I}^{-1} (\mathbf{L}_\text{ref} - \mathbf{L}_\text{pred}) $.
   - Normalization constants (`disp_norm_const`, `vel_norm_const`) reverse the earlier target scaling.

4. **Efficient variant**: `EfficientTrajCastModel` shares the same mathematics but caches encoding layers and transposes cueEquiv outputs to minimize overhead.

---

## 5. Loss Functions and Training Objective

- **Targets**: stacked tensor $ \mathbf{t}_i = [\Delta\mathbf{r}_i, \Delta\mathbf{v}_i] $ from MD trajectories.
- **Main loss**: configurable as MAE or MSE
  $$
  \mathcal{L}_\text{main} = \frac{1}{N} \sum_i \| \hat{\mathbf{t}}_i - \mathbf{t}_i \|_p^p, \quad p \in \{1, 2\}.
  $$
- **Cross-angle penalty (optional)**: encourages consistent relative orientations via cosine or $ L^p $ angle differences between all prediction/reference pairs.
- **Atom-wise cosine penalty (optional)**: enforces alignment of each predicted vector with its ground truth.
- **Weighting**: `MultiobjectiveLoss` concatenates chosen components and learns (or accepts fixed) positive weights $ w_k $ normalized by softmax so that the total loss is $ \sum_k w_k \mathcal{L}_k $.

Regularization occurs via:
- Gradient clipping (`max_grad_norm`).
- Neighbor-count normalization ($\bar{N}$).
- Optional time-reversal augmentation (see §7).

---

## 6. Optimization Pipeline

1. **Trainer configuration** (`trajcast/model/training.py`):
   - Loads `AtomicGraphDataset` for both train and validation, respecting precision (float32/float64).
   - Builds model type: `Flexible` (layer list defined in YAML), `TrajCast` (standard stack), or `EfficientTrajCast`.
   - Computes dataset RMS for each predicted field to populate the normalization buffers.
2. **Data loading**: PyG `DataLoader` with `batch_size` mixes multiple atomic graphs; each batch is processed in one forward call.
3. **Optimizers**: SGD, Adam, or AdamW with user-provided hyperparameters.
4. **Schedulers**: Any torch scheduler (MultiStep, Exponential, Cosine, ReduceLROnPlateau, Linear) can be chained through `CustomChainedScheduler`, which interprets milestones in steps or epochs.
5. **TensorBoard logging**: Tracks training/validation loss, per-field MAEs (displacement, velocity), LR, gradient/weight histograms, and max/min statistics. Validation runs use a separate dataset spec.
6. **Checkpointing**: `CheckpointHandler` periodically stores model + optimizer (+ scheduler) states and the best model; restarting from the last epoch is supported.

---

## 7. Dataset Interfaces and Preprocessing

- **AtomicGraphDataset**:
  - Accepts ASE-readable trajectories (EXTXYZ, LAMMPS dumps, etc.) or `.npz` archives.
  - `process()` converts every frame to `AtomicGraph`, storing masses, velocities, displacements, forces, etc. Processed objects are serialized as a single `*.pt` file for fast reload.
  - Optional **time reversibility augmentation** flips velocities and displacements and swaps initial/final positions, effectively doubling the data while preserving physics.
- **LAMMPS wrapper**: `data/wrappers/_lammps.py` converts LAMMPS dump text with unit conversions (forces, velocities, energies) and optional species remapping.
- **Field typing**: `trajcast/data/_types.py` enforces consistent dtypes; positions/velocities default to the global torch dtype, atomic numbers are `long`, etc.

---

## 8. Autoregressive Forecasting Engine

`trajcast/model/forecast.py` supplies an MD-like driver:

1. **Protocol inputs** include timestep $ \Delta t $, run length, desired temperature schedule, thermostat settings, whether to zero net momentum, and file outputs.
2. **Model loading**: Accepts either (a) instantiated PyTorch models, (b) weight checkpoint + YAML architecture, or (c) YAML-only (random weights) for experimentation. `cueEquivariance` acceleration is supported via the `o3_backend` flag.
3. **State preparation**:
   - Builds `AtomicGraph` from ASE atoms or dictionaries (with optional wrappers).
   - Fills in `batch`, timestep tensor, and ensures velocities are present. If not, `init_velocity()` samples Gaussian or uniform velocities, optionally removes COM/rotational motion, and rescales to the requested temperature using the `Temperature` module.
   - `ZeroMomentum` can periodically remove linear and/or angular momentum from the rollout state.
4. **Thermostat**: `CSVRThermostat` (canonical stochastic velocity rescaling) rescales velocities every step (or at user-defined intervals) to sample from $ NVT $. It draws from gamma distributions to preserve the kinetic energy distribution described by Bussi et al. (2007).
5. **Autoregressive step**:
   $$
   (\Delta\mathbf{r}_i, \Delta\mathbf{v}_i) = \text{Model}(G_t), \quad
   \mathbf{r}_i^{t+1} = \text{wrap}\big(\mathbf{r}_i^{t} + \Delta\mathbf{r}_i \sigma_{\Delta r}\big),\;
   \mathbf{v}_i^{t+1} = \Delta\mathbf{v}_i \sigma_{\Delta v},
   $$
   followed by thermostat/momentum adjustments and neighbor list rebuild.
6. **I/O**: Configurable writers export frames to EXTXYZ (or other ASE formats) at chosen cadence along with a CSV log of instantaneous temperatures.

---

## 9. Stabilization and “Secret Sauce”

- **Equivariant gating** across all layers keeps higher-order tensors bounded while allowing non-linear expressivity.
- **Velocity-aware conditioning** ensures updates remain sensitive to kinetic state—vital when rolling trajectories with larger $ \Delta t $ than classical integrators.
- **Neighbor normalization** ($1/\bar{N}$) prevents variance blow-up when density changes.
- **ConservationLayer** guarantees that accumulated numerical error does not introduce spurious drift in linear/Angular momentum.
- **Thermostat + ZeroMomentum** modules integrate tightly with the predictor, allowing canonical sampling or constraint enforcement without modifying the network weights.
- **cueEquivariance backend** optionally swaps e3nn operations for fused CUDA kernels, accelerating both training and inference.

---

## 10. Extensibility Hooks

- **FlexibleModel**: consumes a YAML list of layer definitions (encoders, message passing layers, normalization, conservation, timestep conditioning) so alternative architectures can be prototyped without changing code.
- **Forecast horizon conditioning**: `ForecastHorizonConditioning` couples scalar channels with sinusoidal timestep embeddings for time-aware rollouts or curriculum learning.
- **Tensor norm encoders**: `TensorNormEncoding` generalizes the radial encoder to any vector/tensor field (e.g., forces).
- **Additional targets**: Because target irreps are constructed from the field list, adding new vector quantities (e.g., forces) requires only specifying their irreps in `FIELD_IRREPS` plus data loading.

---

## 11. Key Hyperparameters (per `model` config)

| Parameter | Meaning |
|-----------|---------|
| `max_rotation_order` | Maximum spherical harmonic degree $ \ell_\text{max} $. |
| `num_hidden_channels` | Multiplicity per irrep block inside message passing. |
| `num_mp_layers` | Number of stacked residual MP layers. |
| `edge_cutoff` | Radius (Å) for neighbor graph construction. |
| `num_edge_rbf`, `num_edge_poly_cutoff` | Size of edge radial basis and cutoff polynomial order. |
| `vel_max`, `num_vel_rbf` | Range/size of velocity norm basis. |
| `edge_mlp_kwargs`, `vel_mlp_kwargs` | Hidden sizes + activations for the two weight-generating MLPs. |
| `nl_gate_kwargs` | Custom gate irreps and activation per parity. |
| `avg_num_neighbors` | Empirical normalization constant (auto-estimated if omitted). |
| `conserve_ang_mom` | Toggle for angular momentum correction. |
| `net_lin_mom`, `net_ang_mom` | Target momenta when simulating driven systems. |

---

## 12. Putting It All Together

1. **Prepare data** → convert simulation frames to `AtomicGraphDataset`, optionally augment via time reversal.
2. **Configure model** → choose architecture + hyperparameters; compute RMS statistics for all predicted fields.
3. **Train** → run the `Trainer` loop with your optimizer/scheduler choice, monitoring TensorBoard for convergence; checkpoints capture both best and periodic weights.
4. **Forecast** → feed the trained weights plus protocol into `Forecast.generate_trajectory()` to autoregressively roll MD trajectories under physical constraints, thermostats, and optional momentum controls.

With this blueprint, every component—from edge encodings to conservation corrections—is explicitly defined, enabling faithful reimplementation or targeted modifications (e.g., adding new observables, swapping encoders, or experimenting with different loss couplings).

