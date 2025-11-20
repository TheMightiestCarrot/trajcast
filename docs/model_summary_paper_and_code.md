# TrajCast Architecture & Training Blueprint (updated 2025-11-18)

> Combined reference that unifies the arXiv preprint description with the implementation blueprints so you can rebuild TrajCast end-to-end—graph construction, encoders, equivariant tensor algebra, conservation tricks, thermostatting, datasets, optimization, and rollout tooling—without reverse‑engineering the codebase.

## 1. Autoregressive Graph Formulation
- **Atomic graph definition.** Each rollout step constructs a periodic graph $\mathcal{G}_k = (\mathcal{V}, \mathcal{E}, \mathbf{X})$ where nodes carry positions $\mathbf{r}_i$, velocities $\mathbf{v}_i$, element IDs $Z_i$, masses $m_i$, cached displacements $\Delta\mathbf{r}_i$, velocity updates $\Delta\mathbf{v}_i$, optional timestep scalars, and any user-specified fields. Edges connect atoms satisfying $\|\mathbf{r}_i - \mathbf{r}_j\| < r_\text{cut}$ and store lattice shifts from `torch_nl` so periodic wrapping stays consistent.
- **Neighbor maintenance.** `AtomicGraph.update_edge_index()` recomputes `edge_index` and lattice offsets each rollout step, while `AtomicGraph.from_atoms_dict()` remaps $Z$ to contiguous species IDs, attaches masses, and precomputes `TOTAL_MASS_KEY` for conservation layers.
- **Autoregressive operator.** The learned evolution rule replaces classical integrators at coarse $\Delta t$ by directly forecasting displacements and velocity increments:
  $$
  \mathbf{y}_{k+1} = F_\theta(\mathcal{G}_k, \Delta t) = \big(\mathbf{R}_k + \Delta \hat{\mathbf{R}}_k,\; \mathbf{V}_k + \Delta \hat{\mathbf{V}}_k\big).
  $$
- **General message passing form.** Latent node states follow the Gilmer MP framework, later specialized to equivariant tensor products:
  $$
  \mathbf{m}_i^t = \sum_{j \in \mathcal{N}(i)} M^{(t)}(\mathbf{h}_i^t, \mathbf{h}_j^t, \mathbf{e}_{ij}), \qquad \mathbf{h}_i^{t+1} = U^{(t)}(\mathbf{h}_i^t, \mathbf{m}_i^t).
  $$

## 2. Feature Engineering Pipeline
1. **Target normalization.** `NormalizationLayer` keeps per-field means $\mu_y$ and RMS $\sigma_y$ for $y \in \{\Delta\mathbf{r}, \Delta\mathbf{v}, ...\}$ so the network trains on z-scored targets and rescales predictions during inference.
2. **Atom-type encoding.** `OneHotAtomTypeEncoding` maps species IDs into $0^+$ irreps; these scalars seed every node’s equivariant feature dictionary.
3. **Radial edge filters.** `EdgeLengthEncoding` evaluates a trainable Bessel basis of size `num_edge_rbf`, multiplies it by a polynomial cutoff $C_p(r_{ij}; r_\text{cut})$, and feeds the result to an MLP that produces tensor-product weights $\Phi^{(t)}(r_{ij})$.
4. **Velocity-norm encoding.** `ElementBasedNormEncoding` projects $v_i = \|\mathbf{v}_i\|$ onto Gaussian basis elements $\phi_m$, then mixes them with species-specific weights $W_{Z_i,m}$ to obtain
  $$
  R_\mu(v_i) = \sum_{m=1}^M W_{Z_i,m}\,\phi_m(v_i),
  $$
  which modulates both edge and velocity-conditioning weights.
5. **Spherical harmonics.** `SphericalHarmonicProjection` provides real $Y_{\ell m}(\hat{\mathbf{r}}_{ij})$ (edges) up to `max_rotation_order`. Velocities are projected to $Y_{\ell m}(\hat{\mathbf{v}}_i)$, excluding $\ell=0$ to avoid double-counting scalars.
6. **Initial node features.** Concatenate species one-hot, $R(v_i)$, and the directional velocity components, then mix via `LinearTensorMixer` into $\bigoplus_{\ell=0}^{\ell_\text{max}} n_\text{hid}\times(\ell^+ \oplus \ell^-)$ where `num_hidden_channels`=$n_\text{hid}$.
7. **Optional conditioning.** `TimestepEncoding` and `ForecastHorizonConditioning` inject scalars like $\Delta t$ or curriculum horizons; `TensorNormEncoding` generalizes the radial pipeline to arbitrary tensor fields when adding new targets.

## 3. Message Passing & Equivariance
- **Edge-wise convolution.** Messages apply Clebsch–Gordan tensor products between sender irreps and edge harmonics:
  $$
  \mathbf{m}_{j\rightarrow i}^{(t)} = \big(\mathbf{h}_j^{(t)} \otimes_\text{CG} \mathbf{Y}(\hat{\mathbf{r}}_{ij})\big) \, \Phi^{(t)}(r_{ij}).
  $$
- **Neighbor pooling.** Pooled messages divide by the buffered average neighbor count $\bar{N}$ (estimated from data or provided) for density robustness:
  $$
  \tilde{\mathbf{m}}_i^{(t)} = \frac{1}{\bar{N}} \sum_{j \in \mathcal{N}(i)} \mathbf{m}_{j\rightarrow i}^{(t)}.
  $$
- **Velocity conditioning.** A second CG product injects the target node’s instantaneous velocity orientation, weighted by $R(v_i)$:
  $$
  \mathbf{c}_i^{(t)} = \tilde{\mathbf{m}}_i^{(t)} \otimes_\text{CG} \mathbf{Y}(\hat{\mathbf{v}}_i)\; \Psi^{(t)}(v_i, Z_i).
  $$
  This disambiguates states with identical positions but different kinetics and is critical for stable autoregressive rollouts.
- **Residual gated updates.** `ResidualConditionedMessagePassingLayer` contracts $\mathbf{c}_i^{(t)}$ with `LinearTensorMixer`, adds a species-conditioned shortcut $\mathbf{h}_i^{(t)} \otimes \text{onehot}(Z_i)$, scales by $1/\sqrt{2}$, and applies `GatedNonLinearity` (SiLU/Tanh for scalars, SiLU gates for higher-order parts). `ConditionedMessagePassingLayer` provides the non-residual variant for custom stacks.
- **Backend flexibility.** Setting `o3_backend="cueq"` swaps e3nn ops with CuEquivariance kernels while keeping the mathematical structure intact.

## 4. Readout, Normalization & Conservation
1. **Compression head.** After `num_mp_layers`, features are linearly compressed (often 4× smaller) before the final projection.
2. **Target projection.** `LinearTensorMixer` maps the compressed irreps into the concatenated target space $\mathrm{Irrep}(1^-) \oplus \mathrm{Irrep}(1^-)$ for $[\Delta \hat{\mathbf{r}}_i, \Delta \hat{\mathbf{v}}_i]$ (extend with additional irreps when forecasting forces or other tensors).
3. **Z-scoring buffers.** `NormalizationLayer` multiplies outputs by precomputed RMS values and adds means from the dataset so the loss sees physical units.
4. **Conservation layer.** `ConservationLayer` enforces physics before unnormalizing:
   - **Linear momentum.** Subtract COM momentum and optionally add a user-specified target $\mathbf{P}^*$:
     $$
     \mathbf{v}_i' = \hat{\mathbf{v}}_i - \frac{1}{M}\sum_j m_j \hat{\mathbf{v}}_j + \frac{\mathbf{P}^*}{M}, \quad M = \sum_j m_j.
     $$
     When $\mathbf{P}^* = 0$, predicted displacements are shifted to keep the center of mass fixed.
   - **Angular momentum (optional).** Compute inertia tensors about the COM, recover $\boldsymbol{\omega} = \mathbf{I}^{-1}(\mathbf{L}_\text{ref} - \mathbf{L}_\text{pred})$, and rotate velocities accordingly.
   - **Normalization constants.** Stored `disp_norm_const` and `vel_norm_const` invert the earlier scaling.
5. **Efficient variant.** `EfficientTrajCastModel` reuses encoders, compresses irreps earlier, and caches cueEquiv outputs to reduce memory with identical physics.

## 5. Losses, Objectives & Regularization
- **Primary targets.** $\mathbf{t}_i = [\Delta\mathbf{r}_i, \Delta\mathbf{v}_i]$ (extendable to forces or other observables).
- **Main loss.** Configurable MAE or MSE:
  $$
  \mathcal{L}_\text{main} = \frac{1}{N} \sum_i \| \hat{\mathbf{t}}_i - \mathbf{t}_i \|_p^p, \qquad p \in \{1, 2\}.
  $$
  Reported experiments in the preprint use $\lambda_d = \lambda_v = 0.5$ to weight displacement and velocity errors equally.
- **Auxiliary penalties.** Optional cross-angle penalties encourage consistent relative orientations, and atom-wise cosine penalties align each predicted vector with ground truth.
- **Multi-objective combination.** `MultiobjectiveLoss` concatenates chosen components and learns (or accepts fixed) positive weights via a softmax-normalized log-parameterization.
- **Regularization knobs.** Gradient clipping (`max_grad_norm`), neighbor-count normalization ($1/\bar{N}$), and time-reversal augmentation (see §7) stabilize training.

## 6. Optimization Pipeline
1. **Trainer bootstrap.** `trajcast/model/training.py:Trainer` validates the YAML config, seeds RNGs, and selects precision (fp32/fp64) before instantiating datasets and models.
2. **Dataset loading.** `AtomicGraphDataset(**config["data"])` materializes processed graphs; PyG `DataLoader` batches multiple graphs per forward pass.
3. **Optimizer menu.** Choose `sgd`, `adam`, or `adamw` (AMSGrad via `optimizer_settings={"amsgrad": true}`) with user-specified hyperparameters.
4. **Schedulers.** Any torch scheduler (MultiStep, Exponential, Cosine, ReduceLROnPlateau, Linear, etc.) can be chained through `CustomChainedScheduler`, which interprets milestones per-epoch or per-step and optionally monitors validation metrics.
5. **TensorBoard logging.** Tracks train/val losses, per-field MAE for displacements/velocities, learning rates, gradient/weight histograms, and extrema.
6. **Checkpointing.** `CheckpointHandler` saves periodic and best states (model + optimizer + scheduler) and supports restarting from the last checkpoint.

## 7. Datasets & Preprocessing
- **AtomicGraphDataset behavior.** Accepts ASE-readable trajectories (EXTXYZ, LAMMPS dumps, etc.) or `.npz` archives, converts each frame into `AtomicGraph`, and serializes processed data as a single `*.pt` for rapid reloads. Field dtypes are enforced via `trajcast/data/_types.py` (positions/velocities follow the global torch dtype; atomic numbers are `long`).
- **Wrappers & augmentation.** `data/wrappers/_lammps.py` handles unit conversions and species remapping for raw LAMMPS dumps. Optional time-reversal augmentation swaps initial/final frames and flips velocities/displacements, balancing forward/backward motions.
- **Published benchmark splits.**
  - *Water (64 molecules):* SPC water, 0.5 fs MD, downsampled to 5 fs windows.
  - *Quartz (α-quartz, 162 atoms):* BKS potential, equilibrated at 300 K with dual-stage thermostat/barostat, trained on 30 fs windows.
  - *Paracetamol (isolated molecule):* OPLS-AA, 0.5 fs MD, subsampled to 7 fs.
  Use the provided splits for direct comparison to the paper; custom datasets follow the same preprocessing hooks.

## 8. Forecasting Workflow, Thermostats & Momentum Filters
1. **Protocol inputs.** `trajcast/model/forecast.py` consumes timestep $\Delta t$, rollout length, thermostat schedule, zero-momentum cadence, and output writers.
2. **Model loading.** `Forecast` accepts (a) instantiated modules, (b) checkpoints + YAML configs, or (c) YAML-only definitions for random-weight experiments; `o3_backend` is restored from checkpoints when available.
3. **State preparation.** Build an `AtomicGraph` from ASE atoms or dictionaries, attach `batch` indices, ensure velocities exist, and if needed call `init_velocity()` to sample Gaussian or uniform draws, remove COM/rotational drift, and rescale to the target temperature via `Temperature` helpers.
4. **Autoregressive step.**
  $$
  (\Delta\mathbf{r}_i, \Delta\mathbf{v}_i) = \text{Model}(G_t), \qquad
  \mathbf{r}_i^{t+1} = \text{wrap}\big(\mathbf{r}_i^{t} + \Delta\mathbf{r}_i\, \sigma_{\Delta r}\big), \;
  \mathbf{v}_i^{t+1} = \Delta\mathbf{v}_i\, \sigma_{\Delta v},
  $$
  followed by conservation, thermostat/momentum filters, and neighbor-list rebuild.
5. **Thermostat math.** The microscopic temperature is
  $$
  T = \frac{2K}{N_f k_B}, \qquad K = \tfrac{1}{2}\sum_i m_i \|\mathbf{v}_i\|^2,
  $$
  with degrees of freedom $N_f$ adjusted after constraints. `CSVRThermostat` performs stochastic velocity rescaling using
  $$
  \alpha^2 = c_1 + c_2 (r_1^2 + r_2) + 2 r_1 \sqrt{c_1 c_2}, \quad c_1 = e^{-\Delta t/\tau}, \quad c_2 = \frac{(1-c_1)K_\text{target}}{K_\text{current} N_\text{dof}},
  $$
  with $r_1 \sim \mathcal{N}(0,1)$ and $r_2$ gamma-distributed.
6. **Zero-momentum filter.** `ZeroMomentum` can run every $s$ steps to remove COM drift and optionally angular momentum using the same formulas as `ConservationLayer`.
7. **Output writers.** Frames are emitted to EXTXYZ (or any ASE-supported format) at configurable cadence alongside CSV logs of instantaneous temperature and thermostat statistics.

## 9. Stabilization & Implementation Tricks
- Equivariant gating keeps higher-order tensors bounded while preserving non-linear expressivity.
- Velocity-aware conditioning maintains sensitivity to kinetic state when rolling at larger $\Delta t$ than classical integrators.
- Neighbor normalization ($1/\bar{N}$) prevents variance blow-up as density changes; average counts are cached as buffers.
- Conservation layers plus `ZeroMomentum` bound long-term drift; combining them with CSVR enables canonical sampling without altering network weights.
- `cueEquivariance` backend swaps e3nn ops for fused CUDA kernels to speed both training and inference while keeping checkpoints portable.

## 10. Model Variants & Extensibility
- **TrajCastModel.** Canonical stack matching the paper.
- **EfficientTrajCastModel.** Shares all math but aggressively reuses encoders and compresses irreps earlier for better memory/runtime trade-offs.
- **FlexibleModel.** Consumes YAML layer lists so you can mix `MessagePassingLayer`, `ConditionedMessagePassingLayer`, custom encoders, normalization blocks, and conservation modules without code changes.
- **Forecast horizon conditioning.** `ForecastHorizonConditioning` adds sinusoidal timestep embeddings for time-aware rollouts or curriculum learning.
- **Tensor norm encoders & new targets.** `TensorNormEncoding` generalizes radial encoders to arbitrary vector/tensor inputs, and additional targets only require specifying their irreps in `FIELD_IRREPS` plus data loading hooks.
- **Backend toggles.** `o3_backend` can be `"e3nn"` or `"cueq"`; the config also exposes gating irreps, cutoff bases, and conservation toggles for rapid ablations.

## 11. Hyperparameter Reference
| Parameter | Meaning |
|-----------|---------|
| `max_rotation_order` | Maximum spherical harmonic degree $\ell_\text{max}$. |
| `num_hidden_channels` | Multiplicity per irrep block inside message passing. |
| `num_mp_layers` | Number of stacked residual message-passing layers. |
| `edge_cutoff` | Neighbor radius (Å) for graph construction. |
| `num_edge_rbf`, `num_edge_poly_cutoff` | Size of the edge radial basis and cutoff polynomial order. |
| `vel_max`, `num_vel_rbf` | Range and size of the velocity-norm basis. |
| `edge_mlp_kwargs`, `vel_mlp_kwargs` | Hidden sizes and activations for the radial/velocity weighting MLPs. |
| `nl_gate_kwargs` | Custom gate irreps and activation per parity for `GatedNonLinearity`. |
| `avg_num_neighbors` | Empirical normalization constant (auto-estimated if omitted). |
| `conserve_ang_mom` | Enables angular momentum correction in `ConservationLayer`. |
| `net_lin_mom`, `net_ang_mom` | Target linear/angular momenta for driven systems. |
| `o3_backend` | Chooses e3nn vs. cuEquivariance implementations. |

## 12. Construction Recipe
1. **Prepare data.** Curate NVE/NVT trajectories, compute species mapping, select $r_\text{cut}$, and run `AtomicGraphDataset.process()` (optionally enabling time reversal or LAMMPS wrappers).
2. **Configure the model.** Populate YAML with `num_hidden_channels`, `num_mp_layers`, $\ell_\text{max}$, radial/velocity bases, gating irreps, conservation toggles, and RMS targets. Dump configs via `Trainer.dump_config_to_yaml()` for reproducibility.
3. **Train.** Launch `Trainer`, monitor TensorBoard (loss, MAE, LR, temperature), and checkpoint regularly. Select optimizer/scheduler combos appropriate for your dataset size; clip gradients if exploding.
4. **Validate.** Compare coarse rollouts to MD baselines using MSD, VACF, and RDF metrics; tune thermostat cadence and conservation settings as needed.
5. **Forecast.** Load checkpoints into `Forecast`, configure CSVR + `ZeroMomentum`, run long-horizon simulations, and export frames/logs for downstream analysis.

## 13. Key Takeaways
- Dual velocity encodings (norm + direction) let filters adapt to local flow conditions.
- Species-conditioned residuals preserve long-range context without violating equivariance.
- Physics layers (COM removal, optional angular correction, CSVR thermostat) bound integration drift even at aggressive $\Delta t$.
- Stored RMS targets and neighbor statistics tie training/inference units together, preventing train/serve skew.
- YAML-driven `FlexibleModel` and tensor-product modules make it easy to prototype new architectures while staying within the TrajCast toolchain.

## 14. Paper-Replicating Runs (commands)
- **Datasets** (repeat for `water`, `quartz`; add `--overwrite` to refresh):
```bash
python scripts/download_datasets.py --dataset paracetamol
```
- **Train — Paracetamol (Δt=7 fs)**:
```bash
python scripts/train_trajcast.py --system paracetamol --data-root data/paracetamol \
  --edge-cutoff 4.0 --num-hidden-channels 64 --num-mp-layers 4 --max-rotation-order 2 \
  --precision 64 --batch-size 10 --num-epochs 1500 --learning-rate 0.01 --max-grad-norm 0.5 \
  --vel-max 0.14 --o3-backend cueq --run-dir runs/paper/paracetamol
```
- **Train — Quartz (Δt=30 fs)**:
```bash
python scripts/train_trajcast.py --system quartz --data-root data/quartz \
  --edge-cutoff 4.5 --num-hidden-channels 64 --num-mp-layers 4 --max-rotation-order 2 \
  --precision 64 --batch-size 2 --num-epochs 1500 --learning-rate 0.01 --max-grad-norm 0.5 \
  --vel-max 0.035 --o3-backend cueq --run-dir runs/paper/quartz
```
- **Train — Water (Δt=5 fs)**:
```bash
python scripts/train_trajcast.py --system water --data-root data/water \
  --edge-cutoff 6.0 --num-hidden-channels 64 --num-mp-layers 4 --max-rotation-order 2 \
  --precision 64 --batch-size 2 --num-epochs 1500 --learning-rate 0.01 --max-grad-norm 0.5 \
  --vel-max 0.14 --o3-backend cueq --run-dir runs/paper/water
```
- **Smoke test (minutes, CPU)**:
```bash
python scripts/download_datasets.py --dataset example
python scripts/train_trajcast.py --system example --num-epochs 2 --batch-size 2 --precision 32 \
  --run-dir runs/smoke/example --device cpu --no-wandb
```
- **Pretrained inference** (pick system files from `ibm-research/trajcast.models-arxiv2025`):
```bash
# download config_e3nn.yaml and state_dict_e3nn.pt for your system
# then open examples/inference/forecasting.ipynb and set:
#   MODEL_KEY  -> path to state_dict_e3nn.pt
#   CONFIG_KEY -> path to your starting .extxyz
```
