# PaiNN (Polarizable atom Interaction Neural Network) — math-y cheat sheet

## Graph, nodes & edges

* **Atoms (nodes):** Index $i=1,\dots,N$ with nuclear charge $Z_i \in \mathbb{N}$ and position $\mathbf{r}_i \in \mathbb{R}^3$.
* **Neighbors (directed edges):** $j \in \mathcal{N}(i) = \{ j \neq i : \|\mathbf{r}_{ij}\| \le r_{\text{cut}} \}$ with $\mathbf{r}_{ij} = \mathbf{r}_j - \mathbf{r}_i$, distance $r_{ij} = \|\mathbf{r}_{ij}\|$, and unit direction $\hat{\mathbf{r}}_{ij} = \mathbf{r}_{ij} / r_{ij}$.

## Features (per atom)

* **Scalar (rotation–invariant):** $q_i \in \mathbb{R}^{F}$.
* **Vector (rotation–equivariant):** $\boldsymbol{\mu}_i \in \mathbb{R}^{3 \times F}$ (3D direction for each of $F$ channels).
* **Init:** $q_i^{(0)} = \mathrm{Embed}(Z_i)$, $\boldsymbol{\mu}_i^{(0)} = \mathbf{0}$.

## Edge feature engineering (radial)

* **Radial basis:** $\boldsymbol{\phi}(r_{ij}) \in \mathbb{R}^{B}$, e.g.
  $$\phi_n(r) = \frac{\sin\!\left(n\pi r / r_{\text{cut}}\right)}{r}, \quad n=1,\dots,B.$$
* **Cutoff:** $f_{\text{cut}}(r)$ (e.g., cosine) to enforce smooth locality.
* **Filters (per edge):**
  $$\mathbf{W}_{ij} = \mathrm{MLP}_{\text{filter}}\big(\boldsymbol{\phi}(r_{ij})\big) \cdot f_{\text{cut}}(r_{ij}).$$
  (Optionally shared across interaction blocks.)

## High-level architecture

Repeat **L** times (interaction + mixing), with residuals:
$$
(q_i,\boldsymbol{\mu}_i) \xrightarrow{\text{Interaction}} (q_i',\boldsymbol{\mu}_i') \xrightarrow{\text{Mixing}} (q_i^{+},\boldsymbol{\mu}_i^{+}).
$$

---

## 1) Interatomic **interaction** (message passing)

A small atomwise MLP produces **3F** per-atom channels:
$$
\mathbf{x}_i = \mathrm{MLP}_{\text{inter}}(q_i) \in \mathbb{R}^{3F}, \qquad
\mathbf{x}_i \equiv \big[\underbrace{\mathbf{x}_i^{(q)}}_{F},\underbrace{\mathbf{x}_i^{(R)}}_{F},\underbrace{\mathbf{x}_i^{(\mu)}}_{F}\big].
$$

For each edge $(i \leftarrow j)$:

* **Gate by filter:** $\tilde{\mathbf{x}}_{ij} = \mathbf{W}_{ij} \odot \mathbf{x}_j$.
* **Split:** $\tilde{\mathbf{x}}_{ij}^{(q)}, \tilde{\mathbf{x}}_{ij}^{(R)}, \tilde{\mathbf{x}}_{ij}^{(\mu)} \in \mathbb{R}^{F}$.

Aggregate to the receiver $i$ (degree-normalized mean over neighbors):

* **Scalar message (to $q_i$):**
  $$\Delta q_i^{\text{(m)}} = \frac{1}{|\mathcal{N}(i)|} \sum_{j \in \mathcal{N}(i)} \tilde{\mathbf{x}}_{ij}^{(q)}.$$
* **Vector message (to $\boldsymbol{\mu}_i$):**
  $$
  \Delta \boldsymbol{\mu}_i^{\text{(m)}} = \frac{1}{|\mathcal{N}(i)|} \sum_{j \in \mathcal{N}(i)} \left(
  \tilde{\mathbf{x}}_{ij}^{(R)} \otimes \hat{\mathbf{r}}_{ij}
  + \tilde{\mathbf{x}}_{ij}^{(\mu)} \odot \boldsymbol{\mu}_j
  \right).
  $$

Note: While the original PaiNN uses neighbor summation, for the N-body rollout targets (per-node `pos_dt` and `vel`) normalizing by the in-degree empirically stabilizes training without affecting equivariance. This matches the EGNN coordinate update, which averages over neighbors.
  (First term injects **new** directional info via $\hat{\mathbf{r}}_{ij}$; second term **propagates** existing equivariant info $\boldsymbol{\mu}_j$ across the graph.)

Residual update:
$$
q_i \leftarrow q_i + \Delta q_i^{\text{(m)}}, \qquad
\boldsymbol{\mu}_i \leftarrow \boldsymbol{\mu}_i + \Delta \boldsymbol{\mu}_i^{\text{(m)}}.
$$

---

## 2) Intra-atomic **mixing** (equivariant gating)

Linear mix two vector streams from $\boldsymbol{\mu}_i$:
$$
[\boldsymbol{\mu}_{V,i}\ \ \boldsymbol{\mu}_{W,i}] = \mathrm{Linear}(\boldsymbol{\mu}_i), \qquad
\boldsymbol{\mu}_{V,i},\ \boldsymbol{\mu}_{W,i} \in \mathbb{R}^{3 \times F}.
$$
Compute per-feature norms (used as scalar context):
$$
\mathbf{s}_{V,i} = \|\boldsymbol{\mu}_{V,i}\|_2 \in \mathbb{R}^{F} \quad \text{(norm over the 3D axis)}.
$$
Feed $[q_i; \mathbf{s}_{V,i}]$ to an MLP yielding **3F** scalars:
$$
\big[\delta \mathbf{q}^{\text{(intra)}},\ \delta \boldsymbol{\mu}^{\text{(intra)}},\ \delta \mathbf{q\mu}^{\text{(intra)}}\big]
= \mathrm{MLP}_{\text{intra}}\big([q_i; \mathbf{s}_{V,i}]\big).
$$

Form scalar-vector couplings:
$$
\langle \boldsymbol{\mu}_{V,i}, \boldsymbol{\mu}_{W,i} \rangle
= \sum_{a=1}^3 \boldsymbol{\mu}_{V,i}^{(a)} \odot \boldsymbol{\mu}_{W,i}^{(a)} \in \mathbb{R}^{F}.
$$

Residual updates (per feature $f=1,\dots,F$):
$$
\begin{aligned}
q_i &\leftarrow q_i + \delta \mathbf{q}^{\text{(intra)}}
+ \delta \mathbf{q\mu}^{\text{(intra)}} \odot \langle \boldsymbol{\mu}_{V,i}, \boldsymbol{\mu}_{W,i} \rangle, \\
\boldsymbol{\mu}_i &\leftarrow \boldsymbol{\mu}_i
+ \big(\delta \boldsymbol{\mu}^{\text{(intra)}} \odot \boldsymbol{\mu}_{W,i}\big).
\end{aligned}
$$

These gates are **equivariant**: all nonlinearities are scalar; vectors are only scaled or linearly combined, preserving $R\boldsymbol{\mu}$ under a global rotation $R \in \mathrm{SO}(3)$.

---

## Symmetries & complexity

* **Translation invariance:** only relative positions $\mathbf{r}_{ij}$ are used.
* **Permutation invariance:** atomwise updates + neighbor sums.
* **Rotation properties:** scalars $q_i$ invariant; vectors $\boldsymbol{\mu}_i$ equivariant.
* **Per-block cost:** $\mathcal{O}(|E|F)$ with $|E| = \sum_i |\mathcal{N}(i)| = \mathcal{O}(N)$ for fixed cutoff; avoids $\mathcal{O}(N|\mathcal{N}|^2)$ angle triplets.

---

## Readouts (examples)

### (a) Scalar properties (e.g., energy)

Atomwise head $\varepsilon(\cdot)$ on $q_i$ and sum:
$$
E = \sum_{i=1}^N \varepsilon(q_i).
$$
**Forces** enforced by energy conservation:
$$
\mathbf{F}_i = -\frac{\partial E}{\partial \mathbf{r}_i}.
$$

### (b) Molecular dipole (rank-1 tensor)

Latent atomic **charges** and **dipoles** from final features:
$$
\boldsymbol{\mu} = \sum_{i=1}^N \left(
\underbrace{\boldsymbol{\mu}_{\text{atom}}(\boldsymbol{\mu}_i)}_{\text{local dipole}}
+ \underbrace{q_{\text{atom}}(q_i)}_{\text{charge}} \ \mathbf{r}_i
\right).
$$

### (c) Polarizability (rank-2 tensor)

$$
\boldsymbol{\alpha} = \sum_{i=1}^N \left(
\alpha_0(q_i)\mathbf{I}_3
+ \boldsymbol{\nu}(\boldsymbol{\mu}_i) \otimes \mathbf{r}_i
+ \mathbf{r}_i \otimes \boldsymbol{\nu}(\boldsymbol{\mu}_i)
\right).
$$

(Here, $\varepsilon$, $q_{\text{atom}}$, $\boldsymbol{\mu}_{\text{atom}}$, $\alpha_0$, $\boldsymbol{\nu}$ are small MLP heads; $\boldsymbol{\nu}(\cdot) \in \mathbb{R}^3$.)

---

## Layer shapes & flow (one block)

* **Inputs:** $q \in \mathbb{R}^{N \times F}$, $\boldsymbol{\mu} \in \mathbb{R}^{N \times 3 \times F}$.
* **Edge preprocessing:** $r_{ij}$, $\hat{\mathbf{r}}_{ij}$, $\boldsymbol{\phi}(r_{ij})$, $\mathbf{W}_{ij}$.
* **Interaction MLP:** $F \to 3F$ per atom → gate per edge → neighbor sum.
* **Mixing:** vector linear $F \to 2F$; norm over $3$ → concat $[q; s_V] \in \mathbb{R}^{2F}$ → MLP $2F \to 3F$.
* **Residuals:** add to get next $(q,\boldsymbol{\mu})$.
* **Repeat** $L$ times; final heads read from $q$ and/or $\boldsymbol{\mu}$.

---

## Why it works (core math ideas)

* **Equivariance preserved** by restricting nonlinearities to scalars and using only linear/scalar scaling on vectors.
* **Directional information** is injected via $\hat{\mathbf{r}}_{ij}$ and **propagated** across hops via the $\tilde{\mathbf{x}}_{ij}^{(\mu)} \odot \boldsymbol{\mu}_j$ term, achieving angular sensitivity with **linear** neighbor complexity $\mathcal{O}(|\mathcal{N}(i)|)$, unlike angle-triplet models.
* **Scalar–vector coupling** (inner products, norms) lets vectors influence scalars while maintaining invariance of scalar channels.

---

## Minimal set of equations (all together)

**Edge filters**
$$
\mathbf{W}_{ij} = \mathrm{MLP}_{\text{filter}}(\boldsymbol{\phi}(r_{ij})) \cdot f_{\text{cut}}(r_{ij}), \qquad \mathbf{W}_{ij} \in \mathbb{R}^{3F}.
$$

**Interaction**
$$
\begin{aligned}
\mathbf{x}_j &= \mathrm{MLP}_{\text{inter}}(q_j) \in \mathbb{R}^{3F}, \\
\tilde{\mathbf{x}}_{ij} &= \mathbf{W}_{ij} \odot \mathbf{x}_j, \\
\Delta q_i^{\text{(m)}} &= \sum_{j\in\mathcal{N}(i)} \tilde{\mathbf{x}}_{ij}^{(q)}, \\
\Delta \boldsymbol{\mu}_i^{\text{(m)}} &= \sum_{j\in\mathcal{N}(i)} \left(
\tilde{\mathbf{x}}_{ij}^{(R)} \otimes \hat{\mathbf{r}}_{ij}
+ \tilde{\mathbf{x}}_{ij}^{(\mu)} \odot \boldsymbol{\mu}_j
\right).
\end{aligned}
$$

**Mixing**
$$
\begin{aligned}
[\boldsymbol{\mu}_{V,i}, \boldsymbol{\mu}_{W,i}] &= \mathrm{Linear}(\boldsymbol{\mu}_i), \\
\mathbf{s}_{V,i} &= \|\boldsymbol{\mu}_{V,i}\|_2, \\
[\delta\mathbf{q}, \delta\boldsymbol{\mu}, \delta\mathbf{q\mu}] &= \mathrm{MLP}_{\text{intra}}([q_i; \mathbf{s}_{V,i}]), \\
q_i &\leftarrow q_i + \Delta q_i^{\text{(m)}} + \delta\mathbf{q} + \delta\mathbf{q\mu} \odot \langle \boldsymbol{\mu}_{V,i}, \boldsymbol{\mu}_{W,i} \rangle, \\
\boldsymbol{\mu}_i &\leftarrow \boldsymbol{\mu}_i + \Delta \boldsymbol{\mu}_i^{\text{(m)}} + \delta\boldsymbol{\mu} \odot \boldsymbol{\mu}_{W,i}.
\end{aligned}
$$

That’s the PaiNN core: scalable equivariant message passing (via $\hat{\mathbf{r}}_{ij}$ and vector transport), plus tight scalar–vector coupling for expressive, symmetry-respecting predictions of both scalar and tensorial molecular properties.

---

## Stabilized PaiNN for N-body (current working variant) and ablation findings

This section documents the variant we now train stably on the N-body dataset and summarizes why the original setting exploded and how we fixed it. All modifications preserve permutation/translation invariance and $\mathrm{SO}(3)$ equivariance (vector operations are isotropic over the 3D axis).

### Where the explosions came from

In deeper layers (L4–L5), the mixing residual
$$
q_i \leftarrow q_i + \delta \mathbf{q} + \underbrace{\delta \mathbf{q\mu} \odot \langle \boldsymbol{\mu}_{V,i}, \boldsymbol{\mu}_{W,i} \rangle}_{\text{quadratic in }\|\boldsymbol{\mu}\|}
\quad \text{and} \quad
\boldsymbol{\mu}_i \leftarrow \boldsymbol{\mu}_i + \underbrace{\big(\delta \boldsymbol{\mu}\big) \odot \boldsymbol{\mu}_{W,i}}_{\text{amplifies }\|\boldsymbol{\mu}\|}
$$
created a positive feedback loop: large $\|\boldsymbol{\mu}\|$ increased $\delta \mathbf{q\mu}$ (via the MLP), which in turn increased $q$, leading to larger interaction messages and even larger $\|\boldsymbol{\mu}\|$ in subsequent layers.

### Stabilized interaction (message passing)

We keep the standard filtered messages but add degree-normalized aggregation and bounded pre/post-aggregation magnitudes.

1) Filters with global gain $g$ and cutoff
$$
\mathbf{W}_{ij} = g\,\mathrm{MLP}_{\!f}\big(\boldsymbol{\phi}(r_{ij})\big)\, f_{\text{cut}}(r_{ij}),\qquad g>0.
$$

2) Per-edge message gating with tanh saturation (scale $s_{\text{msg}}$)
$$
\begin{aligned}
\tilde{\mathbf{x}}_{ij}^{(q)} &= \operatorname{sat}_{s_{\text{msg}}}\!\left(\mathbf{x}_j^{(q)} \odot \mathbf{W}_{ij}^{(q)}\right),\\
\tilde{\mathbf{x}}_{ij}^{(R)} &= \operatorname{sat}_{s_{\text{msg}}}\!\left(\mathbf{x}_j^{(R)} \odot \mathbf{W}_{ij}^{(R)}\right),\\
\tilde{\mathbf{x}}_{ij}^{(\mu)} &= \operatorname{sat}_{s_{\text{msg}}}\!\left(\mathbf{x}_j^{(\mu)} \odot \mathbf{W}_{ij}^{(\mu)}\right),
\end{aligned}
$$
with $\operatorname{sat}_{s}(u)= s\,\tanh(u/s)$ applied elementwise.

3) Messages and degree-normalized aggregation
$$
\begin{aligned}
\Delta q_i^{\text{(m)}} &= \frac{1}{|\mathcal{N}(i)|}\sum_{j\in\mathcal{N}(i)} \tilde{\mathbf{x}}_{ij}^{(q)},\\
\Delta \boldsymbol{\mu}_i^{\text{(m)}} &= \frac{1}{|\mathcal{N}(i)|}\sum_{j\in\mathcal{N}(i)}\Big(\tilde{\mathbf{x}}_{ij}^{(R)}\otimes \hat{\mathbf{r}}_{ij} + \tilde{\mathbf{x}}_{ij}^{(\mu)}\odot \boldsymbol{\mu}_j\Big).
\end{aligned}
$$

4) Post-aggregation safety clamps (optional)
$$
\begin{aligned}
\Delta q_i^{\text{(m)}} &\leftarrow \operatorname{clip}\big(\Delta q_i^{\text{(m)}},\;[-C_s,\,C_s]\big),\\
\Delta \boldsymbol{\mu}_i^{\text{(m)}} &\leftarrow \operatorname{clip\_norm}\big(\Delta \boldsymbol{\mu}_i^{\text{(m)}},\; C_v\big),
\end{aligned}
$$
where $\operatorname{clip\_norm}(\cdot,C)$ rescales each feature’s 3D vector to have L2-norm at most $C$ (isotropic, preserves equivariance).

5) Residual update with scale $\alpha_{\text{inter}}$:
$$
q_i \leftarrow q_i + \alpha_{\text{inter}}\,\Delta q_i^{\text{(m)}},\qquad
\boldsymbol{\mu}_i \leftarrow \boldsymbol{\mu}_i + \alpha_{\text{inter}}\,\Delta \boldsymbol{\mu}_i^{\text{(m)}}.
$$

### Stabilized mixing (equivariant gating)

We keep the original structure but bound the scalars and apply a residual scale, followed by state clamps.

1) Vector linear and norms as before:
$$
[\boldsymbol{\mu}_{V,i}\ \ \boldsymbol{\mu}_{W,i}] = \mathrm{Linear}(\boldsymbol{\mu}_i),\quad \mathbf{s}_{V,i}=\|\boldsymbol{\mu}_{V,i}\|_2.
$$

2) Bounded scalar outputs (scale $s_{\text{mix}}$)
$$
[\delta\mathbf{q},\,\delta\boldsymbol{\mu},\,\delta\mathbf{q\mu}] = \operatorname{sat}_{s_{\text{mix}}}\!\Big(\mathrm{MLP}_{\text{intra}}\big([q_i;\mathbf{s}_{V,i}]\big)\Big).
$$

3) Residual with scale $\alpha_{\text{mix}}$:
$$
\begin{aligned}
q_i &\leftarrow q_i + \alpha_{\text{mix}}\,\Big(\delta\mathbf{q} + \delta\mathbf{q\mu} \odot \langle \boldsymbol{\mu}_{V,i},\boldsymbol{\mu}_{W,i} \rangle\Big),\\
\boldsymbol{\mu}_i &\leftarrow \boldsymbol{\mu}_i + \alpha_{\text{mix}}\,\Big(\delta\boldsymbol{\mu} \odot \boldsymbol{\mu}_{W,i}\Big).
\end{aligned}
$$

4) Post-mixing state clamps (optional)
$$
q_i \leftarrow \operatorname{clip}(q_i,[-C_q,\,C_q]),\qquad
\boldsymbol{\mu}_i \leftarrow \operatorname{clip\_norm}(\boldsymbol{\mu}_i, C_{\mu}).
$$

### Hyperparameters used in the stable run (R2_full)

$$
\begin{aligned}
&\text{Degree norm: mean (}1/|\mathcal{N}(i)|\text{)};\quad g=0.5;\quad s_{\text{msg}}=5;\quad s_{\text{mix}}=5;\\
&\alpha_{\text{inter}}=0.5;\quad \alpha_{\text{mix}}=0.5;\\
&C_v=10;\quad C_s=10;\quad C_{\mu}=20;\quad C_q=100.
\end{aligned}
$$

These bounds kept per-layer statistics finite with no NaN/Inf events while preserving the desired symmetries.
