"""PaiNN model integration for TrajCast.

This adapts the PaiNN implementation used in the Extended N-body benchmark to
the :class:`trajcast.data.atomic_graph.AtomicGraph` data structure and the
training pipeline in this repository. The wrapper class :class:`PaiNNModel`
exposes the same ``target`` field convention used by the Trainer so the model
can be selected via ``model_type: PaiNN`` in a training configuration.
"""

from __future__ import annotations

import math
from typing import Dict, Sequence

import torch
import torch.nn as nn
from torch_scatter import scatter

from trajcast.data._keys import (
    ATOMIC_MASSES_KEY,
    DISPLACEMENTS_KEY,
    EDGE_LENGTHS_KEY,
    EDGE_VECTORS_KEY,
    POSITIONS_KEY,
    UPDATE_VELOCITIES_KEY,
    VELOCITIES_KEY,
)
from trajcast.data.atomic_graph import AtomicGraph
from trajcast.model.models import AbstractModel


def gaussian_rbf(inputs: torch.Tensor, offsets: torch.Tensor, widths: torch.Tensor) -> torch.Tensor:
    """Standard Gaussian radial basis values."""

    coeff = -0.5 / torch.pow(widths, 2)
    diff = inputs[..., None] - offsets
    return torch.exp(coeff * torch.pow(diff, 2))


class GaussianRBF(nn.Module):
    """Gaussian radial basis functions mirroring the original PaiNN formulation."""

    def __init__(self, n_rbf: int, cutoff: float, start: float = 0.0, trainable: bool = False):
        super().__init__()
        self.n_rbf = n_rbf

        offsets = torch.linspace(start, cutoff, n_rbf)
        step = torch.abs(offsets[1] - offsets[0]) if n_rbf > 1 else offsets.new_tensor(cutoff - start)
        widths = torch.full_like(offsets, step)

        if trainable:
            self.offsets = nn.Parameter(offsets)
            self.widths = nn.Parameter(widths)
        else:
            self.register_buffer("offsets", offsets)
            self.register_buffer("widths", widths)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return gaussian_rbf(inputs, self.offsets, self.widths)


def cosine_cutoff(input_tensor: torch.Tensor, cutoff: torch.Tensor) -> torch.Tensor:
    """Behler-style cosine cutoff."""

    cutoff_values = 0.5 * (torch.cos(input_tensor * math.pi / cutoff) + 1.0)
    return cutoff_values * (input_tensor < cutoff).float()


class CosineCutoff(nn.Module):
    """Behler-style cosine cutoff module."""

    def __init__(self, cutoff: float):
        super().__init__()
        self.register_buffer("cutoff", torch.tensor([cutoff], dtype=torch.float32))

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return cosine_cutoff(input_tensor, self.cutoff)


class EquivariantLinear(nn.Module):
    """Linear layer acting on the feature dimension of vector features."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        # x: (N, 3, F_in)
        return torch.einsum("ncf,fo->nco", x, self.weight)


class PaiNNInteraction(nn.Module):
    """Implements the message passing block of PaiNN."""

    def __init__(
        self,
        hidden_channels: int,
        num_rbf: int,
        *,
        residual_scale: float = 1.0,
        tanh_message_scale: float | None = None,
        clip_scalar_msg_value: float | None = None,
        clip_vector_msg_norm: float | None = None,
        filter_gain: float = 1.0,
        enable_debug_stats: bool = False,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.residual_scale = float(residual_scale)
        self.tanh_message_scale = tanh_message_scale
        self.clip_scalar_msg_value = clip_scalar_msg_value
        self.clip_vector_msg_norm = clip_vector_msg_norm
        self.filter_gain = float(filter_gain)
        self.enable_debug_stats = enable_debug_stats
        self._last_debug: dict[str, float] | None = None
        self.inter_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 3),
            nn.SiLU(),
            nn.Linear(hidden_channels * 3, hidden_channels * 3),
        )
        self.filter_network = nn.Sequential(
            nn.Linear(num_rbf, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels * 3),
        )

    def forward(
        self,
        q: torch.Tensor,
        mu: torch.Tensor,
        edge_index: torch.Tensor,
        rbf: torch.Tensor,
        unit_vectors: torch.Tensor,
        cutoff_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_nodes = q.size(0)
        source = edge_index[1]
        target = edge_index[0]

        filters = self.filter_network(rbf) * cutoff_values.unsqueeze(-1)
        if self.filter_gain != 1.0:
            filters = filters * self.filter_gain
        filter_q, filter_r, filter_mu = filters.chunk(3, dim=-1)

        x = self.inter_mlp(q)
        x_q, x_r, x_mu = x.chunk(3, dim=-1)

        x_q_src = x_q[source] * filter_q
        if self.tanh_message_scale is not None:
            s = self.tanh_message_scale
            x_q_src = torch.tanh(x_q_src / s) * s
        scalar_msg = scatter(x_q_src, target, dim=0, dim_size=num_nodes, reduce="sum")

        x_r_src = x_r[source] * filter_r
        x_mu_src = x_mu[source] * filter_mu
        if self.tanh_message_scale is not None:
            s = self.tanh_message_scale
            x_r_src = torch.tanh(x_r_src / s) * s
            x_mu_src = torch.tanh(x_mu_src / s) * s
        mu_src = mu[source]

        vec_new = unit_vectors.unsqueeze(-1) * x_r_src.unsqueeze(-2)
        vec_propagated = mu_src * x_mu_src.unsqueeze(-2)
        vector_msg = scatter(
            vec_new + vec_propagated,
            target,
            dim=0,
            dim_size=num_nodes,
            reduce="sum",
        )

        # Degree normalization (mean aggregation)
        deg = torch.bincount(target, minlength=num_nodes).clamp(min=1)
        scalar_msg = scalar_msg / deg.view(-1, 1)
        vector_msg = vector_msg / deg.view(-1, 1, 1)

        # Optional clipping of aggregated messages
        if self.clip_scalar_msg_value is not None:
            c = self.clip_scalar_msg_value
            scalar_msg = torch.clamp(scalar_msg, min=-c, max=c)
        if self.clip_vector_msg_norm is not None:
            c = self.clip_vector_msg_norm
            vnorm = torch.sqrt((vector_msg**2).sum(dim=1) + 1e-12)
            scale = torch.clamp(c / (vnorm + 1e-12), max=1.0)
            vector_msg = vector_msg * scale.unsqueeze(1)

        # Residual scaling to damp updates
        scalar_msg = self.residual_scale * scalar_msg
        vector_msg = self.residual_scale * vector_msg

        q = q + scalar_msg
        mu = mu + vector_msg

        if self.enable_debug_stats:
            with torch.no_grad():
                stats = {}
                stats["scalar_msg_max"] = float(torch.max(torch.abs(scalar_msg)).detach().cpu())
                vnorm = torch.sqrt((vector_msg**2).sum(dim=1))
                stats["vector_msg_norm_max"] = float(torch.max(vnorm).detach().cpu())
                stats["deg_max"] = int(deg.max().item())
                stats["x_q_src_max"] = float(torch.max(torch.abs(x_q_src)).detach().cpu())
                stats["x_r_src_max"] = float(torch.max(torch.abs(x_r_src)).detach().cpu())
                stats["x_mu_src_max"] = float(torch.max(torch.abs(x_mu_src)).detach().cpu())
                any_nan = (
                    torch.isnan(q).any()
                    or torch.isnan(mu).any()
                    or torch.isinf(q).any()
                    or torch.isinf(mu).any()
                )
                stats["nan_or_inf"] = bool(any_nan)
                self._last_debug = stats
        return q, mu


class PaiNNMixing(nn.Module):
    """Implements the equivariant mixing block."""

    def __init__(
        self,
        hidden_channels: int,
        *,
        residual_scale: float = 1.0,
        tanh_mixing_scale: float | None = None,
        clip_mu_norm: float | None = None,
        clip_q_value: float | None = None,
        enable_debug_stats: bool = False,
    ):
        super().__init__()
        self.vec_linear = EquivariantLinear(hidden_channels, hidden_channels * 2)
        self.scalar_mlp = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels * 3),
            nn.SiLU(),
            nn.Linear(hidden_channels * 3, hidden_channels * 3),
        )
        self.residual_scale = float(residual_scale)
        self.tanh_mixing_scale = tanh_mixing_scale
        self.clip_mu_norm = clip_mu_norm
        self.clip_q_value = clip_q_value
        self.enable_debug_stats = enable_debug_stats
        self._last_debug: dict[str, float] | None = None

    def forward(self, q: torch.Tensor, mu: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # noqa: D401
        mu_cat = self.vec_linear(mu)
        mu_v, mu_w = mu_cat.chunk(2, dim=-1)

        mu_v_norm = torch.sqrt((mu_v**2).sum(dim=1) + 1e-8)
        scalar_input = torch.cat([q, mu_v_norm], dim=-1)
        delta = self.scalar_mlp(scalar_input)
        dq, dmu_scale, dqmu = delta.chunk(3, dim=-1)
        if self.tanh_mixing_scale is not None:
            s = self.tanh_mixing_scale
            dq = torch.tanh(dq / s) * s
            dmu_scale = torch.tanh(dmu_scale / s) * s
            dqmu = torch.tanh(dqmu / s) * s

        inner = (mu_v * mu_w).sum(dim=1)
        dq_total = dq + dqmu * inner
        dmu_total = mu_w * dmu_scale.unsqueeze(1)

        # Residual scaling
        q = q + self.residual_scale * dq_total
        mu = mu + self.residual_scale * dmu_total

        # Optional clipping of states
        if self.clip_q_value is not None:
            c = self.clip_q_value
            q = torch.clamp(q, min=-c, max=c)
        if self.clip_mu_norm is not None:
            c = self.clip_mu_norm
            mu_norm = torch.sqrt((mu**2).sum(dim=1) + 1e-8)
            scale = torch.clamp(c / (mu_norm + 1e-8), max=1.0)
            mu = mu * scale.unsqueeze(1)

        if self.enable_debug_stats:
            with torch.no_grad():
                stats = {}
                stats["mu_norm_max"] = float(torch.max(torch.sqrt((mu**2).sum(dim=1))).detach().cpu())
                stats["dq_max"] = float(torch.max(torch.abs(dq_total)).detach().cpu())
                stats["dmu_norm_max"] = float(torch.max(torch.sqrt((dmu_total**2).sum(dim=1))).detach().cpu())
                self._last_debug = stats
        return q, mu


class PaiNNBlock(nn.Module):
    """Full interaction + mixing block."""

    def __init__(self, hidden_channels: int, num_rbf: int, *, inter_kwargs: dict | None = None, mix_kwargs: dict | None = None):
        super().__init__()
        inter_kwargs = inter_kwargs or {}
        mix_kwargs = mix_kwargs or {}
        self.interaction = PaiNNInteraction(hidden_channels, num_rbf, **inter_kwargs)
        self.mixing = PaiNNMixing(hidden_channels, **mix_kwargs)
        self._last_debug: dict[str, float] | None = None

    def forward(
        self,
        q: torch.Tensor,
        mu: torch.Tensor,
        edge_index: torch.Tensor,
        rbf: torch.Tensor,
        unit_vectors: torch.Tensor,
        cutoff_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q, mu = self.interaction(q, mu, edge_index, rbf, unit_vectors, cutoff_values)
        q, mu = self.mixing(q, mu)
        if self.interaction.enable_debug_stats or self.mixing.enable_debug_stats:
            self._last_debug = {}
            if self.interaction._last_debug:
                for k, v in self.interaction._last_debug.items():
                    self._last_debug[f"inter.{k}"] = v
            if self.mixing._last_debug:
                for k, v in self.mixing._last_debug.items():
                    self._last_debug[f"mix.{k}"] = v
        return q, mu


class PaiNNReadout(nn.Module):
    """Vector readout head gated by scalar context."""

    def __init__(self, hidden_channels: int, vector_outputs: int = 1):
        super().__init__()
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.vector_linear = EquivariantLinear(hidden_channels, hidden_channels)
        self.out_linear = EquivariantLinear(hidden_channels, vector_outputs)

    def forward(self, q: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:  # noqa: D401
        gate = self.gate_mlp(q).unsqueeze(1)
        mu_gated = mu * gate
        mu_proj = self.vector_linear(mu_gated)
        out = self.out_linear(mu_proj)
        return out  # (N, 3, vector_outputs)


class PaiNN(nn.Module):
    """PaiNN model adapted for AtomicGraph inputs.

    The network predicts displacements (pos_dt) and updated velocities (vel) per
    node. The ``targets`` argument controls the order of concatenation.
    """

    def __init__(
        self,
        hidden_features: int = 128,
        num_layers: int = 6,
        num_rbf: int = 64,
        cutoff: float = 10.0,
        targets: Sequence[str] | None = None,
        use_velocity_input: bool = True,
        include_velocity_norm: bool = True,
        residual_scale_interaction: float = 1.0,
        residual_scale_mixing: float = 1.0,
        tanh_message_scale: float | None = None,
        tanh_mixing_scale: float | None = None,
        clip_scalar_msg_value: float | None = None,
        clip_vector_msg_norm: float | None = None,
        clip_q_value: float | None = None,
        clip_mu_norm: float | None = None,
        filter_gain: float = 1.0,
        enable_debug_stats: bool = False,
    ):
        super().__init__()
        self.hidden_features = hidden_features
        self.num_layers = num_layers
        self.num_rbf = num_rbf
        self.cutoff = cutoff
        self.targets = tuple(targets or ("pos_dt", "vel"))
        self.use_velocity_input = use_velocity_input
        self.include_velocity_norm = include_velocity_norm
        self.enable_debug_stats = enable_debug_stats
        self._debug_stats_current_pass: list[dict[str, float]] = []

        scalar_inputs = 1  # mass
        if include_velocity_norm:
            scalar_inputs += 1

        self.scalar_embedding = nn.Sequential(
            nn.Linear(scalar_inputs, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, hidden_features),
        )

        if use_velocity_input:
            self.vector_gate = nn.Sequential(
                nn.Linear(scalar_inputs, hidden_features),
                nn.SiLU(),
                nn.Linear(hidden_features, hidden_features),
            )
        else:
            self.vector_gate = None

        self.rbf = GaussianRBF(num_rbf, cutoff)
        self.cutoff_fn = CosineCutoff(cutoff)
        inter_kwargs = dict(
            residual_scale=residual_scale_interaction,
            tanh_message_scale=tanh_message_scale,
            clip_scalar_msg_value=clip_scalar_msg_value,
            clip_vector_msg_norm=clip_vector_msg_norm,
            filter_gain=filter_gain,
            enable_debug_stats=enable_debug_stats,
        )
        mix_kwargs = dict(
            residual_scale=residual_scale_mixing,
            tanh_mixing_scale=tanh_mixing_scale,
            clip_mu_norm=clip_mu_norm,
            clip_q_value=clip_q_value,
            enable_debug_stats=enable_debug_stats,
        )
        self.blocks = nn.ModuleList(
            PaiNNBlock(hidden_features, num_rbf, inter_kwargs=inter_kwargs, mix_kwargs=mix_kwargs)
            for _ in range(num_layers)
        )

        self.pos_head = PaiNNReadout(hidden_features, vector_outputs=1)
        self.vel_head = PaiNNReadout(hidden_features, vector_outputs=1)

    def _prepare_scalar_inputs(self, graph: AtomicGraph) -> torch.Tensor:
        features = [graph.__getattr__(ATOMIC_MASSES_KEY)]
        if self.include_velocity_norm:
            vel_norm = torch.linalg.norm(graph.__getattr__(VELOCITIES_KEY), dim=-1, keepdim=True)
            features.append(vel_norm)
        return torch.cat(features, dim=-1)

    @staticmethod
    def _unit_vectors(edge_vectors: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        distances = torch.linalg.norm(edge_vectors, dim=-1, keepdim=True)
        safe_denominator = torch.where(distances > eps, distances, torch.ones_like(distances))
        unit_vec = edge_vectors / safe_denominator
        unit_vec = torch.where(distances > eps, unit_vec, torch.zeros_like(unit_vec))
        return unit_vec

    def forward(self, graph: AtomicGraph, *, return_components: bool = False):  # noqa: D401
        # Positions are stored under the standard PyG name `pos`
        pos = graph.pos
        vel = graph.__getattr__(VELOCITIES_KEY)
        edge_index = graph.edge_index
        if edge_index is None:
            raise ValueError("edge_index must be provided by the dataloader for PaiNN.")

        scalar_inputs = self._prepare_scalar_inputs(graph)
        q = self.scalar_embedding(scalar_inputs)

        if self.use_velocity_input:
            velocity_scale = self.vector_gate(scalar_inputs)
            mu = vel.unsqueeze(-1) * velocity_scale.unsqueeze(1)
        else:
            mu = torch.zeros(pos.size(0), 3, self.hidden_features, device=pos.device, dtype=pos.dtype)

        row, col = edge_index
        edge_vectors = getattr(graph, EDGE_VECTORS_KEY, None)
        if edge_vectors is None:
            shifts = getattr(graph, "shifts", None)
            edge_vectors = pos[col] - pos[row]
            if shifts is not None:
                edge_vectors = edge_vectors + shifts

        distances = getattr(graph, EDGE_LENGTHS_KEY, None)
        if distances is None:
            distances = torch.linalg.norm(edge_vectors, dim=-1)
        else:
            distances = distances.view(-1)

        unit_vectors = self._unit_vectors(edge_vectors)
        rbf = self.rbf(distances)
        cutoff_values = self.cutoff_fn(distances)

        self._debug_stats_current_pass = []
        for li, block in enumerate(self.blocks):
            q, mu = block(q, mu, edge_index, rbf, unit_vectors, cutoff_values)
            if self.enable_debug_stats and block._last_debug is not None:
                self._debug_stats_current_pass.append({f"L{li}.{k}": v for k, v in block._last_debug.items()})

        pos_delta = self.pos_head(q, mu).squeeze(-1)
        vel_delta = self.vel_head(q, mu).squeeze(-1)
        vel_pred = vel + vel_delta

        outputs_map: Dict[str, torch.Tensor] = {"pos_dt": pos_delta, "vel": vel_pred}
        concatenated = torch.cat([outputs_map[target] for target in self.targets], dim=-1)

        if return_components:
            return concatenated, outputs_map
        return concatenated

    def get_and_reset_debug_stats(self) -> list[dict[str, float]]:
        stats = self._debug_stats_current_pass
        self._debug_stats_current_pass = []
        return stats


class PaiNNModel(AbstractModel):
    """Wrapper exposing PaiNN inside the TrajCast training pipeline."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_field = self.config.get("target_field", "target")
        self._build_model()

    def _build_model(self):  # noqa: D401
        cfg = self.config

        painn_targets = []
        for field in self._predicted_fields:
            if field == DISPLACEMENTS_KEY:
                painn_targets.append("pos_dt")
            elif field == UPDATE_VELOCITIES_KEY:
                painn_targets.append("vel")
            else:
                raise KeyError(f"Predicted field '{field}' is not supported by PaiNN.")

        defaults = {
            "hidden_features": 128,
            "num_layers": 6,
            "num_rbf": 64,
            "cutoff": cfg.get("edge_cutoff", cfg.get("cutoff", 4.0)),
            "use_velocity_input": True,
            "include_velocity_norm": True,
            "residual_scale_interaction": 1.0,
            "residual_scale_mixing": 1.0,
            "filter_gain": 1.0,
            "enable_debug_stats": False,
        }

        merged = {**defaults, **cfg}
        self.config = merged

        self.painn = PaiNN(
            hidden_features=merged["hidden_features"],
            num_layers=merged["num_layers"],
            num_rbf=merged["num_rbf"],
            cutoff=merged["cutoff"],
            targets=painn_targets,
            use_velocity_input=merged["use_velocity_input"],
            include_velocity_norm=merged["include_velocity_norm"],
            residual_scale_interaction=merged.get("residual_scale_interaction", 1.0),
            residual_scale_mixing=merged.get("residual_scale_mixing", 1.0),
            tanh_message_scale=merged.get("tanh_message_scale"),
            tanh_mixing_scale=merged.get("tanh_mixing_scale"),
            clip_scalar_msg_value=merged.get("clip_scalar_msg_value"),
            clip_vector_msg_norm=merged.get("clip_vector_msg_norm"),
            clip_q_value=merged.get("clip_q_value"),
            clip_mu_norm=merged.get("clip_mu_norm"),
            filter_gain=merged.get("filter_gain", 1.0),
            enable_debug_stats=merged.get("enable_debug_stats", False),
        )

        return self.painn

    def forward(self, data: AtomicGraph) -> AtomicGraph:  # noqa: D401
        data.compute_edge_vectors()

        predictions, components = self.painn(data, return_components=True)

        outputs = []
        for field in self._predicted_fields:
            if field == DISPLACEMENTS_KEY:
                value = components["pos_dt"]
            elif field == UPDATE_VELOCITIES_KEY:
                value = components["vel"]
            else:
                raise KeyError(f"Predicted field '{field}' is not supported by PaiNN.")

            data.__setattr__(field, value)
            outputs.append(value)

        data.__setattr__(self.target_field, torch.cat(outputs, dim=1))

        return data
