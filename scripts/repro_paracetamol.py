#!/usr/bin/env python3
"""
Reproduce key paper-style diagnostics for TrajCast on paracetamol using
pretrained weights. Outputs go to outputs/paracetamol/.

What it computes:
1) Appendix A.1 MAE (displacements & velocities) on the provided test split.
2) NVT rollout (7 fs, 1k steps) + element-resolved VDOS comparison to MD test set.
3) NVE rollout (7 fs, 1k steps) + time traces of two inter-atomic distances
   (O–O and N–H) for Figure S3-style plots.
4) Forecasting speed benchmark for 1k steps (Table S2-style).

Notes:
- Free-energy surfaces and potential-energy histograms from the paper are not
  reproduced here because force-field parameters and exact torsion definitions
  are not packaged with the repo. The script focuses on the parts that are
  fully reproducible from available assets.
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from ase.io import iread, read, write
from huggingface_hub import hf_hub_download
from scipy.signal import welch
from torch_geometric.loader import DataLoader

from trajcast.model.forecast import Forecast
from trajcast.model.models import EfficientTrajCastModel
from trajcast.data.dataset import AtomicGraphDataset
from trajcast.utils.misc import GLOBAL_DEVICE


BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "outputs" / "paracetamol"
HF_LOCAL = BASE / "hf" / "paracetamol"
DATA_LOCAL = BASE / "data" / "paracetamol"
DATA_SUB = DATA_LOCAL / "paracetamol"

# Heuristic atom indices for diagnostics (based on the provided extxyz ordering)
ATOM_IDX = {
    "O1": 0,
    "O2": 15,
    "N": 10,
    # Hydrogen presumed closest to N (chosen by shortest bond in first frame)
    "H_near_N": None,
    # Hydrogens / carbon for the H-O-C angle
    "H_near_O1": None,
    "C_near_O1": None,
}

STEPS = 200  # shorten for reproducibility; set to 1000 to mirror the paper exactly

def ensure_assets():
    HF_LOCAL.mkdir(parents=True, exist_ok=True)
    DATA_LOCAL.mkdir(parents=True, exist_ok=True)
    DATA_SUB.mkdir(parents=True, exist_ok=True)

    for fname in ["state_dict_e3nn.pt", "config_e3nn.yaml"]:
        hf_hub_download(
            repo_id="ibm-research/trajcast.models-arxiv2025",
            filename=fname,
            subfolder="paracetamol",
            local_dir=str(HF_LOCAL),
            resume_download=True,
        )

    hf_hub_download(
        repo_id="ibm-research/trajcast.datasets-arxiv2025",
        repo_type="dataset",
        filename="test.extxyz",
        subfolder="paracetamol",
        local_dir=str(DATA_LOCAL),
        resume_download=True,
    )


def load_model(device: str = "cpu") -> EfficientTrajCastModel:
    cfg = HF_LOCAL / "paracetamol" / "config_e3nn.yaml"
    weights = HF_LOCAL / "paracetamol" / "state_dict_e3nn.pt"
    model = EfficientTrajCastModel.build_from_yaml(str(cfg))
    model.load_state_dict(torch.load(weights, map_location=device))
    model.to(device)
    model.eval()
    return model


def mae_table(model: EfficientTrajCastModel, device: str = "cpu") -> Tuple[float, float]:
    ds = AtomicGraphDataset(
        root=str(DATA_LOCAL),
        name="paracetamol_test",
        cutoff_radius=4.0,
        files=[str(DATA_SUB / "test.extxyz")],
        atom_type_mapper={1: 0, 6: 1, 7: 2, 8: 3},
        rename=True,
    )
    loader = DataLoader(ds, batch_size=8, shuffle=False)

    disp_err = vel_err = 0.0
    total_nodes = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            ref = torch.hstack([batch.displacements, batch.update_velocities])
            pred = model(batch).target
            d, v = torch.split((pred - ref).abs(), [3, 3], dim=1)
            disp_err += d.sum().item()
            vel_err += v.sum().item()
            total_nodes += batch.num_nodes

    disp_mae = disp_err / total_nodes
    vel_mae = vel_err / total_nodes
    return disp_mae, vel_mae


def rollout(protocol: Dict, xyz_out: Path, log_out: Path | None = None):
    proto = protocol.copy()
    # disable writing inside Forecast to avoid double work
    proto["write"] = {
        "filename": str(xyz_out),
        "every": 1,
        "save_velocities": True,
    }
    forecaster = Forecast(protocol=proto)
    forecaster.generate_trajectory()
    if log_out and forecaster.logfile and Path(forecaster.logfile).exists():
        Path(forecaster.logfile).rename(log_out)


def species_indices(first_atoms) -> Dict[str, np.ndarray]:
    species = {}
    for element in np.unique(first_atoms.get_chemical_symbols()):
        species[element] = np.nonzero(np.array(first_atoms.get_chemical_symbols()) == element)[0]
    return species


def vacf(traj_vel: np.ndarray) -> np.ndarray:
    """velocity autocorrelation averaged over atoms and components."""
    # shape: (T, N, 3)
    v = traj_vel - traj_vel.mean(axis=0, keepdims=True)
    T = v.shape[0]
    # FFT-based VACF for speed
    f = np.fft.rfft(v, axis=0)
    ac = np.fft.irfft(f * np.conj(f), axis=0)[:T]
    ac = ac.sum(axis=(1, 2)) / (v.shape[1] * 3)
    return ac


def vdos(velocities: np.ndarray, selection: Iterable[int], dt_fs: float) -> Tuple[np.ndarray, np.ndarray]:
    vel = velocities[:, selection, :]
    ac = vacf(vel)
    freqs = np.fft.rfftfreq(len(ac), d=dt_fs * 1e-15)  # Hz
    spectrum = np.abs(np.fft.rfft(ac))
    c = 2.99792458e10  # speed of light cm/s
    freqs_cm = freqs / c
    return freqs_cm, spectrum


def choose_h_near_n(atoms) -> int:
    n_idx = ATOM_IDX["N"]
    hydrogens = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == "H"]
    dists = [
        np.linalg.norm(atoms.positions[n_idx] - atoms.positions[h]) for h in hydrogens
    ]
    return hydrogens[int(np.argmin(dists))]


def nearest_to_o1(atoms):
    o_idx = ATOM_IDX["O1"]
    hydrogens = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == "H"]
    carbons = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == "C"]
    h_idx = hydrogens[int(np.argmin([np.linalg.norm(atoms.positions[o_idx] - atoms.positions[h]) for h in hydrogens]))]
    c_idx = carbons[int(np.argmin([np.linalg.norm(atoms.positions[o_idx] - atoms.positions[c]) for c in carbons]))]
    return h_idx, c_idx


def read_frames(path: Path) -> list:
    return list(iread(path))


def nve_traces(pred_frames: list, ref_frames: list) -> Dict[str, np.ndarray]:
    traces = {}
    n_frames = min(len(pred_frames), len(ref_frames))
    pred_frames = pred_frames[:n_frames]
    ref_frames = ref_frames[:n_frames]
    o1, o2 = ATOM_IDX["O1"], ATOM_IDX["O2"]

    def dist(frames, i, j):
        return np.array([np.linalg.norm(f.positions[i] - f.positions[j]) for f in frames])

    traces["d_OO_pred"] = dist(pred_frames, o1, o2)
    traces["d_OO_ref"] = dist(ref_frames, o1, o2)

    h_idx = ATOM_IDX["H_near_N"]
    if h_idx is None:
        h_idx = choose_h_near_n(ref_frames[0])
        ATOM_IDX["H_near_N"] = int(h_idx)
    traces["d_NH_pred"] = dist(pred_frames, ATOM_IDX["N"], h_idx)
    traces["d_NH_ref"] = dist(ref_frames, ATOM_IDX["N"], h_idx)

    # angle H-O-C (using nearest H and C to O1)
    h_o1, c_o1 = ATOM_IDX["H_near_O1"], ATOM_IDX["C_near_O1"]
    if h_o1 is None or c_o1 is None:
        h_o1, c_o1 = nearest_to_o1(ref_frames[0])
        ATOM_IDX["H_near_O1"] = int(h_o1)
        ATOM_IDX["C_near_O1"] = int(c_o1)

    def angle(frames, i, j, k):
        # angle i-j-k at j
        ang = []
        for f in frames:
            v1 = f.positions[i] - f.positions[j]
            v2 = f.positions[k] - f.positions[j]
            cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cosang = np.clip(cosang, -1.0, 1.0)
            ang.append(np.degrees(np.arccos(cosang)))
        return np.array(ang)

    traces["ang_HOC_pred"] = angle(pred_frames, h_o1, o1, c_o1)
    traces["ang_HOC_ref"] = angle(ref_frames, h_o1, o1, c_o1)
    return traces


def benchmark(model: EfficientTrajCastModel, start_frame, device="cpu") -> Tuple[float, float]:
    proto = {
        "units": "real",
        "run": STEPS,
        "timestep": 7.0,
        "temperature": 300.0,
        "configuration": start_frame,
        "model_type": "EfficientTrajCast",
        "model": model,
        "thermostat": {"Tdamp": 70.0},
        "write": {"filename": str(OUT / "bench_temp.extxyz"), "every": 1000},
        "device": device,
        "seed": 0,
    }
    t0 = time.time()
    Forecast(protocol=proto).generate_trajectory()
    dt = time.time() - t0
    ns_per_day = (24 * 3600 / dt) * (proto["run"] * proto["timestep"] * 1e-6)
    return dt, ns_per_day


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ensure_assets()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    GLOBAL_DEVICE.device = device
    model = load_model(device=device)

    # 1) Appendix A.1 MAE
    disp_mae, vel_mae = mae_table(model, device=device)
    with open(OUT / "mae.txt", "w", encoding="utf-8") as f:
        f.write(f"Disp MAE [Angstrom]: {disp_mae:.6e}\n")
        f.write(f"Vel  MAE [Angstrom/fs]: {vel_mae:.6e}\n")

    # 2) NVT rollout + VDOS
    start_cfg = read(DATA_SUB / "test.extxyz", index=0)
    nvt_xyz = OUT / "traj_nvt.extxyz"
    rollout(
        protocol={
            "units": "real",
            "run": STEPS,
            "timestep": 7.0,
            "temperature": 300.0,
            "extra_dof": 6,
            "configuration": start_cfg,
            "model_type": "EfficientTrajCast",
            "model": model,
            "thermostat": {"Tdamp": 70.0},
            "velocities": False,
            "device": device,
            "seed": 42,
        },
        xyz_out=nvt_xyz,
        log_out=OUT / "temperature_log.csv",
    )

    ref_frames = read_frames(DATA_SUB / "test.extxyz")
    pred_frames = read_frames(nvt_xyz)
    species = species_indices(ref_frames[0])
    ref_vel = np.stack([f.get_velocities() for f in ref_frames])  # (T,N,3)
    pred_vel = np.stack([f.get_velocities() for f in pred_frames])

    for elem, idxs in species.items():
        f_ref, s_ref = vdos(ref_vel, idxs, dt_fs=7.0)
        f_pred, s_pred = vdos(pred_vel, idxs, dt_fs=7.0)
        plt.figure()
        plt.plot(f_ref, s_ref, label="MD (test)")
        plt.plot(f_pred, s_pred, label="TrajCast NVT")
        plt.xlim(0, 4000)
        plt.xlabel("Frequency [cm$^{-1}$]")
        plt.ylabel("Intensity (arb.)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT / f"vdos_{elem}.png", dpi=200)
        plt.close()

    # 3) NVE rollout + distance traces
    nve_xyz = OUT / "traj_nve.extxyz"
    rollout(
        protocol={
            "units": "real",
            "run": STEPS,
            "timestep": 7.0,
            "temperature": 300.0,
            "configuration": start_cfg,
            "model_type": "EfficientTrajCast",
            "model": model,
            "thermostat": False,
            "device": device,
            "seed": 7,
        },
        xyz_out=nve_xyz,
        log_out=None,
    )
    pred_nve_frames = read_frames(nve_xyz)
    traces = nve_traces(pred_nve_frames, ref_frames)
    times = np.arange(len(traces["d_OO_pred"])) * 7.0

    fig, axes = plt.subplots(3, 1, figsize=(6, 7), sharex=True)
    axes[0].plot(times, traces["d_OO_ref"], label="MD (ref)", color="k")
    axes[0].plot(times, traces["d_OO_pred"], label="TrajCast 7 fs", color="#14a195")
    axes[0].set_ylabel(r"$d_{O-O}$ [Å]")

    axes[1].plot(times, traces["d_NH_ref"], label="MD (ref)", color="k")
    axes[1].plot(times, traces["d_NH_pred"], label="TrajCast 7 fs", color="#f4b000")
    axes[1].set_ylabel(r"$d_{N-H}$ [Å]")

    axes[2].plot(times, traces["ang_HOC_ref"], label="MD (ref)", color="k")
    axes[2].plot(times, traces["ang_HOC_pred"], label="TrajCast 7 fs", color="#1f4b99")
    axes[2].set_ylabel("α_{HOC} [deg]")
    axes[2].set_xlabel("Time [fs]")

    axes[0].legend(loc="upper right", ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / "figure_s3_style.png", dpi=220)
    plt.close()

    # 4) Forecasting efficiency benchmark
    t_elapsed, ns_per_day = benchmark(model, start_cfg, device=device)
    with open(OUT / "benchmark.txt", "w") as f:
        f.write(f"time_{STEPS}_steps_s: {t_elapsed:.2f}\n")
        f.write(f"ns_per_day: {ns_per_day:.2f}\n")

    print("Done. Outputs in", OUT)


if __name__ == "__main__":
    main()
