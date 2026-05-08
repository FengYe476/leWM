#!/usr/bin/env python3
"""DINO-WM PushT final-pool audit for endpoint-planning decoupling.

This script is intentionally an integration shim: it imports DINO-WM at runtime,
instruments the CEM loop from outside the cloned repository, and writes audit
artifacts under this repo. It does not patch or edit DINO-WM source files.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_DINO_REPO = WORKSPACE_ROOT / "dino_wm"
DEFAULT_CKPT_BASE = DEFAULT_DINO_REPO / "checkpoints"
DEFAULT_PAIRS_PATH = PROJECT_ROOT / "results" / "phase1" / "track_a_pairs.json"
DEFAULT_RERANK_PATH = PROJECT_ROOT / "results" / "phase2" / "protocol_match" / "pusht_rerank_only.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "revision" / "dino_wm_pool_audit.json"
DEFAULT_POOL_DIR = PROJECT_ROOT / "results" / "revision" / "dino_wm_pusht_pools"

STAGE_A_PAIR_SELECTION_COUNTS = {
    "invisible_quadrant": 8,
    "ordinary": 12,
    "latent_favorable": 5,
    "v1_favorable": 5,
}
STAGE_A_SUBSET_ORDER = ("invisible_quadrant", "ordinary", "latent_favorable", "v1_favorable")
ALL_MEMBERSHIP_ORDER = (
    "invisible_quadrant",
    "sign_reversal",
    "latent_favorable",
    "v1_favorable",
    "ordinary",
)

BLOCK_SUCCESS_THRESHOLD_PX = 20.0
ANGLE_SUCCESS_THRESHOLD_RAD = math.pi / 9.0
HINGE_ALPHA = BLOCK_SUCCESS_THRESHOLD_PX / ANGLE_SUCCESS_THRESHOLD_RAD


def parse_int_list(raw: str) -> tuple[int, ...]:
    values = tuple(int(chunk.strip()) for chunk in str(raw).split(",") if chunk.strip())
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated integer")
    return tuple(dict.fromkeys(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dino-repo", type=Path, default=DEFAULT_DINO_REPO)
    parser.add_argument("--ckpt-base-path", type=Path, default=DEFAULT_CKPT_BASE)
    parser.add_argument("--model-name", default="pusht")
    parser.add_argument("--model-epoch", default="latest")
    parser.add_argument("--pairs-path", type=Path, default=DEFAULT_PAIRS_PATH)
    parser.add_argument("--rerank-path", type=Path, default=DEFAULT_RERANK_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pool-dir", type=Path, default=DEFAULT_POOL_DIR)
    parser.add_argument("--pair-ids", type=parse_int_list, default=None)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", type=parse_int_list, default=(0,))
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--opt-steps", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--var-scale", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--objective-mode", choices=("last", "all"), default="last")
    parser.add_argument("--objective-base", type=float, default=2.0)
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument(
        "--endpoint-source",
        choices=("first_iter_real_z", "final_pool_real_z", "none"),
        default="first_iter_real_z",
        help=(
            "R_endpoint source. first_iter_real_z scores the broad first CEM sample "
            "pool with real terminal DINO-WM embeddings; final_pool_real_z reuses "
            "the final pool's real terminal embeddings."
        ),
    )
    parser.add_argument(
        "--render-goal-mode",
        choices=("dino_default", "track_goal"),
        default="dino_default",
        help=(
            "Whether DINO PushT renders its default target overlay or the Track A "
            "goal pose overlay when creating observations. dino_default matches "
            "the upstream DINO-WM planning path."
        ),
    )
    parser.add_argument("--save-endpoint-pools", action="store_true")
    parser.add_argument("--save-float16-latents", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--list-pairs", action="store_true")
    parser.add_argument(
        "--patch-device-compat",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Install a runtime monkey patch for DINO-WM ViT attention masks.",
    )
    return parser.parse_args()


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_git_commit(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def clean_float(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if "torch" in sys.modules:
        torch = sys.modules["torch"]
        if torch.is_tensor(value):
            return jsonable(value.detach().cpu().tolist())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(jsonable(payload), indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    xr = rankdata(x)
    yr = rankdata(y)
    value = float(np.corrcoef(xr, yr)[0, 1])
    return clean_float(value)


def scalar_summary(values: list[float | int | bool | None]) -> dict[str, Any]:
    arr = np.asarray(
        [float(value) for value in values if value is not None and math.isfinite(float(value))],
        dtype=np.float64,
    )
    return {
        "mean": clean_float(float(arr.mean())) if len(arr) else None,
        "std": clean_float(float(arr.std(ddof=1))) if len(arr) > 1 else None,
        "min": clean_float(float(arr.min())) if len(arr) else None,
        "max": clean_float(float(arr.max())) if len(arr) else None,
        "n": int(len(arr)),
        "ddof": 1,
    }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_anchor_definitions(path: Path) -> dict[str, Any]:
    data = load_json(path)
    anchors = data.get("metadata", {}).get("anchor_definitions")
    if not isinstance(anchors, dict):
        raise RuntimeError(f"Missing metadata.anchor_definitions in {path}")
    return anchors


def membership_map(anchor_definitions: dict[str, Any]) -> dict[int, list[str]]:
    memberships: dict[int, list[str]] = {}
    for name in ALL_MEMBERSHIP_ORDER:
        for pair_id in anchor_definitions.get(name, {}).get("pair_ids", []):
            memberships.setdefault(int(pair_id), []).append(name)
    return memberships


def select_mppi_30_pairs(
    *,
    pairs_path: Path,
    rerank_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[int, list[str]], dict[str, Any]]:
    pairs_data = load_json(pairs_path)
    all_pairs = sorted(pairs_data["pairs"], key=lambda pair: int(pair["pair_id"]))
    by_id = {int(pair["pair_id"]): pair for pair in all_pairs}
    anchor_definitions = load_anchor_definitions(rerank_path)
    memberships = membership_map(anchor_definitions)

    selected: list[dict[str, Any]] = []
    used: set[int] = set()
    for subset in STAGE_A_SUBSET_ORDER:
        pair_ids = [int(pair_id) for pair_id in anchor_definitions[subset]["pair_ids"]]
        chosen: list[int] = []
        for pair_id in pair_ids:
            if pair_id in used:
                continue
            chosen.append(pair_id)
            if len(chosen) == int(STAGE_A_PAIR_SELECTION_COUNTS[subset]):
                break
        if len(chosen) != int(STAGE_A_PAIR_SELECTION_COUNTS[subset]):
            raise RuntimeError(
                f"Could not select {STAGE_A_PAIR_SELECTION_COUNTS[subset]} pairs for {subset}; "
                f"got {len(chosen)}"
            )
        for pair_id in chosen:
            if pair_id not in by_id:
                raise RuntimeError(f"Selected pair_id={pair_id} missing from {pairs_path}")
            used.add(pair_id)
            pair = dict(by_id[pair_id])
            pair["primary_subset"] = subset
            pair["subset_memberships"] = memberships.get(pair_id, [])
            selected.append(pair)

    if len(selected) != 30 or len({int(pair["pair_id"]) for pair in selected}) != 30:
        raise RuntimeError("Stage A pair selection must contain 30 unique pairs")
    return pairs_data, selected, memberships, anchor_definitions


def validate_pair_offsets(pairs: list[dict[str, Any]], *, expected_offset: int) -> None:
    mismatches = [
        {
            "pair_id": int(pair["pair_id"]),
            "start_row": int(pair["start_row"]),
            "goal_row": int(pair["goal_row"]),
            "delta": int(pair["goal_row"]) - int(pair["start_row"]),
        }
        for pair in pairs
        if int(pair["goal_row"]) - int(pair["start_row"]) != int(expected_offset)
    ]
    if mismatches:
        raise ValueError(f"Pair offset mismatch. Expected {expected_offset}; examples={mismatches[:5]}")


class H5Rows:
    """Minimal Track A HDF5 row reader, avoiding stable-worldmodel imports."""

    def __init__(self, path: Path):
        import h5py

        self.path = Path(path)
        self.handle = h5py.File(self.path, "r")

    def close(self) -> None:
        self.handle.close()

    def _dataset(self, key: str):
        if key in self.handle:
            return self.handle[key]
        for group_name in ("data", "observations"):
            if group_name in self.handle and key in self.handle[group_name]:
                return self.handle[group_name][key]
        raise KeyError(f"{self.path} does not contain dataset {key!r}")

    def row(self, row_idx: int, keys: tuple[str, ...] = ("state",)) -> dict[str, np.ndarray]:
        return {key: np.asarray(self._dataset(key)[int(row_idx)]) for key in keys}


def angular_distance(a: float, b: float) -> float:
    diff = (float(a) - float(b) + math.pi) % (2 * math.pi) - math.pi
    return abs(diff)


def block_pose_metrics(state: np.ndarray, goal_state: np.ndarray) -> dict[str, Any]:
    state = np.asarray(state, dtype=np.float64)
    goal_state = np.asarray(goal_state, dtype=np.float64)
    block_pos_dist = float(np.linalg.norm(state[2:4] - goal_state[2:4]))
    angle_dist = angular_distance(float(state[4]), float(goal_state[4]))
    success = block_pos_dist < BLOCK_SUCCESS_THRESHOLD_PX and angle_dist < ANGLE_SUCCESS_THRESHOLD_RAD
    c_real_state = block_pos_dist + angle_dist
    v1_cost = max(block_pos_dist - BLOCK_SUCCESS_THRESHOLD_PX, 0.0) + HINGE_ALPHA * max(
        angle_dist - ANGLE_SUCCESS_THRESHOLD_RAD,
        0.0,
    )
    return {
        "block_pos_dist": clean_float(block_pos_dist),
        "angle_dist": clean_float(angle_dist),
        "c_real_state": clean_float(c_real_state),
        "v1_hinge_cost": clean_float(v1_cost),
        "success": bool(success),
    }


def as_dino_state(state: np.ndarray, *, with_velocity: bool) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    if with_velocity:
        if len(state) >= 7:
            return state[:7].astype(np.float32, copy=False)
        if len(state) == 5:
            return np.concatenate([state, np.zeros(2, dtype=np.float32)]).astype(np.float32)
        raise ValueError(f"Expected 5D or 7D PushT state for velocity env, got shape {state.shape}")
    if len(state) < 5:
        raise ValueError(f"Expected at least 5D PushT state, got shape {state.shape}")
    return state[:5].astype(np.float32, copy=False)


def configure_dino_imports(dino_repo: Path) -> None:
    dino_repo = dino_repo.expanduser().resolve()
    if not dino_repo.exists():
        raise FileNotFoundError(f"DINO-WM repo not found: {dino_repo}")
    if str(dino_repo) not in sys.path:
        sys.path.insert(0, str(dino_repo))


def patch_vit_attention_mask(device: str) -> bool:
    """Runtime-only patch for models/vit.py hardcoded `.to('cuda')` masks."""
    import torch
    from torch import nn

    import models.vit as vit

    if getattr(vit.Attention, "_dino_audit_device_patch", False):
        return False

    def patched_init(self, dim, heads=8, dim_head=64, dropout=0.0):
        nn.Module.__init__(self)
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )
        mask = vit.generate_mask_matrix(vit.NUM_PATCHES, vit.NUM_FRAMES)
        self.register_buffer("bias", mask.to(torch.device(device)), persistent=False)

    vit.Attention.__init__ = patched_init
    vit.Attention._dino_audit_device_patch = True
    return True


def resolve_torch_device(device: str):
    import torch

    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device}, but torch.cuda.is_available() is false")
    return torch_device


def load_dino_components(args: argparse.Namespace) -> dict[str, Any]:
    configure_dino_imports(args.dino_repo)

    import gym  # noqa: F401 - DINO-WM uses gym registration side effects.
    import hydra
    import torch
    from omegaconf import OmegaConf

    import env as dino_env  # noqa: F401
    from datasets import pusht_dset
    from plan import load_model
    from planning.objectives import create_objective_fn
    from preprocessor import Preprocessor

    device = resolve_torch_device(args.device)
    if args.patch_device_compat:
        patch_vit_attention_mask(str(device))

    model_path = args.ckpt_base_path / "outputs" / args.model_name
    hydra_path = model_path / "hydra.yaml"
    model_ckpt = model_path / "checkpoints" / f"model_{args.model_epoch}.pth"
    if not hydra_path.exists():
        raise FileNotFoundError(f"Missing DINO-WM training config: {hydra_path}")
    if not model_ckpt.exists():
        raise FileNotFoundError(f"Missing DINO-WM checkpoint: {model_ckpt}")

    model_cfg = OmegaConf.load(hydra_path)
    wm = load_model(model_ckpt, model_cfg, model_cfg.num_action_repeat, device=device)
    wm.eval()
    wm.requires_grad_(False)

    transform = hydra.utils.instantiate(model_cfg.env.dataset.transform)
    with_velocity = bool(model_cfg.env.kwargs.get("with_velocity", True))
    proprio_dim = 4 if with_velocity else 2
    state_dim = 7 if with_velocity else 5
    preprocessor = Preprocessor(
        action_mean=pusht_dset.ACTION_MEAN,
        action_std=pusht_dset.ACTION_STD,
        state_mean=pusht_dset.STATE_MEAN[:state_dim],
        state_std=pusht_dset.STATE_STD[:state_dim],
        proprio_mean=pusht_dset.PROPRIO_MEAN[:proprio_dim],
        proprio_std=pusht_dset.PROPRIO_STD[:proprio_dim],
        transform=transform,
    )
    objective_fn = create_objective_fn(alpha=args.alpha, base=args.objective_base, mode=args.objective_mode)

    return {
        "torch": torch,
        "gym": gym,
        "wm": wm,
        "model_cfg": model_cfg,
        "preprocessor": preprocessor,
        "objective_fn": objective_fn,
        "device": device,
        "model_path": model_path,
        "model_ckpt": model_ckpt,
        "with_velocity": with_velocity,
        "frameskip": int(model_cfg.frameskip),
    }


def make_env(components: dict[str, Any]):
    model_cfg = components["model_cfg"]
    gym = components["gym"]
    return gym.make(model_cfg.env.name, *model_cfg.env.args, **model_cfg.env.kwargs)


def set_render_goal_if_needed(env, goal_state: np.ndarray, *, mode: str) -> None:
    if mode != "track_goal":
        return
    pose = np.asarray(goal_state, dtype=np.float32)[2:5]
    unwrapped = getattr(env, "unwrapped", env)
    if hasattr(unwrapped, "set_task_goal"):
        unwrapped.set_task_goal(pose)
    elif hasattr(unwrapped, "goal_pose"):
        unwrapped.goal_pose = pose
    else:
        raise AttributeError("DINO PushT env does not expose a goal pose setter")


def prepare_obs_from_state(
    env,
    state: np.ndarray,
    *,
    seed: int,
    with_velocity: bool,
    render_goal_state: np.ndarray | None,
    render_goal_mode: str,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    dino_state = as_dino_state(state, with_velocity=with_velocity)
    env.seed(int(seed))
    env.reset_to_state = dino_state
    obs, actual_state = env.reset()
    env.reset_to_state = None
    if render_goal_state is not None:
        set_render_goal_if_needed(env, render_goal_state, mode=render_goal_mode)
        if render_goal_mode == "track_goal":
            obs["visual"] = env.render("rgb_array")
    batched = {
        "visual": np.expand_dims(np.expand_dims(np.asarray(obs["visual"], dtype=np.uint8), axis=0), axis=0),
        "proprio": np.expand_dims(np.expand_dims(np.asarray(obs["proprio"], dtype=np.float32), axis=0), axis=0),
    }
    return batched, np.asarray(actual_state, dtype=np.float32)


def rollout_raw_actions(
    env,
    *,
    initial_state: np.ndarray,
    goal_state: np.ndarray,
    raw_actions: np.ndarray,
    seed: int,
    with_velocity: bool,
    render_goal_mode: str,
) -> dict[str, Any]:
    dino_state = as_dino_state(initial_state, with_velocity=with_velocity)
    env.seed(int(seed))
    env.reset_to_state = dino_state
    obs, state = env.reset()
    env.reset_to_state = None
    set_render_goal_if_needed(env, goal_state, mode=render_goal_mode)
    if render_goal_mode == "track_goal":
        obs["visual"] = env.render("rgb_array")

    for action in np.asarray(raw_actions, dtype=np.float32):
        obs, _, _, info = env.step(np.asarray(action, dtype=np.float32))
        state = info.get("state", state)

    if render_goal_mode == "track_goal":
        obs["visual"] = env.render("rgb_array")
    terminal_state = np.asarray(state, dtype=np.float32)
    metrics = block_pose_metrics(terminal_state, as_dino_state(goal_state, with_velocity=with_velocity))
    return {
        "terminal_state": terminal_state,
        "terminal_obs": {
            "visual": np.asarray(obs["visual"], dtype=np.uint8),
            "proprio": np.asarray(obs["proprio"], dtype=np.float32),
        },
        "metrics": metrics,
    }


def move_to_device(dct: dict[str, Any], device: Any) -> dict[str, Any]:
    out = {}
    for key, value in dct.items():
        out[key] = value.to(device) if hasattr(value, "to") else value
    return out


def terminal_goal_dict(z_goal_single: dict[str, Any], *, n: int, device: Any) -> dict[str, Any]:
    return {
        key: value.to(device).unsqueeze(0).unsqueeze(0).repeat(n, 1, *([1] * value.ndim))
        for key, value in z_goal_single.items()
    }


def maybe_half(tensor: Any, *, enabled: bool) -> Any:
    if not enabled:
        return tensor
    return tensor.half() if hasattr(tensor, "half") else tensor


def blocked_to_raw_actions(blocked_actions: np.ndarray, *, preprocessor: Any, frameskip: int) -> np.ndarray:
    torch = sys.modules["torch"]
    blocked_actions = np.asarray(blocked_actions, dtype=np.float32)
    if blocked_actions.ndim != 3 or blocked_actions.shape[-1] % int(frameskip) != 0:
        raise ValueError(f"Unexpected blocked action shape: {blocked_actions.shape}")
    raw_action_dim = blocked_actions.shape[-1] // int(frameskip)
    normalized = torch.as_tensor(
        blocked_actions.reshape(-1, raw_action_dim),
        dtype=torch.float32,
    )
    raw = preprocessor.denormalize_actions(normalized).detach().cpu().numpy().astype(np.float32)
    return raw.reshape(blocked_actions.shape[0], blocked_actions.shape[1] * int(frameskip), raw_action_dim)


def run_cem_capture(
    *,
    wm: Any,
    preprocessor: Any,
    objective_fn: Any,
    obs_0: dict[str, np.ndarray],
    obs_g: dict[str, np.ndarray],
    device: Any,
    seed: int,
    num_samples: int,
    topk: int,
    opt_steps: int,
    horizon: int,
    action_dim: int,
    var_scale: float,
    capture_iters: tuple[int, ...],
    save_float16_latents: bool,
) -> dict[str, Any]:
    torch = sys.modules["torch"]
    from einops import repeat

    generator = torch.Generator(device=device).manual_seed(int(seed))
    trans_obs_0 = move_to_device(preprocessor.transform_obs(obs_0), device)
    trans_obs_g = move_to_device(preprocessor.transform_obs(obs_g), device)

    with torch.inference_mode():
        z_obs_g = wm.encode_obs(trans_obs_g)

    mu = torch.zeros(1, int(horizon), int(action_dim), dtype=torch.float32, device=device)
    sigma = float(var_scale) * torch.ones_like(mu)
    captures: dict[int, dict[str, Any]] = {}

    for iter_idx in range(1, int(opt_steps) + 1):
        cur_trans_obs_0 = {
            key: repeat(arr[0].unsqueeze(0), "1 ... -> n ...", n=int(num_samples))
            for key, arr in trans_obs_0.items()
        }
        cur_z_obs_g = {
            key: repeat(arr[0].unsqueeze(0), "1 ... -> n ...", n=int(num_samples))
            for key, arr in z_obs_g.items()
        }
        actions = torch.randn(
            int(num_samples),
            int(horizon),
            int(action_dim),
            generator=generator,
            device=device,
        )
        actions = actions * sigma[0] + mu[0]
        actions[0] = mu[0]

        with torch.inference_mode():
            z_obs_pred, _ = wm.rollout(obs_0=cur_trans_obs_0, act=actions)
            loss = objective_fn(z_obs_pred, cur_z_obs_g)

        topk_idx = torch.argsort(loss)[: int(topk)]
        elite_actions = actions[topk_idx]

        if iter_idx in capture_iters:
            z_pred_visual = z_obs_pred["visual"][:, -1].detach().cpu()
            z_pred_proprio = z_obs_pred["proprio"][:, -1].detach().cpu()
            z_goal_visual = z_obs_g["visual"][0, 0].detach().cpu()
            z_goal_proprio = z_obs_g["proprio"][0, 0].detach().cpu()
            captures[int(iter_idx)] = {
                "iter_idx": int(iter_idx),
                "blocked_actions": actions.detach().cpu(),
                "candidate_indices": torch.arange(int(num_samples), dtype=torch.int64),
                "default_costs": loss.detach().cpu().to(dtype=torch.float64),
                "z_pred": {
                    "visual": maybe_half(z_pred_visual, enabled=save_float16_latents),
                    "proprio": maybe_half(z_pred_proprio, enabled=save_float16_latents),
                },
                "z_goal": {
                    "visual": maybe_half(z_goal_visual, enabled=save_float16_latents),
                    "proprio": maybe_half(z_goal_proprio, enabled=save_float16_latents),
                },
                "top30_costs": loss[topk_idx].detach().cpu().to(dtype=torch.float64),
                "rank1_candidate_index": int(topk_idx[0].detach().cpu().item()),
                "mu_before_update": mu[0].detach().cpu(),
                "sigma_before_update": sigma[0].detach().cpu(),
            }

        mu[0] = elite_actions.mean(dim=0)
        sigma[0] = elite_actions.std(dim=0)

    return {
        "captures": captures,
        "planned_mean_after_final_update": mu[0].detach().cpu(),
        "planned_sigma_after_final_update": sigma[0].detach().cpu(),
    }


def encode_real_endpoint_costs(
    *,
    wm: Any,
    preprocessor: Any,
    objective_fn: Any,
    terminal_visuals: np.ndarray,
    terminal_proprios: np.ndarray,
    z_goal_single: dict[str, Any],
    device: Any,
    batch_size: int,
) -> np.ndarray:
    torch = sys.modules["torch"]
    costs: list[np.ndarray] = []
    n = int(terminal_visuals.shape[0])
    for start in range(0, n, int(batch_size)):
        end = min(start + int(batch_size), n)
        obs = {
            "visual": terminal_visuals[start:end, None, ...],
            "proprio": terminal_proprios[start:end, None, ...],
        }
        trans_obs = move_to_device(preprocessor.transform_obs(obs), device)
        with torch.inference_mode():
            z_real = wm.encode_obs(trans_obs)
            z_goal = terminal_goal_dict(z_goal_single, n=end - start, device=device)
            cost = objective_fn(z_real, z_goal)
        costs.append(cost.detach().cpu().numpy().astype(np.float64))
    return np.concatenate(costs, axis=0)


def score_pool(
    *,
    env: Any,
    raw_actions: np.ndarray,
    initial_state: np.ndarray,
    goal_state: np.ndarray,
    seed_base: int,
    with_velocity: bool,
    render_goal_mode: str,
    wm: Any,
    preprocessor: Any,
    objective_fn: Any,
    z_goal_single: dict[str, Any],
    device: Any,
    encode_batch_size: int,
) -> dict[str, Any]:
    raw_actions = np.asarray(raw_actions, dtype=np.float32)
    n = int(raw_actions.shape[0])
    terminal_states = []
    terminal_visuals = []
    terminal_proprios = []
    metrics = []
    for idx, candidate in enumerate(raw_actions):
        result = rollout_raw_actions(
            env,
            initial_state=initial_state,
            goal_state=goal_state,
            raw_actions=candidate,
            seed=int(seed_base) + int(idx),
            with_velocity=with_velocity,
            render_goal_mode=render_goal_mode,
        )
        terminal_states.append(result["terminal_state"])
        terminal_visuals.append(result["terminal_obs"]["visual"])
        terminal_proprios.append(result["terminal_obs"]["proprio"])
        metrics.append({"candidate_index": int(idx), "seed": int(seed_base) + int(idx), **result["metrics"]})

    terminal_visual_arr = np.asarray(terminal_visuals, dtype=np.uint8)
    terminal_proprio_arr = np.asarray(terminal_proprios, dtype=np.float32)
    c_real_z = encode_real_endpoint_costs(
        wm=wm,
        preprocessor=preprocessor,
        objective_fn=objective_fn,
        terminal_visuals=terminal_visual_arr,
        terminal_proprios=terminal_proprio_arr,
        z_goal_single=z_goal_single,
        device=device,
        batch_size=encode_batch_size,
    )

    torch = sys.modules["torch"]
    return {
        "terminal_states": torch.as_tensor(np.asarray(terminal_states, dtype=np.float32)),
        "c_real_z": torch.as_tensor(c_real_z, dtype=torch.float64),
        "v1_hinge_costs": torch.as_tensor(
            np.asarray([item["v1_hinge_cost"] for item in metrics], dtype=np.float64),
            dtype=torch.float64,
        ),
        "c_real_state": torch.as_tensor(
            np.asarray([item["c_real_state"] for item in metrics], dtype=np.float64),
            dtype=torch.float64,
        ),
        "block_pos_dist": torch.as_tensor(
            np.asarray([item["block_pos_dist"] for item in metrics], dtype=np.float64),
            dtype=torch.float64,
        ),
        "angle_dist": torch.as_tensor(
            np.asarray([item["angle_dist"] for item in metrics], dtype=np.float64),
            dtype=torch.float64,
        ),
        "success": torch.as_tensor(np.asarray([item["success"] for item in metrics], dtype=bool)),
        "candidate_metrics": metrics,
        "n_scored": n,
    }


def tensor_to_numpy(value: Any, dtype: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy().astype(dtype, copy=False)
    return np.asarray(value, dtype=dtype)


def pool_summary_record(
    *,
    pool: dict[str, Any],
    endpoint_record: dict[str, Any] | None,
    pool_path: Path,
    pair: dict[str, Any],
    seed: int,
    planned_action_metrics: dict[str, Any],
) -> dict[str, Any]:
    default_costs = tensor_to_numpy(pool["default_costs"], np.float64)
    c_real_state = tensor_to_numpy(pool["c_real_state"], np.float64)
    c_real_z = tensor_to_numpy(pool["c_real_z"], np.float64)
    success = tensor_to_numpy(pool["success"], bool)
    rank1 = int(pool["default_rank1_candidate_index"])
    oracle_best = int(np.argmin(c_real_state))
    rpool = spearman_corr(default_costs, c_real_state)
    r_real_z_on_pool = spearman_corr(c_real_z, c_real_state)
    rendpoint = None if endpoint_record is None else endpoint_record.get("Rendpoint")
    delta = None if rendpoint is None or rpool is None else float(rendpoint) - float(rpool)

    return {
        "pair_id": int(pair["pair_id"]),
        "cell": str(pair["cell"]),
        "seed": int(seed),
        "primary_subset": pair.get("primary_subset"),
        "subset_memberships": list(pair.get("subset_memberships", [])),
        "pool_path": str(pool_path),
        "rank1_candidate_index": rank1,
        "oracle_best_candidate_index": oracle_best,
        "rank1_success": bool(success[rank1]),
        "rank1_c_real_state": clean_float(float(c_real_state[rank1])),
        "oracle_best_c_real_state": clean_float(float(c_real_state[oracle_best])),
        "selection_regret_pool_rank1": clean_float(float(c_real_state[rank1] - c_real_state[oracle_best])),
        "rank1_C_model": clean_float(float(default_costs[rank1])),
        "pool_Cmodel_std": clean_float(float(np.std(default_costs, ddof=0))),
        "top30_Cmodel_std": clean_float(float(np.std(np.sort(default_costs, kind="mergesort")[:30], ddof=0))),
        "pool_Creal_std": clean_float(float(np.std(c_real_state, ddof=0))),
        "pool_Creal_range": clean_float(float(np.max(c_real_state) - np.min(c_real_state))),
        "pool_success_mass": clean_float(float(np.mean(success))),
        "Rpool_Cmodel": rpool,
        "Rpool_Creal_z_final": r_real_z_on_pool,
        "Rendpoint": clean_float(rendpoint),
        "Delta_CEM": clean_float(delta),
        "endpoint_source": None if endpoint_record is None else endpoint_record.get("source"),
        "planned_mean_after_final_update": planned_action_metrics,
    }


def endpoint_summary_record(
    *,
    source: str,
    costs: np.ndarray,
    c_real_state: np.ndarray,
    success: np.ndarray,
    raw_actions_shape: tuple[int, ...],
) -> dict[str, Any]:
    return {
        "source": source,
        "Rendpoint": spearman_corr(costs, c_real_state),
        "n_endpoint_candidates": int(len(costs)),
        "endpoint_Creal_std": clean_float(float(np.std(c_real_state, ddof=0))),
        "endpoint_success_mass": clean_float(float(np.mean(success))),
        "raw_actions_shape": list(raw_actions_shape),
    }


def score_planned_mean(
    *,
    env: Any,
    planned_raw_actions: np.ndarray,
    initial_state: np.ndarray,
    goal_state: np.ndarray,
    seed: int,
    with_velocity: bool,
    render_goal_mode: str,
) -> dict[str, Any]:
    result = rollout_raw_actions(
        env,
        initial_state=initial_state,
        goal_state=goal_state,
        raw_actions=planned_raw_actions,
        seed=seed,
        with_velocity=with_velocity,
        render_goal_mode=render_goal_mode,
    )
    return result["metrics"]


def pool_path(pool_dir: Path, pair_id: int, seed: int) -> Path:
    return pool_dir / f"pair_{int(pair_id):03d}_seed_{int(seed)}.pt"


def endpoint_pool_path(pool_dir: Path, pair_id: int, seed: int) -> Path:
    return pool_dir / f"pair_{int(pair_id):03d}_seed_{int(seed)}_endpoint.pt"


def load_existing_records(path: Path, *, resume: bool, expected_keys: set[tuple[int, int]]) -> list[dict[str, Any]]:
    if not resume or not path.exists():
        return []
    data = load_json(path)
    records = []
    for record in data.get("records", []):
        key = (int(record.get("pair_id", -1)), int(record.get("seed", -1)))
        if key in expected_keys and record.get("pool_path") and Path(record["pool_path"]).exists():
            records.append(record)
    return records


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n_records": int(len(records)),
        "rank1_success": scalar_summary([record.get("rank1_success") for record in records]),
        "planned_mean_success": scalar_summary(
            [
                record.get("planned_mean_after_final_update", {}).get("success")
                for record in records
            ]
        ),
        "selection_regret_pool_rank1": scalar_summary(
            [record.get("selection_regret_pool_rank1") for record in records]
        ),
        "Rendpoint": scalar_summary([record.get("Rendpoint") for record in records]),
        "Rpool_Cmodel": scalar_summary([record.get("Rpool_Cmodel") for record in records]),
        "Rpool_Creal_z_final": scalar_summary([record.get("Rpool_Creal_z_final") for record in records]),
        "Delta_CEM": scalar_summary([record.get("Delta_CEM") for record in records]),
        "pool_success_mass": scalar_summary([record.get("pool_success_mass") for record in records]),
        "pool_Creal_std": scalar_summary([record.get("pool_Creal_std") for record in records]),
        "pool_Cmodel_std": scalar_summary([record.get("pool_Cmodel_std") for record in records]),
    }


def print_selected_pairs(selected_pairs: list[dict[str, Any]]) -> None:
    print("Selected DINO-WM audit 30-pair PushT subset")
    print("Pair | Cell  | Primary subset     | Memberships")
    print("-----+-------+--------------------+-------------------------------")
    for pair in selected_pairs:
        print(
            f"{int(pair['pair_id']):<4} | {str(pair['cell']):<5} | "
            f"{str(pair['primary_subset']):<18} | {','.join(pair['subset_memberships'])}"
        )


def build_output(
    *,
    args: argparse.Namespace,
    pairs_data: dict[str, Any],
    selected_pairs: list[dict[str, Any]],
    anchor_definitions: dict[str, Any],
    records: list[dict[str, Any]],
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "metadata": {
            "format": "dino_wm_pusht_pool_audit_v1",
            "created_at": iso_now(),
            "git_commit": get_git_commit(PROJECT_ROOT),
            "dino_wm_git_commit": get_git_commit(args.dino_repo),
            "script_path": str(Path(__file__).resolve()),
            "dino_repo": str(args.dino_repo),
            "ckpt_base_path": str(args.ckpt_base_path),
            "model_name": args.model_name,
            "model_epoch": args.model_epoch,
            "pairs_path": str(args.pairs_path),
            "rerank_path": str(args.rerank_path),
            "pool_dir": str(args.pool_dir),
            "output": str(args.output),
            "dataset_path": str(Path(pairs_data["metadata"]["dataset_path"])),
            "seeds": [int(seed) for seed in args.seeds],
            "planner_config": {
                "num_samples": int(args.num_samples),
                "topk": int(args.topk),
                "opt_steps": int(args.opt_steps),
                "horizon": int(args.horizon),
                "var_scale": float(args.var_scale),
                "candidate_0_forced_to_search_mean": True,
            },
            "objective": {
                "alpha": float(args.alpha),
                "base": float(args.objective_base),
                "mode": str(args.objective_mode),
                "definition": "MSE(DINO terminal visual patches, goal patches) + alpha*MSE(proprio embedding, goal proprio embedding)",
            },
            "endpoint_source": str(args.endpoint_source),
            "render_goal_mode": str(args.render_goal_mode),
            "selected_pairs": selected_pairs,
            "anchor_definitions": anchor_definitions,
            "candidate_pool_scored_n": int(args.num_samples),
            "runtime": runtime,
        },
        "records": sorted(records, key=lambda item: (int(item["pair_id"]), int(item["seed"]))),
        "summary": aggregate_records(records),
    }


def main() -> int:
    args = parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    args.dino_repo = args.dino_repo.expanduser().resolve()
    args.ckpt_base_path = args.ckpt_base_path.expanduser().resolve()
    args.pairs_path = args.pairs_path.expanduser().resolve()
    args.rerank_path = args.rerank_path.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.pool_dir = args.pool_dir.expanduser().resolve()

    pairs_data, selected_pairs, _memberships, anchor_definitions = select_mppi_30_pairs(
        pairs_path=args.pairs_path,
        rerank_path=args.rerank_path,
    )
    if args.pair_ids is not None:
        requested = {int(pair_id) for pair_id in args.pair_ids}
        selected_pairs = [pair for pair in selected_pairs if int(pair["pair_id"]) in requested]
        missing = sorted(requested - {int(pair["pair_id"]) for pair in selected_pairs})
        if missing:
            raise ValueError(f"Requested pair IDs are not in the MPPI 30-pair subset: {missing}")
    if args.max_pairs is not None:
        if int(args.max_pairs) < 1:
            raise ValueError("--max-pairs must be positive")
        selected_pairs = selected_pairs[: int(args.max_pairs)]
    print_selected_pairs(selected_pairs)
    if args.list_pairs:
        return 0

    offset = int(pairs_data["metadata"]["offset"])
    validate_pair_offsets(selected_pairs, expected_offset=offset)
    dataset_path = Path(pairs_data["metadata"]["dataset_path"]).expanduser().resolve()
    expected_keys = {
        (int(pair["pair_id"]), int(seed))
        for pair in selected_pairs
        for seed in args.seeds
    }
    records = load_existing_records(args.output, resume=not args.no_resume, expected_keys=expected_keys)
    seen = {(int(record["pair_id"]), int(record["seed"])) for record in records}

    print("\n== DINO-WM PushT pool audit setup ==")
    print(f"dino_repo: {args.dino_repo}")
    print(f"ckpt_base_path: {args.ckpt_base_path}")
    print(f"model_name: {args.model_name}")
    print(f"pairs_path: {args.pairs_path}")
    print(f"dataset_path: {dataset_path}")
    print(f"output: {args.output}")
    print(f"pool_dir: {args.pool_dir}")
    print(f"device: {args.device}")
    print(f"seeds: {[int(seed) for seed in args.seeds]}")
    print(f"endpoint_source: {args.endpoint_source}")
    print(f"resume_records: {len(records)}")

    components = load_dino_components(args)
    torch = components["torch"]
    wm = components["wm"]
    preprocessor = components["preprocessor"]
    objective_fn = components["objective_fn"]
    device = components["device"]
    frameskip = int(components["frameskip"])
    with_velocity = bool(components["with_velocity"])
    action_dim = 2 * frameskip
    if int(args.horizon) != 5:
        print(f"warning: DINO-WM PushT configs use horizon=5; requested horizon={args.horizon}")
    if action_dim != frameskip * 2:
        raise RuntimeError(f"Unexpected DINO action dimension: {action_dim}")

    args.pool_dir.mkdir(parents=True, exist_ok=True)
    total_started = time.time()
    h5 = H5Rows(dataset_path)
    env = make_env(components)
    try:
        for pair_idx, pair in enumerate(selected_pairs, start=1):
            pair_id = int(pair["pair_id"])
            initial_row = h5.row(int(pair["start_row"]), keys=("state",))
            goal_row = h5.row(int(pair["goal_row"]), keys=("state",))
            initial_state = as_dino_state(initial_row["state"], with_velocity=with_velocity)
            goal_state = as_dino_state(goal_row["state"], with_velocity=with_velocity)

            for seed_idx, seed in enumerate(args.seeds, start=1):
                seed = int(seed)
                if (pair_id, seed) in seen:
                    print(f"[pair {pair_idx}/30 seed {seed_idx}/{len(args.seeds)}] pair_id={pair_id} seed={seed}: resume")
                    continue

                started = time.time()
                print(
                    f"[pair {pair_idx}/30 seed {seed_idx}/{len(args.seeds)}] "
                    f"pair_id={pair_id} seed={seed}: DINO CEM + score final {args.num_samples}"
                )
                obs_0, _ = prepare_obs_from_state(
                    env,
                    initial_state,
                    seed=seed + pair_id * 10_000,
                    with_velocity=with_velocity,
                    render_goal_state=goal_state,
                    render_goal_mode=args.render_goal_mode,
                )
                obs_g, _ = prepare_obs_from_state(
                    env,
                    goal_state,
                    seed=seed + pair_id * 10_000 + 1,
                    with_velocity=with_velocity,
                    render_goal_state=goal_state,
                    render_goal_mode=args.render_goal_mode,
                )
                capture_iters = (1, int(args.opt_steps))
                cem = run_cem_capture(
                    wm=wm,
                    preprocessor=preprocessor,
                    objective_fn=objective_fn,
                    obs_0=obs_0,
                    obs_g=obs_g,
                    device=device,
                    seed=seed + pair_id * 1009,
                    num_samples=args.num_samples,
                    topk=args.topk,
                    opt_steps=args.opt_steps,
                    horizon=args.horizon,
                    action_dim=action_dim,
                    var_scale=args.var_scale,
                    capture_iters=capture_iters,
                    save_float16_latents=args.save_float16_latents,
                )

                final_capture = cem["captures"][int(args.opt_steps)]
                final_blocked = tensor_to_numpy(final_capture["blocked_actions"], np.float32)
                final_raw = blocked_to_raw_actions(
                    final_blocked,
                    preprocessor=preprocessor,
                    frameskip=frameskip,
                )
                final_scored = score_pool(
                    env=env,
                    raw_actions=final_raw,
                    initial_state=initial_state,
                    goal_state=goal_state,
                    seed_base=seed + pair_id * 100_000,
                    with_velocity=with_velocity,
                    render_goal_mode=args.render_goal_mode,
                    wm=wm,
                    preprocessor=preprocessor,
                    objective_fn=objective_fn,
                    z_goal_single=final_capture["z_goal"],
                    device=device,
                    encode_batch_size=args.encode_batch_size,
                )

                planned_raw = blocked_to_raw_actions(
                    tensor_to_numpy(cem["planned_mean_after_final_update"].unsqueeze(0), np.float32),
                    preprocessor=preprocessor,
                    frameskip=frameskip,
                )[0]
                planned_metrics = score_planned_mean(
                    env=env,
                    planned_raw_actions=planned_raw,
                    initial_state=initial_state,
                    goal_state=goal_state,
                    seed=seed + pair_id * 100_000 + 999_999,
                    with_velocity=with_velocity,
                    render_goal_mode=args.render_goal_mode,
                )

                pool = {
                    "metadata": {
                        "format": "dino_wm_pusht_final_pool_v1",
                        "created_at": iso_now(),
                        "pair_id": pair_id,
                        "cell": str(pair["cell"]),
                        "start_row": int(pair["start_row"]),
                        "goal_row": int(pair["goal_row"]),
                        "seed": seed,
                        "sampling_seed": seed + pair_id * 1009,
                        "render_goal_mode": str(args.render_goal_mode),
                        "planner_config": {
                            "num_samples": int(args.num_samples),
                            "topk": int(args.topk),
                            "opt_steps": int(args.opt_steps),
                            "horizon": int(args.horizon),
                            "var_scale": float(args.var_scale),
                            "frameskip": int(frameskip),
                            "blocked_action_dim": int(action_dim),
                            "raw_action_steps": int(args.horizon) * int(frameskip),
                        },
                        "wallclock_seconds_unfinalized": clean_float(time.time() - started),
                    },
                    "pair_spec": dict(pair),
                    "initial_state": torch.as_tensor(initial_state, dtype=torch.float32),
                    "goal_state": torch.as_tensor(goal_state, dtype=torch.float32),
                    "z_pred": final_capture["z_pred"],
                    "z_goal": final_capture["z_goal"],
                    "blocked_actions": final_capture["blocked_actions"],
                    "raw_actions": torch.as_tensor(final_raw, dtype=torch.float32),
                    "candidate_indices": final_capture["candidate_indices"],
                    "default_costs": final_capture["default_costs"],
                    "default_rank1_candidate_index": int(final_capture["rank1_candidate_index"]),
                    "top30_costs": final_capture["top30_costs"],
                    "mu_before_final_update": final_capture["mu_before_update"],
                    "sigma_before_final_update": final_capture["sigma_before_update"],
                    "planned_mean_after_final_update": cem["planned_mean_after_final_update"],
                    "planned_sigma_after_final_update": cem["planned_sigma_after_final_update"],
                    "planned_raw_actions_after_final_update": torch.as_tensor(planned_raw, dtype=torch.float32),
                    **final_scored,
                }
                pool["metadata"]["wallclock_seconds"] = clean_float(time.time() - started)

                endpoint_record = None
                if args.endpoint_source == "final_pool_real_z":
                    endpoint_record = endpoint_summary_record(
                        source="final_pool_real_z",
                        costs=tensor_to_numpy(pool["c_real_z"], np.float64),
                        c_real_state=tensor_to_numpy(pool["c_real_state"], np.float64),
                        success=tensor_to_numpy(pool["success"], bool),
                        raw_actions_shape=tuple(final_raw.shape),
                    )
                elif args.endpoint_source == "first_iter_real_z":
                    first_capture = cem["captures"][1]
                    first_blocked = tensor_to_numpy(first_capture["blocked_actions"], np.float32)
                    first_raw = blocked_to_raw_actions(
                        first_blocked,
                        preprocessor=preprocessor,
                        frameskip=frameskip,
                    )
                    first_scored = score_pool(
                        env=env,
                        raw_actions=first_raw,
                        initial_state=initial_state,
                        goal_state=goal_state,
                        seed_base=seed + pair_id * 100_000 + 50_000,
                        with_velocity=with_velocity,
                        render_goal_mode=args.render_goal_mode,
                        wm=wm,
                        preprocessor=preprocessor,
                        objective_fn=objective_fn,
                        z_goal_single=first_capture["z_goal"],
                        device=device,
                        encode_batch_size=args.encode_batch_size,
                    )
                    endpoint_record = endpoint_summary_record(
                        source="first_iter_real_z",
                        costs=tensor_to_numpy(first_scored["c_real_z"], np.float64),
                        c_real_state=tensor_to_numpy(first_scored["c_real_state"], np.float64),
                        success=tensor_to_numpy(first_scored["success"], bool),
                        raw_actions_shape=tuple(first_raw.shape),
                    )
                    pool["endpoint_first_iter_summary"] = endpoint_record
                    if args.save_endpoint_pools:
                        endpoint_pool = {
                            "metadata": {
                                "format": "dino_wm_pusht_endpoint_first_iter_pool_v1",
                                "created_at": iso_now(),
                                "pair_id": pair_id,
                                "seed": seed,
                            },
                            "pair_spec": dict(pair),
                            "z_pred": first_capture["z_pred"],
                            "z_goal": first_capture["z_goal"],
                            "blocked_actions": first_capture["blocked_actions"],
                            "raw_actions": torch.as_tensor(first_raw, dtype=torch.float32),
                            "default_costs": first_capture["default_costs"],
                            **first_scored,
                        }
                        torch.save(endpoint_pool, endpoint_pool_path(args.pool_dir, pair_id, seed))

                path = pool_path(args.pool_dir, pair_id, seed)
                torch.save(pool, path)
                record = pool_summary_record(
                    pool=pool,
                    endpoint_record=endpoint_record,
                    pool_path=path,
                    pair=pair,
                    seed=seed,
                    planned_action_metrics=planned_metrics,
                )
                records.append(record)
                seen.add((pair_id, seed))

                print(
                    f"  saved {path}; Rpool={record['Rpool_Cmodel']}; "
                    f"Rendpoint={record['Rendpoint']}; Delta={record['Delta_CEM']}; "
                    f"elapsed={time.time() - started:.1f}s"
                )
                write_json(
                    args.output,
                    build_output(
                        args=args,
                        pairs_data=pairs_data,
                        selected_pairs=selected_pairs,
                        anchor_definitions=anchor_definitions,
                        records=records,
                    ),
                )
    finally:
        env.close()
        h5.close()

    runtime = {
        "wallclock_seconds": clean_float(time.time() - total_started),
        "wallclock_minutes": clean_float((time.time() - total_started) / 60.0),
    }
    output = build_output(
        args=args,
        pairs_data=pairs_data,
        selected_pairs=selected_pairs,
        anchor_definitions=anchor_definitions,
        records=records,
        runtime=runtime,
    )
    write_json(args.output, output)
    print("\n== DINO-WM audit summary ==")
    print(json.dumps(jsonable(output["summary"]), indent=2, allow_nan=False))
    print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
