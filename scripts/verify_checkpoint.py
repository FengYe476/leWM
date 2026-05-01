from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEWM_ROOT = PROJECT_ROOT / "third_party" / "le-wm"
CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints"
CONVERTED_ROOT = CHECKPOINT_ROOT / "converted"

if str(LEWM_ROOT) not in sys.path:
    sys.path.insert(0, str(LEWM_ROOT))

from jepa import JEPA  # noqa: E402
from module import ARPredictor, Embedder, MLP  # noqa: E402


@dataclass
class CheckpointResult:
    name: str
    load_ok: bool
    encoder_ok: bool
    predictor_ok: bool
    embed_dim: int | str
    params_m: str
    encoder_shape: str = "-"
    predictor_shape: str = "-"
    details: str = ""


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def fmt_millions(n: int) -> str:
    return f"{n / 1_000_000:.1f}M"


def render_table(rows: list[dict[str, str]]) -> str:
    headers = tuple(rows[0].keys())
    widths = [
        max(len(header), *(len(str(row[header])) for row in rows))
        for header in headers
    ]

    def fmt(row: dict[str, str]) -> str:
        return " | ".join(
            f"{str(row[h]):<{widths[i]}}" for i, h in enumerate(headers)
        )

    divider = "-+-".join("-" * w for w in widths)
    return "\n".join([fmt({h: h for h in headers}), divider] + [fmt(r) for r in rows])


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def resolve_norm(norm_cfg: dict | None):
    if not norm_cfg:
        return nn.Identity
    target = norm_cfg.get("_target_")
    if target == "torch.nn.BatchNorm1d":
        return nn.BatchNorm1d
    raise ValueError(f"Unsupported norm config: {norm_cfg}")


def build_model_from_config(cfg: dict) -> JEPA:
    encoder_cfg = cfg["encoder"]
    predictor_cfg = cfg["predictor"]
    action_cfg = cfg["action_encoder"]
    projector_cfg = cfg["projector"]
    pred_proj_cfg = cfg["pred_proj"]

    encoder = spt.backbone.utils.vit_hf(
        encoder_cfg["size"],
        patch_size=encoder_cfg["patch_size"],
        image_size=encoder_cfg["image_size"],
        pretrained=encoder_cfg.get("pretrained", False),
        use_mask_token=encoder_cfg.get("use_mask_token", False),
    )

    predictor = ARPredictor(
        num_frames=predictor_cfg["num_frames"],
        input_dim=predictor_cfg["input_dim"],
        hidden_dim=predictor_cfg["hidden_dim"],
        output_dim=predictor_cfg["output_dim"],
        depth=predictor_cfg["depth"],
        heads=predictor_cfg["heads"],
        mlp_dim=predictor_cfg["mlp_dim"],
        dim_head=predictor_cfg.get("dim_head", 64),
        dropout=predictor_cfg.get("dropout", 0.0),
        emb_dropout=predictor_cfg.get("emb_dropout", 0.0),
    )

    action_encoder = Embedder(
        input_dim=action_cfg["input_dim"],
        emb_dim=action_cfg["emb_dim"],
    )

    projector = MLP(
        input_dim=projector_cfg["input_dim"],
        hidden_dim=projector_cfg["hidden_dim"],
        output_dim=projector_cfg["output_dim"],
        norm_fn=resolve_norm(projector_cfg.get("norm_fn")),
    )

    pred_proj = MLP(
        input_dim=pred_proj_cfg["input_dim"],
        hidden_dim=pred_proj_cfg["hidden_dim"],
        output_dim=pred_proj_cfg["output_dim"],
        norm_fn=resolve_norm(pred_proj_cfg.get("norm_fn")),
    )

    model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=pred_proj,
    )
    return model


def get_env_action_dim(name: str) -> int:
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    import ogbench  # noqa: F401

    if name == "lewm-pusht":
        env = gym.make("swm/PushT-v1")
    elif name == "lewm-cube":
        os.environ["MUJOCO_GL"] = "disabled"
        env = gym.make("swm/OGBCube-v0", render_mode=None)
    else:
        raise ValueError(name)

    try:
        shape = getattr(env.action_space, "shape", None)
        if not shape:
            raise RuntimeError(f"Action space for {name} has no shape: {env.action_space}")
        return int(shape[0])
    finally:
        env.close()


def convert_to_object_checkpoint(model: JEPA, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "lewm_object.ckpt"
    torch.save(model, out_path)
    return out_path


def verify_checkpoint(name: str) -> CheckpointResult:
    ckpt_dir = CHECKPOINT_ROOT / name
    cfg = load_json(ckpt_dir / "config.json")
    model = build_model_from_config(cfg)

    state_dict = torch.load(ckpt_dir / "weights.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    converted_dir = CONVERTED_ROOT / name
    object_ckpt = convert_to_object_checkpoint(model, converted_dir)
    auto_model = swm.policy.AutoCostModel(str(converted_dir))
    auto_model.eval()

    total_params = count_params(model)
    encoder_params = count_params(model.encoder)
    predictor_stack = (
        count_params(model.predictor)
        + count_params(model.action_encoder)
        + count_params(model.projector)
        + count_params(model.pred_proj)
    )

    env_action_dim = get_env_action_dim(name)
    effective_action_dim = cfg["action_encoder"]["input_dim"]
    num_frames = cfg["predictor"]["num_frames"]
    embed_dim = cfg["projector"]["output_dim"]
    image_size = cfg["encoder"]["image_size"]

    print(f"\n{name}")
    print(f"  config path: {ckpt_dir / 'config.json'}")
    print(f"  weights path: {ckpt_dir / 'weights.pt'}")
    print(f"  converted object checkpoint: {object_ckpt}")
    print(f"  env action dim: {env_action_dim}")
    print(f"  effective action dim in checkpoint: {effective_action_dim}")
    print(f"  total params: {total_params} ({fmt_millions(total_params)})")
    print(f"  encoder params: {encoder_params} ({fmt_millions(encoder_params)})")
    print(f"  predictor stack params: {predictor_stack} ({fmt_millions(predictor_stack)})")

    with torch.inference_mode():
        pixels = torch.randn(1, 3, image_size, image_size)
        encoder_out = auto_model.encoder(pixels, interpolate_pos_encoding=True)
        cls = encoder_out.last_hidden_state[:, 0]
        emb = auto_model.projector(cls)
        print(f"  encoder forward shape: {tuple(emb.shape)}")

        ctx_emb = emb.unsqueeze(1).expand(-1, num_frames, -1).contiguous()
        actions = torch.randn(1, num_frames, effective_action_dim)
        act_emb = auto_model.action_encoder(actions)
        pred = auto_model.predict(ctx_emb, act_emb)
        print(f"  predictor forward shape: {tuple(pred.shape)}")

    load_ok = True
    encoder_ok = tuple(emb.shape) == (1, embed_dim)
    predictor_ok = tuple(pred.shape) == (1, num_frames, embed_dim)

    return CheckpointResult(
        name=name,
        load_ok=load_ok,
        encoder_ok=encoder_ok,
        predictor_ok=predictor_ok,
        embed_dim=embed_dim,
        params_m=fmt_millions(total_params),
        encoder_shape=str(tuple(emb.shape)),
        predictor_shape=str(tuple(pred.shape)),
        details=(
            f"raw_action_dim={env_action_dim}; effective_action_dim={effective_action_dim}; "
            f"object_ckpt={object_ckpt}"
        ),
    )


def main() -> int:
    names = ["lewm-pusht", "lewm-cube"]
    results: list[CheckpointResult] = []

    for name in names:
        try:
            results.append(verify_checkpoint(name))
        except Exception as exc:  # noqa: BLE001
            results.append(
                CheckpointResult(
                    name=name,
                    load_ok=False,
                    encoder_ok=False,
                    predictor_ok=False,
                    embed_dim="-",
                    params_m="-",
                    details=f"{type(exc).__name__}: {exc}",
                )
            )

    print("\nSummary")
    rows = []
    for r in results:
        rows.append(
            {
                "Checkpoint": r.name,
                "Load": "PASS" if r.load_ok else "FAIL",
                "Encoder Forward": (
                    f"PASS {r.encoder_shape}" if r.encoder_ok else "FAIL"
                ),
                "Predictor Forward": (
                    f"PASS {r.predictor_shape}" if r.predictor_ok else "FAIL"
                ),
                "Embed Dim": str(r.embed_dim),
                "Params": r.params_m,
            }
        )
    print(render_table(rows))

    print("\nDetails")
    for r in results:
        print(f"{r.name}: {r.details}")

    return 0 if all(r.load_ok and r.encoder_ok and r.predictor_ok for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
