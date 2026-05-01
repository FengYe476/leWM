from __future__ import annotations

import importlib
import inspect
import multiprocessing as mp
import os
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Callable


def package_version(distribution_name: str) -> str:
    try:
        return importlib_metadata.version(distribution_name)
    except importlib_metadata.PackageNotFoundError:
        return "unknown"


def package_location(module) -> str:
    module_file = getattr(module, "__file__", None)
    if module_file:
        return str(Path(module_file).resolve())
    try:
        return str(Path(inspect.getfile(module)).resolve())
    except (TypeError, OSError):
        return "unknown"


def make_row(name: str, ok: bool, details: str) -> dict[str, str]:
    return {
        "check": name,
        "status": "PASS" if ok else "FAIL",
        "details": details,
    }


def render_table(rows: list[dict[str, str]]) -> str:
    headers = ("Check", "Status", "Details")
    widths = [
        max(len(headers[0]), *(len(row["check"]) for row in rows)),
        max(len(headers[1]), *(len(row["status"]) for row in rows)),
        max(len(headers[2]), *(len(row["details"]) for row in rows)),
    ]

    def fmt(values: tuple[str, str, str]) -> str:
        return (
            f"{values[0]:<{widths[0]}} | "
            f"{values[1]:<{widths[1]}} | "
            f"{values[2]:<{widths[2]}}"
        )

    divider = "-+-".join("-" * width for width in widths)
    lines = [fmt(headers), divider]
    for row in rows:
        lines.append(fmt((row["check"], row["status"], row["details"])))
    return "\n".join(lines)


def run_check(name: str, fn: Callable[[], str]) -> dict[str, str]:
    try:
        details = fn()
        return make_row(name, True, details)
    except Exception as exc:  # noqa: BLE001
        return make_row(name, False, f"{type(exc).__name__}: {exc}")


def _env_worker(candidates: list[str], env_kwargs: dict, env_vars: dict, queue) -> None:
    try:
        queue.put(("ok", try_gym_env(candidates, env_kwargs=env_kwargs, env_vars=env_vars)))
    except Exception as exc:  # noqa: BLE001
        queue.put(("err", f"{type(exc).__name__}: {exc}"))


def run_env_check_with_timeout(
    candidates: list[str],
    *,
    env_kwargs: dict | None = None,
    env_vars: dict | None = None,
    timeout_s: int = 20,
) -> str:
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(
        target=_env_worker,
        args=(candidates, env_kwargs or {}, env_vars or {}, queue),
    )
    proc.start()
    proc.join(timeout_s)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        raise TimeoutError(f"timed out after {timeout_s}s")

    if queue.empty():
        raise RuntimeError("subprocess exited without returning a result")

    status, payload = queue.get()
    if status == "ok":
        return payload
    raise RuntimeError(payload)


def check_stable_worldmodel() -> str:
    module = importlib.import_module("stable_worldmodel")
    version = package_version("stable-worldmodel")
    location = package_location(module)
    print(f"stable_worldmodel version: {version}")
    print(f"stable_worldmodel location: {location}")
    return f"version={version}; location={location}"


def check_stable_pretraining() -> str:
    module = importlib.import_module("stable_pretraining")
    version = package_version("stable-pretraining")
    location = package_location(module)
    print(f"stable_pretraining version: {version}")
    print(f"stable_pretraining location: {location}")
    return f"version={version}; location={location}"


def check_ogbench() -> str:
    module = importlib.import_module("ogbench")
    version = package_version("ogbench")
    location = package_location(module)
    print(f"ogbench version: {version}")
    print(f"ogbench location: {location}")
    return f"version={version}; location={location}"


def check_torch() -> str:
    torch = importlib.import_module("torch")
    version = getattr(torch, "__version__", "unknown")
    cuda_available = bool(torch.cuda.is_available())
    cuda_version = getattr(torch.version, "cuda", None)
    print(f"torch version: {version}")
    print(f"torch CUDA available: {cuda_available}")
    print(f"torch CUDA runtime version: {cuda_version}")
    return (
        f"version={version}; cuda_available={cuda_available}; "
        f"cuda_runtime={cuda_version}"
    )


def try_gym_env(
    candidates: list[str],
    *,
    env_kwargs: dict | None = None,
    env_vars: dict | None = None,
) -> str:
    for key, value in (env_vars or {}).items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value

    # Importing stable_worldmodel registers the `swm/...` Gymnasium namespace.
    importlib.import_module("stable_worldmodel")
    gym = importlib.import_module("gymnasium")
    last_error = None

    for env_id in candidates:
        try:
            env = gym.make(env_id, **(env_kwargs or {}))
            try:
                env.reset()
            finally:
                env.close()
            return env_id
        except Exception as exc:  # noqa: BLE001
            last_error = exc

    if last_error is None:
        raise RuntimeError("no environment IDs were provided")
    raise last_error


def check_pusht_env() -> str:
    env_id = run_env_check_with_timeout(["swm/PushT-v1"])
    print(f"PushT environment instantiated: {env_id}")
    return f"instantiated {env_id}"


def check_ogbench_cube_env() -> str:
    env_id = run_env_check_with_timeout(
        ["swm/OGBCube-v0"],
        env_kwargs={"render_mode": None},
        env_vars={"MUJOCO_GL": "disabled"},
    )
    print(f"OGBench-Cube environment instantiated: {env_id}")
    return f"instantiated {env_id}"


def main() -> int:
    rows = [
        run_check("stable_worldmodel import", check_stable_worldmodel),
        run_check("stable_pretraining import", check_stable_pretraining),
        run_check("ogbench import", check_ogbench),
        run_check("torch import", check_torch),
        run_check("PushT env", check_pusht_env),
        run_check("OGBench-Cube env", check_ogbench_cube_env),
    ]

    print()
    print(render_table(rows))

    failures = [row for row in rows if row["status"] != "PASS"]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
