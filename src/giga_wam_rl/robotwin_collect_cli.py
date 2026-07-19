from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time
import tomllib
from typing import Any, Sequence

from giga_wam_rl.gwp05_action_policy import load_gwp05_action_policy
from giga_wam_rl.robotwin_collection import write_episode
from giga_wam_rl.robotwin_rollout import (
    RolloutSettings,
    collect_episode,
    worker_seed_sequence,
)
from giga_wam_rl.robotwin_runtime import (
    close_task_env,
    create_task_env,
    load_task_args,
)
from giga_wam_rl.workspace import (
    load_registry,
    validate_output_root,
    validate_registry,
)


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as input_file:
        return tomllib.load(input_file)


def _git_revision(path: Path, *, require_clean: bool) -> str:
    revision = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if require_clean:
        status = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain=v1", "-uall"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if status:
            raise RuntimeError(f"collection requires a clean checkout: {path}")
    return revision


def run_collection(
    *,
    project_root: Path,
    registry_path: Path,
    config_path: Path,
    run_id: str,
    worker_id: int,
    num_workers: int,
    episode_count: int | None,
    max_actions: int | None,
    device_name: str,
) -> dict[str, Any]:
    registry = load_registry(registry_path)
    validate_registry(registry)
    config = _load_toml(config_path)
    robotwin_config = config["robotwin"]
    policy_config = config["policy"]
    collection_config = config["collection"]
    output_root = validate_output_root(
        Path(collection_config["output_root"]) / run_id,
        artifact_root=Path(registry["workspace"]["artifact_root"]),
        protected_roots=tuple(
            Path(path) for path in registry["workspace"]["protected_roots"]
        ),
    )
    output_root.mkdir(parents=True, exist_ok=True)
    worker_dir = output_root / f"worker_{worker_id:02d}"
    worker_dir.mkdir()

    project_revision = _git_revision(project_root, require_clean=True)
    upstream_root = Path(policy_config["upstream_root"]).resolve(strict=True)
    gwp_revision = _git_revision(upstream_root, require_clean=True)
    robotwin_root = Path(robotwin_config["root"]).resolve(strict=True)
    task_config_path = Path(robotwin_config["task_config_path"]).resolve(strict=True)
    settings = RolloutSettings(
        task=robotwin_config["task"],
        instruction=robotwin_config["instruction"],
        step_limit=int(robotwin_config["step_limit"]),
        max_actions=int(
            max_actions if max_actions is not None else collection_config["max_actions"]
        ),
        replan_steps=int(policy_config["replan_steps"]),
        simulator_hz=int(robotwin_config["simulator_hz"]),
        simulator_steps_per_action=int(robotwin_config["simulator_steps_per_action"]),
    )
    count = int(
        episode_count
        if episode_count is not None
        else collection_config["episode_count_per_worker"]
    )
    seeds = worker_seed_sequence(
        seed_start=int(collection_config["seed_start"]),
        episode_count=count,
        worker_id=worker_id,
        num_workers=num_workers,
    )
    task_args = load_task_args(
        robotwin_root=robotwin_root,
        task_config_path=task_config_path,
        task_name=settings.task,
        writable_save_root=worker_dir / "robotwin_runtime",
    )
    policy = load_gwp05_action_policy(
        checkpoint=Path(policy_config["checkpoint"]).resolve(strict=True),
        base_model=Path(policy_config["base_model"]).resolve(strict=True),
        upstream_root=upstream_root,
        norm_stats_path=Path(policy_config["norm_stats"]).resolve(strict=True),
        prompt=settings.instruction,
        device_name=device_name,
        num_inference_steps=int(policy_config["num_inference_steps"]),
        clip_normalized_actions=bool(policy_config["clip_normalized_actions"]),
        compile_transformer=bool(policy_config["compile_transformer"]),
    )
    started = time.perf_counter()
    episode_summaries = []
    for episode_index, env_seed in enumerate(seeds):
        env = None
        episode_started = time.perf_counter()
        try:
            env = create_task_env(
                robotwin_root=robotwin_root,
                task_name=settings.task,
                task_args=task_args,
                episode_index=episode_index,
                env_seed=env_seed,
                instruction=settings.instruction,
            )
            env.step_lim = settings.step_limit
            policy_seed = int(collection_config["policy_seed_start"]) + env_seed * 1000
            result = collect_episode(
                env=env,
                policy=policy,
                settings=settings,
                env_seed=env_seed,
                policy_seed=policy_seed,
            )
            manifest = write_episode(
                result.buffer,
                worker_dir / f"episode_seed_{env_seed:08d}",
                success=result.success,
                termination_reason=result.termination_reason,
                code_revision=project_revision,
                upstream_revisions={
                    "giga_world_policy": gwp_revision,
                    "robotwin_runtime": str(robotwin_config["runtime_revision"]),
                },
            )
            episode_summaries.append(manifest)
        finally:
            if env is not None:
                close_task_env(env)
        print(
            json.dumps(
                {
                    "event": "episode_complete",
                    "worker_id": worker_id,
                    "env_seed": env_seed,
                    "wall_time_s": round(time.perf_counter() - episode_started, 3),
                    "success": episode_summaries[-1]["success"],
                    "num_actions": episode_summaries[-1]["num_executed_actions"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "worker_id": worker_id,
        "num_workers": num_workers,
        "seeds": seeds,
        "num_episodes": len(episode_summaries),
        "num_success": sum(bool(row["success"]) for row in episode_summaries),
        "total_wall_time_s": time.perf_counter() - started,
        "project_revision": project_revision,
        "giga_world_policy_revision": gwp_revision,
        "robotwin_runtime_revision": str(robotwin_config["runtime_revision"]),
        "settings": {
            "max_actions": settings.max_actions,
            "replan_steps": settings.replan_steps,
            "simulator_steps_per_action": settings.simulator_steps_per_action,
        },
    }
    (worker_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Collect GWP-0.5 closed-loop RoboTwin trajectories"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root / "configs/rollouts/place_bread_gwp05_clean.toml",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=project_root / "configs/assets.server.toml",
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--episode-count", type=int)
    parser.add_argument("--max-actions", type=int)
    parser.add_argument("--device", default="cuda:0")
    arguments = parser.parse_args(argv)
    summary = run_collection(
        project_root=project_root,
        registry_path=arguments.registry,
        config_path=arguments.config,
        run_id=arguments.run_id,
        worker_id=arguments.worker_id,
        num_workers=arguments.num_workers,
        episode_count=arguments.episode_count,
        max_actions=arguments.max_actions,
        device_name=arguments.device,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
