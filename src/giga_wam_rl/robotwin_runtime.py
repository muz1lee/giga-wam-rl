from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import sys
from typing import Any, Iterator


def add_robotwin_import_paths(robotwin_root: Path) -> None:
    for path in (
        robotwin_root,
        robotwin_root / "script",
        robotwin_root / "description" / "utils",
    ):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


@contextmanager
def robotwin_cwd(robotwin_root: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(robotwin_root)
    try:
        yield
    finally:
        os.chdir(previous)


def load_task_args(
    *,
    robotwin_root: Path,
    task_config_path: Path,
    task_name: str,
    writable_save_root: Path,
) -> dict[str, Any]:
    import yaml

    add_robotwin_import_paths(robotwin_root)
    with task_config_path.open("r", encoding="utf-8") as input_file:
        task_args = yaml.load(input_file.read(), Loader=yaml.FullLoader)
    with robotwin_cwd(robotwin_root):
        from eval_policy import get_embodiment_config

    task_args.update(
        {
            "task_name": task_name,
            "task_config": task_config_path.stem,
            "ckpt_setting": str(writable_save_root),
            "save_path": str(writable_save_root),
            "collect_data": False,
            "eval_mode": True,
            "render_freq": 0,
            "eval_video_log": False,
        }
    )
    embodiment_types = task_args["embodiment"]
    embodiment_config_path = robotwin_root / "task_config" / "_embodiment_config.yml"
    with embodiment_config_path.open("r", encoding="utf-8") as input_file:
        embodiment_configs = yaml.load(input_file.read(), Loader=yaml.FullLoader)

    def embodiment_path(name: str) -> Path:
        configured = Path(embodiment_configs[name]["file_path"])
        return (robotwin_root / configured).resolve(strict=True)

    if len(embodiment_types) != 1:
        raise ValueError("first GWP collector supports one dual-arm embodiment")
    robot_path = embodiment_path(embodiment_types[0])
    task_args["left_robot_file"] = str(robot_path)
    task_args["right_robot_file"] = str(robot_path)
    task_args["dual_arm_embodied"] = True
    with robotwin_cwd(robotwin_root):
        task_args["left_embodiment_config"] = get_embodiment_config(str(robot_path))
        task_args["right_embodiment_config"] = get_embodiment_config(str(robot_path))

    camera_config_path = robotwin_root / "task_config" / "_camera_config.yml"
    with camera_config_path.open("r", encoding="utf-8") as input_file:
        camera_configs = yaml.load(input_file.read(), Loader=yaml.FullLoader)
    head_camera = camera_configs[task_args["camera"]["head_camera_type"]]
    task_args["head_camera_h"] = head_camera["h"]
    task_args["head_camera_w"] = head_camera["w"]
    return task_args


def create_task_env(
    *,
    robotwin_root: Path,
    task_name: str,
    task_args: dict[str, Any],
    episode_index: int,
    env_seed: int,
    instruction: str,
) -> Any:
    add_robotwin_import_paths(robotwin_root)
    with robotwin_cwd(robotwin_root):
        from eval_policy import class_decorator

        env = class_decorator(task_name)
        env.setup_demo(
            now_ep_num=int(episode_index),
            seed=int(env_seed),
            is_test=True,
            **task_args,
        )
        env.set_instruction(instruction=instruction)
    return env


def close_task_env(env: Any) -> None:
    close = getattr(env, "close_env", None)
    if close is None:
        return
    try:
        close(clear_cache=True)
    except TypeError:
        close()
