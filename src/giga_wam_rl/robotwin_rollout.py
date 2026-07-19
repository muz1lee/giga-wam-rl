from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import time
from typing import Any

import numpy as np

from giga_wam_rl.gwp05_action_policy import GWP05ActionPolicy
from giga_wam_rl.robotwin_collection import (
    EpisodeBuffer,
    ReplanProposal,
    fixed_cadence_targets,
)


@dataclass(frozen=True)
class RolloutSettings:
    task: str
    instruction: str
    step_limit: int
    max_actions: int
    replan_steps: int
    simulator_hz: int
    simulator_steps_per_action: int

    def __post_init__(self) -> None:
        if self.step_limit <= 0 or self.max_actions <= 0:
            raise ValueError("step limits must be positive")
        if not 1 <= self.replan_steps <= 48:
            raise ValueError("replan steps must be in [1, 48]")
        if self.simulator_hz <= 0 or self.simulator_steps_per_action <= 0:
            raise ValueError("simulator cadence must be positive")


@dataclass(frozen=True)
class EpisodeCollectionResult:
    buffer: EpisodeBuffer
    success: bool
    termination_reason: str
    wall_time_s: float


def worker_seed_sequence(
    *,
    seed_start: int,
    episode_count: int,
    worker_id: int,
    num_workers: int,
) -> list[int]:
    if episode_count < 0:
        raise ValueError("episode count must be non-negative")
    if num_workers <= 0 or not 0 <= worker_id < num_workers:
        raise ValueError("worker id must be inside the worker count")
    return [
        int(seed_start) + int(worker_id) + index * int(num_workers)
        for index in range(int(episode_count))
    ]


def _current_drive_target(env: Any) -> np.ndarray:
    return np.asarray(
        env.robot.get_left_arm_jointState() + env.robot.get_right_arm_jointState(),
        dtype=np.float32,
    )


def execute_fixed_cadence_target(
    env: Any,
    target: np.ndarray,
    *,
    simulator_hz: int,
    simulator_steps_per_action: int,
) -> np.ndarray:
    target = np.asarray(target, dtype=np.float32).copy()
    if target.shape != (14,) or not np.isfinite(target).all():
        raise ValueError("target must be a finite [14] vector")
    target[[6, 13]] = np.clip(target[[6, 13]], 0.0, 1.0)
    current = _current_drive_target(env)
    path = fixed_cadence_targets(
        current, target, simulator_steps=simulator_steps_per_action
    )
    seconds_per_step = 1.0 / float(simulator_hz)
    previous = current
    for drive_target in path:
        left_velocity = (drive_target[:6] - previous[:6]) / seconds_per_step
        right_velocity = (drive_target[7:13] - previous[7:13]) / seconds_per_step
        env.robot.set_arm_joints(drive_target[:6], left_velocity, "left")
        env.robot.set_arm_joints(drive_target[7:13], right_velocity, "right")
        env.robot.set_gripper(float(drive_target[6]), "left")
        env.robot.set_gripper(float(drive_target[13]), "right")
        env.scene.step()
        env._update_render()
        previous = drive_target
    env.take_action_cnt += 1
    return target


def collect_episode(
    *,
    env: Any,
    policy: GWP05ActionPolicy,
    settings: RolloutSettings,
    env_seed: int,
    policy_seed: int,
) -> EpisodeCollectionResult:
    started = time.perf_counter()
    buffer = EpisodeBuffer(
        task=settings.task,
        instruction=settings.instruction,
        env_seed=int(env_seed),
        policy_seed=int(policy_seed),
        simulator_hz=settings.simulator_hz,
        simulator_steps_per_action=settings.simulator_steps_per_action,
    )
    initial_observation = env.get_obs()
    buffer.append_initial(initial_observation, simulator_step=0, wall_time_s=0.0)
    pending: deque[tuple[int, int, np.ndarray]] = deque()
    replan_index = 0
    simulator_step = 0
    success = False
    action_budget = min(settings.step_limit, settings.max_actions)

    while len(buffer.executed_actions) < action_budget:
        if not pending:
            current = buffer.observations[-1]
            current_policy_seed = int(policy_seed) + replan_index
            prediction = policy.predict(
                cameras=current.cameras,
                state=current.state,
                seed=current_policy_seed,
            )
            committed_length = settings.replan_steps
            proposal = ReplanProposal(
                observation_index=len(buffer.observations) - 1,
                replan_index=replan_index,
                policy_seed=current_policy_seed,
                normalized_action=prediction.normalized_action,
                physical_action=prediction.physical_action,
                committed_length=committed_length,
                inference_time_s=prediction.inference_time_s,
            )
            buffer.record_replan(proposal)
            for proposal_offset in range(committed_length):
                pending.append(
                    (
                        replan_index,
                        proposal_offset,
                        prediction.physical_action[proposal_offset].copy(),
                    )
                )
            replan_index += 1

        current_replan, proposal_offset, target = pending.popleft()
        executed_action = execute_fixed_cadence_target(
            env,
            target,
            simulator_hz=settings.simulator_hz,
            simulator_steps_per_action=settings.simulator_steps_per_action,
        )
        simulator_step += settings.simulator_steps_per_action
        next_observation = env.get_obs()
        buffer.append_transition(
            executed_action=executed_action,
            next_observation=next_observation,
            simulator_step=simulator_step,
            wall_time_s=time.perf_counter() - started,
            replan_index=current_replan,
            proposal_offset=proposal_offset,
        )
        if bool(env.check_success()):
            env.eval_success = True
            success = True
            break

    if success:
        termination_reason = "success"
    elif len(buffer.executed_actions) >= settings.max_actions:
        termination_reason = "max_actions"
    else:
        termination_reason = "step_limit"
    return EpisodeCollectionResult(
        buffer=buffer,
        success=success,
        termination_reason=termination_reason,
        wall_time_s=time.perf_counter() - started,
    )
