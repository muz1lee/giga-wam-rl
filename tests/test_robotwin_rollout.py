import numpy as np

from giga_wam_rl.gwp05_action_policy import PolicyPrediction
from giga_wam_rl.robotwin_rollout import (
    RolloutSettings,
    collect_episode,
    worker_seed_sequence,
)


class _FakeScene:
    def __init__(self) -> None:
        self.steps = 0

    def step(self) -> None:
        self.steps += 1


class _FakeRobot:
    def __init__(self) -> None:
        self.state = np.zeros(14, dtype=np.float32)

    def get_left_arm_jointState(self):
        return self.state[:7].tolist()

    def get_right_arm_jointState(self):
        return self.state[7:].tolist()

    def set_arm_joints(self, position, velocity, arm_tag):
        del velocity
        if arm_tag == "left":
            self.state[:6] = position
        else:
            self.state[7:13] = position

    def set_gripper(self, value, arm_tag):
        if arm_tag == "left":
            self.state[6] = value
        else:
            self.state[13] = value


class _FakeEnv:
    def __init__(self, *, success_after_actions: int) -> None:
        self.robot = _FakeRobot()
        self.scene = _FakeScene()
        self.take_action_cnt = 0
        self.eval_success = False
        self.success_after_actions = success_after_actions

    def _update_render(self):
        return None

    def get_obs(self):
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        return {
            "observation": {
                "head_camera": {"rgb": image},
                "left_camera": {"rgb": image},
                "right_camera": {"rgb": image},
            },
            "joint_action": {"vector": self.robot.state.copy()},
        }

    def check_success(self):
        return self.take_action_cnt >= self.success_after_actions


class _FakePolicy:
    def __init__(self) -> None:
        self.calls = []

    def predict(self, *, cameras, state, seed):
        self.calls.append((cameras, state.copy(), seed))
        physical = np.repeat((state + 0.1)[None], 48, axis=0).astype(np.float32)
        physical[:, 6] = 0.2
        physical[:, 13] = 0.3
        return PolicyPrediction(
            normalized_action=np.zeros((48, 16), dtype=np.float32),
            physical_action=physical,
            inference_time_s=0.01,
            seed=seed,
        )


def test_collect_episode_executes_committed_prefix_at_fixed_cadence() -> None:
    env = _FakeEnv(success_after_actions=2)
    policy = _FakePolicy()
    settings = RolloutSettings(
        task="place_bread_basket",
        instruction="Pick up the bread and place it in the basket.",
        step_limit=700,
        max_actions=4,
        replan_steps=2,
        simulator_hz=250,
        simulator_steps_per_action=15,
    )

    result = collect_episode(
        env=env,
        policy=policy,
        settings=settings,
        env_seed=5,
        policy_seed=100,
    )

    assert result.success is True
    assert result.termination_reason == "success"
    assert len(result.buffer.executed_actions) == 2
    assert len(result.buffer.observations) == 3
    assert len(result.buffer.replans) == 1
    assert result.buffer.simulator_steps == [0, 15, 30]
    assert env.scene.steps == 30
    assert policy.calls[0][2] == 100


def test_collect_episode_replans_with_non_overlapping_policy_seeds() -> None:
    env = _FakeEnv(success_after_actions=99)
    policy = _FakePolicy()
    settings = RolloutSettings(
        task="place_bread_basket",
        instruction="instruction",
        step_limit=700,
        max_actions=3,
        replan_steps=2,
        simulator_hz=250,
        simulator_steps_per_action=1,
    )

    result = collect_episode(
        env=env,
        policy=policy,
        settings=settings,
        env_seed=5,
        policy_seed=100,
    )

    assert result.success is False
    assert result.termination_reason == "max_actions"
    assert [call[2] for call in policy.calls] == [100, 101]
    assert [proposal.committed_length for proposal in result.buffer.replans] == [2, 2]
    assert result.buffer.executed_replan_indices == [0, 0, 1]


def test_worker_seed_sequence_shards_without_overlap() -> None:
    worker_zero = worker_seed_sequence(
        seed_start=0, episode_count=4, worker_id=0, num_workers=4
    )
    worker_three = worker_seed_sequence(
        seed_start=0, episode_count=4, worker_id=3, num_workers=4
    )

    assert worker_zero == [0, 4, 8, 12]
    assert worker_three == [3, 7, 11, 15]
