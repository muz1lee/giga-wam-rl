# Giga-WAM-RL

Independent research workspace for reinforcement learning with GigaWorld-Policy-style world-action models. The project reuses student demonstrations and rollout assets through read-only paths while keeping all new code and artifacts in our namespace.

## Ownership layout

```text
local source:       01_wam_rl_research/giga-wam-rl
server source:      /home/knowin-wenqian/giga-wam-rl
persistent output:  /mnt/nas/wenqian/giga-wam-rl
student assets:     /home/wjh and /mnt/data/wjh (read-only)
```

The server asset registry is [`configs/assets.server.toml`](configs/assets.server.toml). An entry means the asset was located; it does not mean that its latent format, camera order, coordinate frame, or action chunks are already compatible with GigaWorld-Policy-0.5.

## Check the workspace

Python 3.11 or newer is required. The validator uses only the standard library and never creates missing asset paths.

```bash
PYTHONPATH=src python -m giga_wam_rl.workspace check \
  --config configs/assets.server.toml
```

Run tests with:

```bash
python -m unittest discover -s tests -v
```

## Server environment

Set caches for a project shell or job rather than modifying global shell configuration:

```bash
export GIGA_WAM_RL_ARTIFACT_ROOT=/mnt/nas/wenqian/giga-wam-rl
export TMPDIR="$GIGA_WAM_RL_ARTIFACT_ROOT/tmp"
export HF_HOME="$GIGA_WAM_RL_ARTIFACT_ROOT/cache/huggingface"
export TORCH_HOME="$GIGA_WAM_RL_ARTIFACT_ROOT/cache/torch"
export WANDB_DIR="$GIGA_WAM_RL_ARTIFACT_ROOT/runs/wandb"
```

## Initial research sequence

1. Validate the 96-transition rollout-review set and the 16-dimensional action schema.
2. Pin and run a GigaWorld-Policy-0.5 action-conditioned future forward pass.
3. Determine whether LingBot/Wan latents are compatible; re-encode from raw observations only if required.
4. Port only the necessary rollout, reward, PPO, and GRPO utilities from FastWAM-RL.
5. Expand to the full RobotWin evaluation/failure data only after the small probe passes.

The current bootstrap intentionally performs no model download, data conversion, training, rollout, GPU process termination, or shared-storage cleanup.
