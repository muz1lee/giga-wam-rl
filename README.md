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

Pinned upstream source/model revisions are recorded in [`configs/upstreams.toml`](configs/upstreams.toml). The validated model/data interface is in [`configs/gwp05_contract.toml`](configs/gwp05_contract.toml). The upstream source is checked out under the ignored `external/` directory and is never edited in place. See the [interface audit](docs/gwp05-interface-audit-2026-07-19.md) and [GPU smoke results](docs/gwp05-smoke-results-2026-07-19.md).

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

The isolated server smoke environment is `/home/knowin-wenqian/giga-wam-rl/.venv`. Reproduce the validated transformer and VAE contracts with:

```bash
export CUDA_VISIBLE_DEVICES=3

PYTHONPATH=src .venv/bin/python -m giga_wam_rl.gwp05_smoke \
  --checkpoint /mnt/nas/wenqian/giga-wam-rl/models/Giga-World-Policy-0.5 \
  --device cuda:0

PYTHONPATH=src .venv/bin/python -m giga_wam_rl.gwp05_vae_smoke \
  --base-model /mnt/nas/wenqian/giga-wam-rl/models/Wan2.2-TI2V-5B-Diffusers \
  --device cuda:0
```

## Initial research sequence

1. Completed: strict-load the pinned transformer, run a joint forward, reject the stale 32D path, and validate Wan VAE encode/decode.
2. Verify the 14D physical action semantics, units, coordinate frame, and control rate; model inputs are padded from 14D to the validated 16D checkpoint contract.
3. Convert only 3–5 raw HDF5/rollout samples and validate camera order, timestamps, a 48-step action chunk, and five future frames.
4. Add a future-only sampler for action-conditioned counterfactual rollout and test whether imagined-future rankings are calibrated.
5. Port only the necessary rollout, reward, advantage, PPO, and GRPO utilities from FastWAM-RL after the counterfactual probe passes.

No data conversion, training, rollout, GPU process termination, or shared-storage cleanup has been performed yet.
