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

The raw Place Bread contract and three-episode LeRobot v3 pilot are described in the [data pilot report](docs/place-bread-data-pilot-2026-07-19.md). A small provenance record is tracked in [`manifests/place_bread_gwp05_pilot.json`](manifests/place_bread_gwp05_pilot.json); the generated parquet, videos, and detailed manifests stay on NAS.

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

## Place Bread data pilot

The model smoke environment keeps `diffusers==0.36.0`. LeRobot 0.4.4 requires `diffusers<0.36`, so data conversion uses a separate environment while sharing the already installed Torch packages:

```bash
.venv/bin/python -m venv --system-site-packages .venv-convert
.venv-convert/bin/python -m pip install -r requirements-lerobot-convert.txt
```

Probe the student-owned raw HDF5 read-only:

```bash
PYTHONPATH=src .venv/bin/python -m giga_wam_rl.raw_hdf5_probe \
  --config configs/datasets/place_bread_raw_hdf5.toml \
  --registry configs/assets.server.toml
```

Create the three-episode pilot in our NAS namespace:

```bash
PYTHONPATH=src .venv-convert/bin/python -m giga_wam_rl.lerobot_pilot \
  --config configs/datasets/place_bread_raw_hdf5.toml \
  --registry configs/assets.server.toml
```

The converter intentionally refuses to overwrite either configured output path. Change to a fresh output path when reproducing alongside an existing pilot.

## Initial research sequence

1. Completed: strict-load the pinned transformer, run a joint forward, reject the stale 32D path, and validate Wan VAE encode/decode.
2. Completed for the demo pilot: verify the 14D joint-target order and causal alignment; pad the model input from 14D to the validated 16D checkpoint contract. Exact 250 Hz issued setpoints are not present in this HDF5.
3. Completed: convert episodes 0, 25, and 49 to LeRobot v3 and validate camera order/color, a 48-step action chunk, and five visual observations through the LeRobot loader.
4. Add a future-only sampler for action-conditioned counterfactual rollout and test whether imagined-future rankings are calibrated.
5. Port only the necessary rollout, reward, advantage, PPO, and GRPO utilities from FastWAM-RL after the counterfactual probe passes.

Only the three-episode pilot has been converted. No full-dataset conversion, training, model rollout, or student GPU process termination has been performed.
