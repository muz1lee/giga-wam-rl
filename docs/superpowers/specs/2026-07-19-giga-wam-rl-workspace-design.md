# Giga-WAM-RL Workspace Design

## 1. Goal

Build an independent research workspace for GigaWorld-Policy-0.5 world-action-model reinforcement learning while reusing the student's existing demonstrations, rollouts, checkpoints, and FastWAM-RL algorithm work wherever they are compatible.

The workspace must make it impossible by default for our experiments to modify the student's code or data.

## 2. Constraints

- Treat `/home/wjh` and `/mnt/data/wjh` as read-only external assets.
- Do not run training, stop the student's model servers, clean shared storage, or convert the full dataset during bootstrap.
- Keep source code and small reproducible configuration in Git.
- Keep checkpoints, converted datasets, caches, and runs out of Git.
- Do not use `/mnt/data` for new large outputs because it is 96% full.
- Do not use the shared `/tmp`, which is currently full; use a project-specific temporary directory.
- Assume `/home` may be tied to the pod lifecycle, so unpushed code is not a durable copy.

## 3. Options Considered

### A. Hybrid code and artifact layout — selected

- Local canonical checkout: `01_wam_rl_research/giga-wam-rl`
- Server working checkout: `/home/knowin-wenqian/giga-wam-rl`
- Persistent artifacts: `/mnt/nas/wenqian/giga-wam-rl`

This keeps code operations fast while putting large persistent assets on NAS. It requires Git synchronization between the local and server checkouts.

### B. NAS-only workspace

Put code, environments, data, and runs under `/mnt/nas`. This simplifies persistence but can make dependency installation and small-file workloads slower, and couples code development to NFS behavior.

### C. Extend the student's workspace

Develop under `/home/wjh/lingbot-va` or `/mnt/data/wjh`. This offers the shortest import paths but is rejected because the student's repository is dirty, actively serving models, and lacks an ownership boundary.

## 4. Selected Architecture

### 4.1 Source repository

The repository owns only our adapters, validation tools, experiments, and documentation:

```text
giga-wam-rl/
├── AGENTS.md
├── README.md
├── pyproject.toml
├── configs/
│   ├── assets.server.toml
│   └── experiments/
├── docs/
├── scripts/
├── src/giga_wam_rl/
│   ├── data/
│   ├── models/
│   ├── rewards/
│   ├── rl/
│   └── rollout/
└── tests/
```

The first executable component is a workspace validator. It parses the asset registry, rejects unsafe output paths, and reports which read-only student assets are present. It does not load models or mutate assets.

### 4.2 Persistent artifact root

```text
/mnt/nas/wenqian/giga-wam-rl/
├── artifacts/
├── cache/
├── datasets/
│   ├── converted/
│   └── manifests/
├── models/
├── runs/
└── tmp/
```

Only our processes may write below this root. `TMPDIR` and model caches will point into this tree rather than shared `/tmp` or the student's directories.

### 4.3 External asset registry

`configs/assets.server.toml` records external paths and ownership. Initial entries cover:

- 50-demo place-bread dataset;
- RobotWin evaluation and failure rollouts;
- the small rollout-review audit set;
- LingBot RobotWin checkpoint;
- RECAP SigLIP checkpoint;
- student LingBot source checkout;
- our NAS output root.

Every student-owned entry is marked `read_only = true`. The validator rejects an output root that is equal to or nested under `/home/wjh` or `/mnt/data/wjh`, including their resolved physical paths.

The registry is provenance, not a promise of model compatibility. Wan latent compatibility and the larger RoboTwin 50-by-50 dataset remain validation tasks.

## 5. Initial Data Flow

```text
student assets (read-only)
        |
        v
asset/path validation
        |
        v
96-transition rollout_review probe
        |
        v
our adapters and converted manifests on NAS
        |
        v
Giga action-conditioned future forward test
```

The first data experiment uses the 96-transition audit set. Full conversion of `/mnt/data/wjh/robotwin_eval` begins only after action schema, camera ordering, chunk length, coordinate frame, and latent compatibility are validated.

## 6. Reuse Boundaries

### FastWAM-RL

Reuse algorithmic infrastructure such as rollout orchestration, reward aggregation, group sampling, PPO/GRPO utilities, logging, and tests through explicit ports or pinned modules. Do not make the new project depend on an unpublished mutable checkout.

### LingBot

Use the current checkpoint and server only as external baselines or data-generation services after coordinating availability with the student. Do not import the dirty LingBot source tree as a writable package.

### GigaWorld-Policy-0.5

Download code and checkpoints into our namespace and pin exact revisions. The existing local `refs/giga-world-policy` checkout is the original project and must not be mislabeled as version 0.5.

## 7. Failure Handling

- Missing read-only assets are reported, not created or repaired automatically.
- Broken rollout symlinks are reported with both logical and resolved paths.
- Unsafe output configuration fails before any directory or file is written.
- Model and latent incompatibility produces a validation report; it does not trigger an automatic full-data re-encode.
- Existing GPU processes are never terminated by bootstrap or validation commands.

## 8. Verification

Bootstrap is complete when:

1. the local repository has a clean committed baseline;
2. the server checkout exists only under `/home/knowin-wenqian`;
3. all writable artifact directories exist only under our NAS root;
4. tests prove that student paths cannot be selected as output roots;
5. the server validator reports asset presence without modifying them;
6. Git status is clean in both our local and server checkouts;
7. student repository status and timestamps are unchanged by our work.

## 9. Non-goals for Bootstrap

- No Giga model download.
- No dataset conversion.
- No training or rollout.
- No GPU process termination.
- No cleanup of `/tmp`, `/mnt/data`, student runs, or checkpoints.
- No architectural refactor of FastWAM-RL or LingBot.
