# Giga-WAM-RL Workspace Bootstrap Plan

> The `writing-plans` skill is listed in the environment but its instruction file is unavailable. This document follows the same test-first, file-specific implementation structure as a fallback.

**Goal:** Create an isolated, reproducible project skeleton that can inspect student-owned assets without writing to them and routes every new artifact to `/mnt/nas/wenqian/giga-wam-rl`.

**Architecture:** A standard-library Python package reads a TOML asset registry. A pure path-safety layer rejects protected output roots before any caller can write, while a read-only CLI reports asset presence and symlink resolution. The repository is developed locally, committed to Git, and copied into a new server checkout; large writable directories live under the dedicated NAS root.

**Technology:** Python 3.11+, `tomllib`, `pathlib`, `unittest`, TOML, Git, SSH, rsync.

---

## Task 1: Record the pre-change boundary

**Read-only targets:**

- `/home/wjh/lingbot-va`
- `/home/wjh/lingbot-va/checkpoints/lingbot-va-posttrain-robotwin`
- `/mnt/data/wjh`

**Steps:**

1. Record the LingBot repository HEAD and a hash of `git status --porcelain=v1 -uall`.
2. Record the type and resolved path of the student code and data roots.
3. Confirm `/home/knowin-wenqian/giga-wam-rl` and `/mnt/nas/wenqian/giga-wam-rl` do not already exist.
4. Preserve the values for comparison after synchronization and validation.

## Task 2: Add repository metadata and the asset registry

**Files:**

- Create: `.gitignore`
- Create: `AGENTS.md`
- Create: `README.md`
- Create: `pyproject.toml`
- Create: `configs/assets.server.toml`
- Create: `src/giga_wam_rl/__init__.py`

**Steps:**

1. Declare the read-only student roots and writable NAS root explicitly.
2. Register the confirmed demonstration, rollout, review, model, and source paths.
3. Document that registry presence does not establish latent or schema compatibility.
4. Keep runtime dependencies within the Python standard library.

## Task 3: Implement path isolation using TDD

**Files:**

- Create: `tests/test_workspace.py`
- Create: `src/giga_wam_rl/workspace.py`

**RED steps:**

1. Write a test accepting an output path under the configured NAS root.
2. Run it and confirm failure because the production module does not exist.
3. Write tests rejecting paths equal to or nested below logical protected roots.
4. Write a test rejecting a path that reaches a protected root through a symlink.
5. Write a test proving asset inspection does not create a missing path.

**GREEN steps:**

1. Implement a small `UnsafeWorkspacePath` exception.
2. Implement normalized and resolved path containment checks.
3. Implement TOML registry loading and read-only asset inspection.
4. Implement a `check` CLI that emits a concise report and a nonzero code only for unsafe configuration or malformed registry data.
5. Run the full test suite after each minimal behavior is added.

## Task 4: Create only our server directories

**Create:**

- `/home/knowin-wenqian/giga-wam-rl`
- `/mnt/nas/wenqian/giga-wam-rl/artifacts`
- `/mnt/nas/wenqian/giga-wam-rl/cache`
- `/mnt/nas/wenqian/giga-wam-rl/datasets/converted`
- `/mnt/nas/wenqian/giga-wam-rl/datasets/manifests`
- `/mnt/nas/wenqian/giga-wam-rl/models`
- `/mnt/nas/wenqian/giga-wam-rl/runs`
- `/mnt/nas/wenqian/giga-wam-rl/tmp`

**Steps:**

1. Create the explicit directories without globs, deletion, permission broadening, or symlinks.
2. Synchronize the committed local repository to the new server checkout without `--delete`.
3. Set no global environment variables; document project-scoped `TMPDIR` and cache exports instead.

## Task 5: Verify locally and on the server

**Steps:**

1. Run `python -m unittest discover -s tests -v` locally.
2. Run `git diff --check` and inspect repository status.
3. Run the same tests in the server checkout.
4. Run the workspace CLI against `configs/assets.server.toml` on the server.
5. Confirm writable directories resolve under `/mnt/nas/wenqian/giga-wam-rl`.
6. Recompute the LingBot HEAD and dirty-status hash and compare them with Task 1.
7. Confirm no new files were created under `/home/wjh` or `/mnt/data/wjh` by our bootstrap.
8. Commit the verified bootstrap baseline locally and synchronize that commit to the server checkout.

## Deferred Work

- Download and pin GigaWorld-Policy-0.5.
- Validate the 96-transition rollout-review schema.
- Validate 16-dimensional action ordering, units, frames, and chunking.
- Test Wan latent compatibility.
- Port selected FastWAM-RL algorithm modules.
- Choose PPO, GRPO, or a staged combination through controlled experiments.
