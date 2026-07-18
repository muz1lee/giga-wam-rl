# Giga-WAM-RL Project Rules

## Ownership boundary

- `/home/wjh` and `/mnt/data/wjh` are student-owned, read-only external assets.
- Their resolved physical paths are also read-only.
- Do not edit, move, delete, re-encode in place, change permissions, or run Git write operations in those trees.
- Do not stop or replace student-owned model servers without explicit confirmation from the PI and the student.

## Our writable roots

- Source checkout on the server: `/home/knowin-wenqian/giga-wam-rl`
- Persistent artifacts: `/mnt/nas/wenqian/giga-wam-rl`
- Do not place new runs, models, converted data, or caches under `/mnt/data`.
- Use the project NAS `tmp` directory instead of shared `/tmp`.

## Development discipline

- Keep code, configuration, tests, manifests, and small reports in Git.
- Keep models, raw/converted data, caches, videos, and run outputs out of Git.
- Add behavior with a failing test first and run the relevant suite after each change.
- Pin external repositories and model revisions before using them in an experiment.
- Treat the asset registry as provenance only; verify schema and model compatibility separately.
