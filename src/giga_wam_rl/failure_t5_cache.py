import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence, TextIO

import tomllib

from giga_wam_rl.failure_rollout_probe import _read_jsonl, _select_episode
from giga_wam_rl.gwp05_pilot_counterfactual import _prompt_clean
from giga_wam_rl.workspace import (
    load_registry,
    validate_output_root,
    validate_registry,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _episode_prompts(config: dict[str, Any]) -> list[str]:
    prompts = []
    for episode_config in config["episodes"]:
        sidecar_path = Path(episode_config["sidecar_path"]).resolve(strict=True)
        sidecar_row = _select_episode(
            _read_jsonl(sidecar_path),
            episode_index=int(episode_config["episode_index"]),
        )
        if int(sidecar_row["seed"]) != int(episode_config["seed"]):
            raise ValueError("sidecar seed differs from configured failure episode")
        prompts.append(sidecar_row["prompt"])
    return prompts


def run_cache(
    config_path: Path,
    registry_path: Path,
    *,
    base_model: Path,
    device: str,
    output: TextIO,
) -> int:
    import torch
    from transformers import AutoTokenizer, UMT5EncoderModel

    with config_path.open("rb") as config_file:
        config = tomllib.load(config_file)
    registry = load_registry(registry_path)
    validate_registry(registry)
    workspace_config = registry["workspace"]
    artifact_root = Path(workspace_config["artifact_root"])
    protected_roots = tuple(Path(path) for path in workspace_config["protected_roots"])
    conversion = config["conversion"]
    dataset_root = validate_output_root(
        Path(conversion["output_root"]),
        artifact_root=artifact_root,
        protected_roots=protected_roots,
    ).resolve(strict=True)
    manifest_path = validate_output_root(
        Path(conversion["t5_manifest"]),
        artifact_root=artifact_root,
        protected_roots=protected_roots,
    )
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite T5 manifest: {manifest_path}")
    prompts = [_prompt_clean(prompt) for prompt in _episode_prompts(config)]
    output_dir = dataset_root / "t5_embedding"
    output_paths = [
        output_dir / f"episode_{episode_index:06d}.pt"
        for episode_index in range(len(prompts))
    ]
    existing = [path for path in output_paths if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite T5 cache: {existing[0]}")

    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(
        base_model / "tokenizer", local_files_only=True
    )
    text_encoder = UMT5EncoderModel.from_pretrained(
        base_model / "text_encoder",
        torch_dtype=dtype,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    text_encoder.eval().requires_grad_(False).to(device)
    tokens = tokenizer(
        prompts,
        padding="max_length",
        max_length=512,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_ids = tokens.input_ids.to(device)
    attention_mask = tokens.attention_mask.to(device)
    with torch.inference_mode():
        hidden = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state
    embeddings = hidden[:, :64].to(dtype=dtype)
    embeddings = embeddings * attention_mask[:, :64, None].to(
        device=device, dtype=dtype
    )
    if tuple(embeddings.shape) != (len(prompts), 64, 4096):
        raise ValueError(f"unexpected T5 embedding shape: {embeddings.shape}")

    output_dir.mkdir(parents=True, exist_ok=False)
    records = []
    for episode_index, (prompt, path) in enumerate(zip(prompts, output_paths)):
        torch.save(
            {"t5_embedding": embeddings[episode_index].cpu()},
            path,
        )
        records.append(
            {
                "episode_index": episode_index,
                "prompt": prompt,
                "path": str(path),
                "shape": [64, 4096],
                "dtype": "bfloat16",
                "sha256": _sha256_file(path),
            }
        )

    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_model": str(base_model.resolve(strict=True)),
        "dataset_root": str(dataset_root),
        "records": records,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), file=output)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Cache per-episode UMT5 embeddings for the failure pilot"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--base-model", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    return run_cache(
        args.config,
        args.registry,
        base_model=args.base_model,
        device=args.device,
        output=sys.stdout,
    )


if __name__ == "__main__":
    raise SystemExit(main())
