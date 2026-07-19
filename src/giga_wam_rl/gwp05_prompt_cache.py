from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import tempfile
from typing import Any, Sequence

from giga_wam_rl.gwp05_pilot_counterfactual import _encode_prompt
from giga_wam_rl.workspace import (
    load_registry,
    validate_output_root,
    validate_registry,
)


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_prompt_cache(
    *,
    embedding: Any,
    output_path: Path,
    prompt: str,
    base_model: str,
) -> dict[str, Any]:
    import torch

    output_path = Path(output_path)
    manifest_path = output_path.with_suffix(output_path.suffix + ".json")
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite prompt cache: {output_path}")
    tensor = torch.as_tensor(embedding).detach().cpu()
    if tuple(tensor.shape) == (1, 64, 4096):
        tensor = tensor[0]
    if tuple(tensor.shape) != (64, 4096):
        raise ValueError("prompt embedding must have shape [64,4096]")
    tensor = tensor.to(dtype=torch.bfloat16).contiguous()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        torch.save({"t5_embedding": tensor}, temporary_path)
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "prompt": prompt,
        "base_model": base_model,
        "path": str(output_path),
        "shape": [64, 4096],
        "dtype": "bfloat16",
        "sha256": _sha256_file(output_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def run_cache(
    *,
    registry_path: Path,
    base_model: Path,
    output_path: Path,
    prompt: str,
    device_name: str,
) -> dict[str, Any]:
    import torch

    registry = load_registry(registry_path)
    validate_registry(registry)
    workspace = registry["workspace"]
    safe_output = validate_output_root(
        output_path,
        artifact_root=Path(workspace["artifact_root"]),
        protected_roots=tuple(Path(path) for path in workspace["protected_roots"]),
    )
    base_model = base_model.resolve(strict=True)
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("prompt cache generation requires CUDA")
    torch.cuda.set_device(device)
    embedding = _encode_prompt(
        base_model,
        prompt,
        device=device,
        dtype=torch.bfloat16,
    )
    return write_prompt_cache(
        embedding=embedding,
        output_path=safe_output,
        prompt=prompt,
        base_model=str(base_model),
    )


def main(argv: Sequence[str] | None = None) -> int:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Cache one fixed GWP T5 prompt")
    parser.add_argument(
        "--registry",
        type=Path,
        default=project_root / "configs/assets.server.toml",
    )
    parser.add_argument("--base-model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--device", default="cuda:0")
    arguments = parser.parse_args(argv)
    manifest = run_cache(
        registry_path=arguments.registry,
        base_model=arguments.base_model,
        output_path=arguments.output,
        prompt=arguments.prompt,
        device_name=arguments.device,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
