import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, TextIO

import tomllib


class UnsafeWorkspacePath(ValueError):
    """Raised when a writable path escapes the project artifact root."""


class RegistryError(ValueError):
    """Raised when the asset registry violates workspace ownership rules."""


@dataclass(frozen=True)
class AssetStatus:
    name: str
    path: Path
    resolved_path: Path
    exists: bool
    is_symlink: bool
    read_only: bool


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def validate_output_root(
    output_path: Path,
    *,
    artifact_root: Path,
    protected_roots: tuple[Path, ...],
) -> Path:
    resolved_output = _resolved(output_path)
    resolved_artifact_root = _resolved(artifact_root)
    if not resolved_output.is_relative_to(resolved_artifact_root):
        raise UnsafeWorkspacePath(
            f"output path must be inside {resolved_artifact_root}: {resolved_output}"
        )
    for protected_root in protected_roots:
        resolved_protected_root = _resolved(protected_root)
        if resolved_output.is_relative_to(resolved_protected_root):
            raise UnsafeWorkspacePath(
                f"output path overlaps protected root {resolved_protected_root}: "
                f"{resolved_output}"
            )
    return resolved_output


def load_registry(registry_path: Path) -> dict[str, Any]:
    with registry_path.open("rb") as registry_file:
        return tomllib.load(registry_file)


def validate_registry(registry: dict[str, Any]) -> None:
    for asset in registry.get("assets", []):
        if asset.get("owner") == "student" and asset.get("read_only") is not True:
            raise RegistryError(
                f"student asset must be read-only: {asset.get('name', '<unnamed>')}"
            )


def inspect_assets(registry: dict[str, Any]) -> list[AssetStatus]:
    statuses = []
    for asset in registry.get("assets", []):
        path = Path(asset["path"])
        statuses.append(
            AssetStatus(
                name=asset["name"],
                path=path,
                resolved_path=path.expanduser().resolve(strict=False),
                exists=path.exists(),
                is_symlink=path.is_symlink(),
                read_only=asset["read_only"],
            )
        )
    return statuses


def run_check(registry_path: Path, *, output: TextIO) -> int:
    registry = load_registry(registry_path)
    try:
        validate_registry(registry)
    except RegistryError as error:
        print(f"workspace status=invalid error={error}", file=output)
        return 2
    workspace = registry["workspace"]
    artifact_root = Path(workspace["artifact_root"])
    protected_roots = tuple(Path(path) for path in workspace["protected_roots"])

    try:
        resolved_artifact_root = validate_output_root(
            artifact_root,
            artifact_root=artifact_root,
            protected_roots=protected_roots,
        )
    except UnsafeWorkspacePath as error:
        print(f"workspace status=unsafe error={error}", file=output)
        return 2

    print(
        f"workspace status=safe artifact_root={resolved_artifact_root}", file=output
    )
    for status in inspect_assets(registry):
        print(
            f"asset name={status.name} exists={str(status.exists).lower()} "
            f"read_only={str(status.read_only).lower()} path={status.path} "
            f"resolved={status.resolved_path}",
            file=output,
        )
    return 0


def main(argv: Sequence[str] | None = None, *, output: TextIO | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the Giga-WAM-RL workspace")
    subparsers = parser.add_subparsers(dest="command", required=True)
    check_parser = subparsers.add_parser("check", help="inspect configured assets")
    check_parser.add_argument("--config", required=True, type=Path)
    arguments = parser.parse_args(argv)

    if output is None:
        output = sys.stdout
    if arguments.command == "check":
        return run_check(arguments.config, output=output)
    raise AssertionError(f"unsupported command: {arguments.command}")


if __name__ == "__main__":
    raise SystemExit(main())
