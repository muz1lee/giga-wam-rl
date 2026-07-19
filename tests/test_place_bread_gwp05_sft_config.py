from pathlib import Path
import runpy


def test_place_bread_sft_config_fixes_action_dim_and_has_safe_defaults(
    monkeypatch,
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    upstream_root = project_root / "external" / "giga-world-policy"
    monkeypatch.setenv("GWP05_UPSTREAM_ROOT", str(upstream_root))
    monkeypatch.setenv("GWP_AGILEX_DATA_PATHS", "/datasets/place_bread")
    monkeypatch.setenv("GWP_NORM_STATS_PATH", "/datasets/norm.json")
    monkeypatch.setenv("GWP05_TRANSFORMER_PRETRAINED", "/models/gwp05")
    monkeypatch.setenv("GWP_WAN_PRETRAINED", "/models/wan")
    monkeypatch.setenv("GWP_T5_LOAD_FROM", "path")
    monkeypatch.setenv("GWP_T5_EMBEDDING_PATH", "/models/prompt.pt")
    monkeypatch.setenv("GWP05_PROJECT_DIR", "/runs/place_bread")
    monkeypatch.setenv("GWP_GPU_IDS", "0")
    monkeypatch.setenv("GWP_BATCH_SIZE_PER_GPU", "1")
    monkeypatch.setenv("GWP_NUM_WORKERS", "0")

    config = runpy.run_path(
        str(project_root / "configs/training/place_bread_gwp05_sft.py")
    )["config"]

    transform = config["dataloaders"]["train"]["transform"]
    assert transform["model_action_dim"] == 16
    assert config["launch"]["until_completion"] is False
    assert config["models"]["enable_gradient_checkpointing"] is True
    assert config["train"]["max_steps"] == 2
    assert config["train"]["checkpoint_interval"] == 2
