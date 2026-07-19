import json

import pytest
import torch

from giga_wam_rl.gwp05_prompt_cache import write_prompt_cache


def test_write_prompt_cache_squeezes_batch_and_records_provenance(tmp_path) -> None:
    output_path = tmp_path / "place_bread.pt"
    embedding = torch.zeros((1, 64, 4096), dtype=torch.bfloat16)

    manifest = write_prompt_cache(
        embedding=embedding,
        output_path=output_path,
        prompt="Pick up the bread and place it in the basket.",
        base_model="/models/wan",
    )

    payload = torch.load(output_path, map_location="cpu", weights_only=True)
    assert payload["t5_embedding"].shape == (64, 4096)
    assert payload["t5_embedding"].dtype == torch.bfloat16
    assert manifest["shape"] == [64, 4096]
    assert manifest["sha256"]
    assert json.loads(output_path.with_suffix(".pt.json").read_text()) == manifest


def test_write_prompt_cache_refuses_overwrite(tmp_path) -> None:
    output_path = tmp_path / "place_bread.pt"
    embedding = torch.zeros((64, 4096), dtype=torch.bfloat16)
    write_prompt_cache(
        embedding=embedding,
        output_path=output_path,
        prompt="prompt",
        base_model="/models/wan",
    )

    with pytest.raises(FileExistsError):
        write_prompt_cache(
            embedding=embedding,
            output_path=output_path,
            prompt="prompt",
            base_model="/models/wan",
        )


def test_write_prompt_cache_rejects_wrong_shape(tmp_path) -> None:
    with pytest.raises(ValueError, match=r"\[64,4096\]"):
        write_prompt_cache(
            embedding=torch.zeros((32, 4096), dtype=torch.bfloat16),
            output_path=tmp_path / "bad.pt",
            prompt="prompt",
            base_model="/models/wan",
        )
