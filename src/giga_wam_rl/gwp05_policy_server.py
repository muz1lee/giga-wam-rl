from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import tomllib
from typing import Any, Sequence

from giga_wam_rl.gwp05_action_policy import load_gwp05_action_policy
from giga_wam_rl.gwp05_policy_rpc import (
    decode_request,
    encode_response,
    receive_message,
    send_message,
)


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as input_file:
        return tomllib.load(input_file)


def serve_policy(
    *,
    policy: Any,
    host: str,
    port: int,
    max_requests: int | None,
) -> int:
    if host not in {"127.0.0.1", "localhost"}:
        raise ValueError("GWP collection policy server must bind to localhost")
    completed = 0
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((host, port))
        listener.listen(8)
        print(
            json.dumps({"event": "policy_server_ready", "host": host, "port": port}),
            flush=True,
        )
        while max_requests is None or completed < max_requests:
            connection, address = listener.accept()
            with connection:
                request = decode_request(receive_message(connection))
                prediction = policy.predict(
                    cameras=request.cameras,
                    state=request.state,
                    seed=request.seed,
                )
                send_message(
                    connection,
                    encode_response(
                        normalized_action=prediction.normalized_action,
                        physical_action=prediction.physical_action,
                        inference_time_s=prediction.inference_time_s,
                        seed=prediction.seed,
                    ),
                )
            completed += 1
            print(
                json.dumps(
                    {
                        "event": "policy_request_complete",
                        "request_index": completed - 1,
                        "client": address[0],
                        "seed": request.seed,
                        "inference_time_s": prediction.inference_time_s,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return completed


def main(argv: Sequence[str] | None = None) -> int:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Serve GWP-0.5 actions locally")
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root / "configs/rollouts/place_bread_gwp05_clean.toml",
    )
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-requests", type=int)
    arguments = parser.parse_args(argv)
    config = _load_toml(arguments.config)
    robotwin_config = config["robotwin"]
    policy_config = config["policy"]
    host = arguments.host or str(policy_config["rpc_host"])
    port = arguments.port or int(policy_config["rpc_port"])
    policy = load_gwp05_action_policy(
        checkpoint=Path(policy_config["checkpoint"]).resolve(strict=True),
        base_model=Path(policy_config["base_model"]).resolve(strict=True),
        upstream_root=Path(policy_config["upstream_root"]).resolve(strict=True),
        norm_stats_path=Path(policy_config["norm_stats"]).resolve(strict=True),
        prompt=str(robotwin_config["instruction"]),
        device_name=arguments.device,
        num_inference_steps=int(policy_config["num_inference_steps"]),
        clip_normalized_actions=bool(policy_config["clip_normalized_actions"]),
        compile_transformer=bool(policy_config["compile_transformer"]),
    )
    completed = serve_policy(
        policy=policy,
        host=host,
        port=port,
        max_requests=arguments.max_requests,
    )
    print(json.dumps({"event": "policy_server_stopped", "requests": completed}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
