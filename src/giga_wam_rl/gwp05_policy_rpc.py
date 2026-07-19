from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import socket
import struct

import numpy as np

from giga_wam_rl.robotwin_collection import PolicyPrediction


MAX_MESSAGE_BYTES = 64 * 1024 * 1024
HEADER = struct.Struct("!Q")


@dataclass(frozen=True)
class PolicyRequest:
    cameras: dict[str, np.ndarray]
    state: np.ndarray
    seed: int


def _npz_bytes(**arrays: np.ndarray) -> bytes:
    output = BytesIO()
    np.savez(output, **arrays)
    return output.getvalue()


def encode_request(
    *, cameras: dict[str, np.ndarray], state: np.ndarray, seed: int
) -> bytes:
    return _npz_bytes(
        protocol_version=np.asarray([1], dtype=np.int32),
        cam_high=np.asarray(cameras["cam_high"], dtype=np.uint8),
        cam_left_wrist=np.asarray(cameras["cam_left_wrist"], dtype=np.uint8),
        cam_right_wrist=np.asarray(cameras["cam_right_wrist"], dtype=np.uint8),
        state=np.asarray(state, dtype=np.float32),
        seed=np.asarray([seed], dtype=np.int64),
    )


def decode_request(payload: bytes) -> PolicyRequest:
    with np.load(BytesIO(payload), allow_pickle=False) as arrays:
        if arrays["protocol_version"].tolist() != [1]:
            raise ValueError("unsupported GWP policy RPC version")
        cameras = {
            "cam_high": arrays["cam_high"].astype(np.uint8, copy=True),
            "cam_left_wrist": arrays["cam_left_wrist"].astype(np.uint8, copy=True),
            "cam_right_wrist": arrays["cam_right_wrist"].astype(np.uint8, copy=True),
        }
        state = arrays["state"].astype(np.float32, copy=True)
        seed = int(arrays["seed"][0])
    if state.shape != (14,):
        raise ValueError("RPC state must have shape [14]")
    for name, image in cameras.items():
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"RPC {name} must have shape [H,W,3]")
    return PolicyRequest(cameras=cameras, state=state, seed=seed)


def encode_response(
    *,
    normalized_action: np.ndarray,
    physical_action: np.ndarray,
    inference_time_s: float,
    seed: int,
) -> bytes:
    return _npz_bytes(
        protocol_version=np.asarray([1], dtype=np.int32),
        normalized_action=np.asarray(normalized_action, dtype=np.float32),
        physical_action=np.asarray(physical_action, dtype=np.float32),
        inference_time_s=np.asarray([inference_time_s], dtype=np.float64),
        seed=np.asarray([seed], dtype=np.int64),
    )


def decode_response(payload: bytes) -> PolicyPrediction:
    with np.load(BytesIO(payload), allow_pickle=False) as arrays:
        if arrays["protocol_version"].tolist() != [1]:
            raise ValueError("unsupported GWP policy RPC version")
        normalized = arrays["normalized_action"].astype(np.float32, copy=True)
        physical = arrays["physical_action"].astype(np.float32, copy=True)
        inference_time_s = float(arrays["inference_time_s"][0])
        seed = int(arrays["seed"][0])
    if normalized.shape != (48, 16) or physical.shape != (48, 14):
        raise ValueError("RPC policy response violates the GWP action contract")
    return PolicyPrediction(
        normalized_action=normalized,
        physical_action=physical,
        inference_time_s=inference_time_s,
        seed=seed,
    )


def _receive_exact(connection: socket.socket, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise ConnectionError("policy RPC connection closed early")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_message(connection: socket.socket, payload: bytes) -> None:
    if not 0 < len(payload) <= MAX_MESSAGE_BYTES:
        raise ValueError("policy RPC message size is invalid")
    connection.sendall(HEADER.pack(len(payload)) + payload)


def receive_message(connection: socket.socket) -> bytes:
    (size,) = HEADER.unpack(_receive_exact(connection, HEADER.size))
    if not 0 < size <= MAX_MESSAGE_BYTES:
        raise ValueError("policy RPC message size is invalid")
    return _receive_exact(connection, size)


class RemoteGWP05ActionPolicy:
    def __init__(self, *, host: str, port: int, timeout_s: float) -> None:
        if not host or not 0 < port < 65536 or timeout_s <= 0:
            raise ValueError("invalid policy RPC endpoint")
        self.host = host
        self.port = int(port)
        self.timeout_s = float(timeout_s)

    def predict(
        self,
        *,
        cameras: dict[str, np.ndarray],
        state: np.ndarray,
        seed: int,
    ) -> PolicyPrediction:
        request = encode_request(cameras=cameras, state=state, seed=seed)
        with socket.create_connection(
            (self.host, self.port), timeout=self.timeout_s
        ) as connection:
            connection.settimeout(self.timeout_s)
            send_message(connection, request)
            response = receive_message(connection)
        prediction = decode_response(response)
        if prediction.seed != int(seed):
            raise ValueError("policy RPC response seed does not match request")
        return prediction
