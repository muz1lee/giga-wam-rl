import socket
import threading
import time

import numpy as np

from giga_wam_rl.gwp05_policy_server import serve_policy
from giga_wam_rl.gwp05_policy_rpc import (
    RemoteGWP05ActionPolicy,
    decode_request,
    encode_request,
    encode_response,
    receive_message,
    send_message,
)
from giga_wam_rl.robotwin_collection import PolicyPrediction


def _cameras() -> dict[str, np.ndarray]:
    return {
        "cam_high": np.zeros((12, 16, 3), dtype=np.uint8),
        "cam_left_wrist": np.ones((12, 16, 3), dtype=np.uint8),
        "cam_right_wrist": np.full((12, 16, 3), 2, dtype=np.uint8),
    }


def test_request_codec_is_pickle_free_and_round_trips() -> None:
    payload = encode_request(cameras=_cameras(), state=np.arange(14), seed=9)
    request = decode_request(payload)

    assert request.seed == 9
    assert request.state.shape == (14,)
    assert request.cameras["cam_high"].dtype == np.uint8


def test_length_prefixed_message_round_trip() -> None:
    sender, receiver = socket.socketpair()
    try:
        send_message(sender, b"hello")
        assert receive_message(receiver) == b"hello"
    finally:
        sender.close()
        receiver.close()


def test_remote_policy_uses_rpc_response() -> None:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]

    def server() -> None:
        connection, _ = listener.accept()
        with connection:
            request = decode_request(receive_message(connection))
            assert request.seed == 123
            send_message(
                connection,
                encode_response(
                    normalized_action=np.zeros((48, 16), dtype=np.float32),
                    physical_action=np.ones((48, 14), dtype=np.float32),
                    inference_time_s=0.75,
                    seed=request.seed,
                ),
            )
        listener.close()

    thread = threading.Thread(target=server)
    thread.start()
    policy = RemoteGWP05ActionPolicy(host="127.0.0.1", port=port, timeout_s=2)
    prediction = policy.predict(cameras=_cameras(), state=np.zeros(14), seed=123)
    thread.join(timeout=2)

    assert prediction.seed == 123
    assert prediction.inference_time_s == 0.75
    assert np.array_equal(prediction.physical_action, np.ones((48, 14)))
    assert not thread.is_alive()


def test_policy_server_serves_one_request() -> None:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()

    class FakePolicy:
        def predict(
            self,
            *,
            cameras: dict[str, np.ndarray],
            state: np.ndarray,
            seed: int,
        ) -> PolicyPrediction:
            return PolicyPrediction(
                normalized_action=np.zeros((48, 16), dtype=np.float32),
                physical_action=np.full((48, 14), 3, dtype=np.float32),
                inference_time_s=0.25,
                seed=seed,
            )

    completed: list[int] = []

    def server() -> None:
        completed.append(
            serve_policy(
                policy=FakePolicy(),
                host="127.0.0.1",
                port=port,
                max_requests=1,
            )
        )

    thread = threading.Thread(target=server)
    thread.start()
    policy = RemoteGWP05ActionPolicy(host="127.0.0.1", port=port, timeout_s=2)
    deadline = time.monotonic() + 2
    while True:
        try:
            prediction = policy.predict(
                cameras=_cameras(), state=np.zeros(14), seed=456
            )
            break
        except ConnectionRefusedError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.01)
    thread.join(timeout=2)

    assert completed == [1]
    assert prediction.seed == 456
    assert np.array_equal(prediction.physical_action, np.full((48, 14), 3))
    assert not thread.is_alive()
