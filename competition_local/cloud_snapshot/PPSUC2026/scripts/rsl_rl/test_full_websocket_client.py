"""Exercise the official WebSocket transport with a local deterministic server."""
from contextlib import contextmanager
import threading

import numpy as np
from websockets.sync.server import serve
from openpi_client import msgpack_numpy
from full_client import PROMPT, RemoteFullPolicy

META = {"action_schema": "b2w_piper_two_phase", "action_dim": 14, "phase_action_semantics": "next_step_latched_v1", "default_prompt": PROMPT}

@contextmanager
def endpoint(meta, response):
    received = []
    def handler(ws):
        ws.send(msgpack_numpy.packb(meta))
        for data in ws:
            received.append(msgpack_numpy.unpackb(data))
            ws.send(response if isinstance(response, str) else msgpack_numpy.packb(response))
    with serve(handler, "127.0.0.1", 0) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield server.socket.getsockname()[1], received
        finally:
            server.shutdown()
            thread.join(timeout=5)
            assert not thread.is_alive()

with endpoint(META, {"actions": np.zeros((50, 14)), "server_timing": {"infer_ms": 1}}) as (port, received):
    p = RemoteFullPolicy("127.0.0.1", port)
    try:
        out = p.infer({"prompt": "legacy stage", "observation/state": np.zeros(13), "observation/task_phase": np.int32(0)})
        assert out["actions"].shape == (50, 14)
        assert out["server_timing"]["infer_ms"] == 1
        assert received[0]["prompt"] == PROMPT
        assert received[0]["observation/task_phase"] == 0
        p.reset()
    finally:
        p.close()
for response, error in [({"actions": np.zeros((50, 4))}, ValueError),
                        ({"actions": np.full((50, 14), np.nan)}, ValueError),
                        ("policy failure", RuntimeError)]:
    with endpoint(META, response) as (port, _):
        p = RemoteFullPolicy("127.0.0.1", port)
        try:
            try:
                p.infer({})
            except error:
                pass
            else:
                raise AssertionError("Invalid server reply accepted")
        finally:
            p.close()
for bad in [{**META, "action_dim": 13}, {**META, "phase_action_semantics": "current_phase"}, [], {**META, "action_dim": 4}, {**META, "default_prompt": "stage"}]:
    with endpoint(bad, {"actions": np.zeros((50, 14))}) as (port, _):
        try:
            RemoteFullPolicy("127.0.0.1", port)
        except ValueError:
            pass
        else:
            raise AssertionError("Incompatible metadata accepted")
print("PASS official WebSocket transport, fixed prompt, metadata, shape/finite checks, server errors and close")
