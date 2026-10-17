"""openpi policy server (`scripts/serve_policy.py`) over its msgpack websocket.

The request dict must use the keys the server's training config repacks from: for pi05_droid
that is observation/exterior_image_1_left, observation/wrist_image_left,
observation/joint_position, observation/gripper_position and prompt. Each logical camera is sent
under image_key.format(name=camera_map[camera]).

The connection is opened here rather than through openpi-client's WebsocketClientPolicy, which
waits forever both for the server and for each reply. openpi-client still provides the msgpack
encoding and image resizing, so the bytes on the wire are the same. Every call has a timeout, and
any failure closes the connection: a reply arriving late on a reused connection would be taken as
the answer to the next request.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from ..config import from_dict
from ..types import ActionChunk, Observation, PolicyInfo
from .base import ActionMapping, PolicyBackend

log = logging.getLogger(__name__)


@dataclass
class OpenPIConfig:
    action_hz: float                # rate of the training data's actions
    camera_map: dict[str, str]      # logical camera -> policy camera name
    host: str = 'localhost'
    port: int | None = 8000
    api_key: str | None = None
    image_key: str = 'observation/{name}'
    state_key: str = 'observation/joint_position'
    gripper_key: str | None = 'observation/gripper_position'
    image_size: list[int] | None = field(default_factory=lambda: [224, 224])  # H, W; None = native
    state_dim: int | None = None    # zero-pad the joint state to this length
    chunk_len: int | None = None    # use only the first N actions of each chunk
    action_mode: str = 'absolute'
    gripper_open: float = 0.0
    gripper_closed: float = 1.0
    gripper_index: int | None = 6
    timeout_s: float = 10.0         # per inference, and per connection attempt
    warmup_timeout_s: float = 300.0 # the first inference: the server JIT-compiles the model
    wait_for_server_s: float = 120.0  # connect() keeps retrying this long (server still loading)


class OpenPIBackend(PolicyBackend):
    def __init__(self, **params):
        self.cfg = from_dict(OpenPIConfig, params, 'policy (openpi)')
        c = self.cfg
        self.mapping = ActionMapping(c.action_mode, c.gripper_open, c.gripper_closed, c.gripper_index)
        self._ws = None
        self._metadata: dict = {}
        self._warm = False

    @property
    def uri(self) -> str:
        uri = self.cfg.host if self.cfg.host.startswith('ws') else f'ws://{self.cfg.host}'
        return uri if self.cfg.port is None else f'{uri}:{self.cfg.port}'

    def connect(self) -> PolicyInfo:
        deadline = time.monotonic() + self.cfg.wait_for_server_s
        while True:
            try:
                self._open()
                break
            except OSError as exc:      # refused / unreachable / timed out
                if time.monotonic() > deadline:
                    raise ConnectionError(f'openpi: no server at {self.uri} after '
                                          f'{self.cfg.wait_for_server_s:.0f} s') from exc
                log.info('openpi: waiting for server at %s (%s)', self.uri, exc)
                time.sleep(2.0)
        return PolicyInfo('openpi', self.cfg.chunk_len, 1.0 / self.cfg.action_hz,
                          list(self.cfg.camera_map), {'server': self._metadata})

    def _open(self) -> None:
        from openpi_client import msgpack_numpy
        from websockets.sync.client import connect
        headers = {'Authorization': f'Api-Key {self.cfg.api_key}'} if self.cfg.api_key else None
        ws = connect(self.uri, compression=None, max_size=None, additional_headers=headers,
                     open_timeout=self.cfg.timeout_s, legacy=True)
        try:
            self._metadata = msgpack_numpy.unpackb(ws.recv(timeout=self.cfg.timeout_s))
        except Exception:
            ws.close()
            raise
        self._ws = ws
        self._packer = msgpack_numpy.Packer()

    def build_request(self, obs: Observation) -> dict:
        from openpi_client import image_tools
        c = self.cfg
        req: dict = {}
        for cam, policy_name in c.camera_map.items():
            img = obs.images[cam].rgb
            if c.image_size is not None:
                img = image_tools.resize_with_pad(img, c.image_size[0], c.image_size[1])
            req[c.image_key.format(name=policy_name)] = image_tools.convert_to_uint8(img)
        state = np.asarray(obs.q, dtype=np.float32)
        if c.state_dim is not None and c.state_dim > len(state):
            state = np.pad(state, (0, c.state_dim - len(state)))
        req[c.state_key] = state
        if c.gripper_key is not None:
            req[c.gripper_key] = np.array([self.mapping.gripper_to_policy(obs.gripper)], np.float32)
        req['prompt'] = obs.prompt
        return req

    def infer(self, obs: Observation) -> ActionChunk:
        from openpi_client import msgpack_numpy
        if self._ws is None:
            self._open()            # one attempt; the control loop retries with backoff
        req = self.build_request(obs)
        timeout = self.cfg.timeout_s if self._warm else self.cfg.warmup_timeout_s
        t0 = time.monotonic()
        try:
            self._ws.send(self._packer.pack(req))
            reply = self._ws.recv(timeout=timeout)
        except TimeoutError as exc:
            self.close()
            raise TimeoutError(f'openpi: no reply within {timeout:.0f} s; reconnecting') from exc
        except Exception:
            self.close()
            raise
        if isinstance(reply, str):  # the server sends a traceback as text, then closes
            self.close()
            raise RuntimeError(f'openpi server error:\n{reply}')
        self._warm = True
        out = msgpack_numpy.unpackb(reply)
        info = {'rtt_s': time.monotonic() - t0}
        for key in ('server_timing', 'policy_timing'):
            if key in out:
                info[key] = out[key]
        actions = np.asarray(out['actions'])
        if self.cfg.chunk_len is not None:
            actions = actions[: self.cfg.chunk_len]
        return self.mapping.to_chunk(actions, obs, 1.0 / self.cfg.action_hz, info)

    def reset(self) -> None:
        pass                        # the openpi server keeps no per-episode state

    def close(self) -> None:
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None
