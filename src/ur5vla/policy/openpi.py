"""openpi policy server (`scripts/serve_policy.py`) over its msgpack websocket, via openpi-client.

The request dict must use the keys the server's training config repacks from: for pi05_droid
that is observation/exterior_image_1_left, observation/wrist_image_left,
observation/joint_position, observation/gripper_position and prompt. Each logical camera is sent
under image_key.format(name=camera_map[camera]).
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


class OpenPIBackend(PolicyBackend):
    def __init__(self, **params):
        self.cfg = from_dict(OpenPIConfig, params, 'policy (openpi)')
        c = self.cfg
        self.mapping = ActionMapping(c.action_mode, c.gripper_open, c.gripper_closed, c.gripper_index)
        self._client = None
        self._metadata: dict = {}

    def connect(self) -> PolicyInfo:
        self._connect()
        return PolicyInfo('openpi', self.cfg.chunk_len, 1.0 / self.cfg.action_hz,
                          list(self.cfg.camera_map), {'server': self._metadata})

    def _connect(self) -> None:
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
        log.info('openpi: connecting to %s:%s (blocks until the server is up)', self.cfg.host, self.cfg.port)
        self._client = WebsocketClientPolicy(self.cfg.host, self.cfg.port, self.cfg.api_key)
        self._metadata = self._client.get_server_metadata()

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
        if self._client is None:
            self._connect()
        req = self.build_request(obs)
        t0 = time.monotonic()
        try:
            out = self._client.infer(req)
        except Exception:
            self.close()            # reconnect on the next call
            raise
        info = {'rtt_s': time.monotonic() - t0}
        for key in ('server_timing', 'policy_timing'):
            if key in out:
                info[key] = out[key]
        actions = np.asarray(out['actions'])
        if self.cfg.chunk_len is not None:
            actions = actions[: self.cfg.chunk_len]
        return self.mapping.to_chunk(actions, obs, 1.0 / self.cfg.action_hz, info)

    def reset(self) -> None:
        if self._client is not None:
            self._client.reset()

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client._ws.close()    # openpi-client exposes no close()
            except Exception:
                pass
            self._client = None
