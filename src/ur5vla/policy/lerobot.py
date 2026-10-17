"""LeRobot async-inference policy server (`python -m lerobot.async_inference.policy_server`)
over gRPC.

Only LeRobot's stubs and message types are used; chunk timing and blending stay in our loop, so
execution is identical to the openpi arm. One request per chunk: send the observation, then
block on GetActions for the chunk it produced.

Protocol notes (lerobot 0.6):
  * Messages are pickled `TimedObservation` / `list[TimedAction]` objects, so this side needs
    lerobot (and torch) installed, and the server must be trusted.
  * The server skips observations whose state is within 1.0 (L2) of the last one it ran unless
    `must_go` is set. In radians that is almost every UR5 observation, so every observation
    here is sent with must_go=True.
  * The policy is loaded on SendPolicyInstructions, whose `lerobot_features` must describe the
    images actually sent. Their shapes are known only once frames exist, so the instructions go
    out with the first infer() call (the loop's warm-up inference).
  * The server resizes each image to the checkpoint's shape by plain bilinear interpolation (no
    padding). Leave image_size unset to let it, or set it to that shape to save bandwidth.
"""

from __future__ import annotations

import logging
import pickle  # nosec: LeRobot's wire format
import time
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from ..config import from_dict
from ..types import ActionChunk, Observation, PolicyInfo
from .base import ActionMapping, PolicyBackend

log = logging.getLogger(__name__)

JOINT_NAMES = ('shoulder_pan', 'shoulder_lift', 'elbow', 'wrist_1', 'wrist_2', 'wrist_3')


@dataclass
class LeRobotConfig:
    action_hz: float                # rate of the training data's actions (the dataset fps)
    camera_map: dict[str, str]      # logical camera -> policy camera (observation.images.<name>)
    policy_type: str                # act | smolvla | diffusion | tdmpc | vqbet | pi0 | pi05 | groot
    pretrained_name_or_path: str
    host: str = 'localhost'
    port: int = 8080
    policy_device: str = 'cuda'     # where the SERVER runs the policy
    actions_per_chunk: int = 50
    image_size: list[int] | None = None   # H, W; None = native, the server resizes
    include_gripper_state: bool = True    # observation.state = 6 joints (+ gripper)
    rename_map: dict[str, str] = field(default_factory=dict)
    timeout_s: float = 10.0         # per observation / action call
    load_timeout_s: float = 600.0   # SendPolicyInstructions: the server loads the checkpoint
    action_mode: str = 'absolute'
    gripper_open: float = 0.0
    gripper_closed: float = 1.0
    gripper_index: int | None = 6


class LeRobotBackend(PolicyBackend):
    def __init__(self, **params):
        self.cfg = from_dict(LeRobotConfig, params, 'policy (lerobot)')
        c = self.cfg
        self.mapping = ActionMapping(c.action_mode, c.gripper_open, c.gripper_closed, c.gripper_index)
        self.state_names = [f'{j}.pos' for j in JOINT_NAMES] + (['gripper.pos'] if c.include_gripper_state else [])
        self._channel = None
        self._stub = None
        self._instructed = False
        self._step = 0
        self._warned_dt = False

    def connect(self) -> PolicyInfo:
        import grpc
        from lerobot.transport import services_pb2, services_pb2_grpc
        from lerobot.transport.utils import grpc_channel_options
        self._pb = services_pb2
        address = f'{self.cfg.host}:{self.cfg.port}'
        log.info('lerobot: connecting to %s', address)
        self._channel = grpc.insecure_channel(address, grpc_channel_options())
        self._stub = services_pb2_grpc.AsyncInferenceStub(self._channel)
        self._stub.Ready(services_pb2.Empty(), timeout=self.cfg.timeout_s, wait_for_ready=True)
        self._instructed = False
        return PolicyInfo('lerobot', self.cfg.actions_per_chunk, 1.0 / self.cfg.action_hz,
                          list(self.cfg.camera_map),
                          {'policy_type': self.cfg.policy_type, 'checkpoint': self.cfg.pretrained_name_or_path})

    def raw_observation(self, obs: Observation) -> dict:
        state = list(obs.q) + ([self.mapping.gripper_to_policy(obs.gripper)] if self.cfg.include_gripper_state else [])
        raw: dict = {name: float(v) for name, v in zip(self.state_names, state, strict=True)}
        for cam, policy_name in self.cfg.camera_map.items():
            img = obs.images[cam].rgb
            if self.cfg.image_size is not None:
                h, w = self.cfg.image_size
                img = np.asarray(Image.fromarray(img).resize((w, h), Image.BILINEAR))
            raw[policy_name] = img
        raw['task'] = obs.prompt
        return raw

    def _send_instructions(self, raw: dict) -> None:
        from lerobot.async_inference.helpers import RemotePolicyConfig
        from lerobot.utils.feature_utils import hw_to_dataset_features
        hw = {name: float for name in self.state_names}
        hw.update({p: tuple(raw[p].shape) for p in self.cfg.camera_map.values()})
        features = hw_to_dataset_features(hw, 'observation', use_video=False)
        c = self.cfg
        spec = RemotePolicyConfig(c.policy_type, c.pretrained_name_or_path, features, c.actions_per_chunk,
                                  c.policy_device, dict(c.rename_map))
        log.info('lerobot: loading %s (%s) on the server ...', c.pretrained_name_or_path, c.policy_type)
        t0 = time.monotonic()
        self._stub.SendPolicyInstructions(self._pb.PolicySetup(data=pickle.dumps(spec)),
                                          timeout=c.load_timeout_s)
        log.info('lerobot: policy loaded in %.1f s', time.monotonic() - t0)
        self._instructed = True

    def infer(self, obs: Observation) -> ActionChunk:
        from lerobot.async_inference.helpers import TimedObservation
        from lerobot.transport.utils import send_bytes_in_chunks
        raw = self.raw_observation(obs)
        if not self._instructed:
            self._send_instructions(raw)
        self._step += 1
        timed = TimedObservation(timestamp=time.time(), timestep=self._step, observation=raw, must_go=True)
        t0 = time.monotonic()
        self._stub.SendObservations(send_bytes_in_chunks(pickle.dumps(timed), self._pb.Observation),
                                    timeout=self.cfg.timeout_s)
        resp = self._stub.GetActions(self._pb.Empty(), timeout=self.cfg.timeout_s)
        rtt = time.monotonic() - t0
        if not resp.data:
            raise RuntimeError(f'lerobot: no actions for observation #{self._step} (see the server log)')
        actions = pickle.loads(resp.data)  # nosec: list[TimedAction] from our own server
        if actions[0].get_timestep() != self._step:
            raise RuntimeError(f'lerobot: chunk is for observation #{actions[0].get_timestep()}, '
                               f'expected #{self._step}')
        self._check_dt(actions)
        a = np.stack([np.asarray(ta.get_action(), dtype=np.float64) for ta in actions])
        return self.mapping.to_chunk(a, obs, 1.0 / self.cfg.action_hz, {'rtt_s': rtt, 'timestep': self._step})

    def _check_dt(self, actions) -> None:
        """The server stamps actions with its --fps, which is a server flag, not a checkpoint
        property. Ours comes from config; say so once if they disagree."""
        if len(actions) < 2 or self._warned_dt:
            return
        server_dt = actions[1].get_timestamp() - actions[0].get_timestamp()
        if abs(server_dt * self.cfg.action_hz - 1.0) > 0.01:
            log.warning('lerobot: server action spacing %.4f s != policy.action_hz %.1f; using '
                        'action_hz. Start the server with --fps=%g to silence this.',
                        server_dt, self.cfg.action_hz, self.cfg.action_hz)
            self._warned_dt = True

    def reset(self) -> None:
        if self._stub is not None:
            # Ready() flushes the server's observation queue; the loaded policy stays.
            self._stub.Ready(self._pb.Empty(), timeout=self.cfg.timeout_s)

    def close(self) -> None:
        if self._channel is not None:
            self._channel.close()
            self._channel = self._stub = None
