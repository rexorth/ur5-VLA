"""Policy adapters against in-process fake servers that speak the real wire formats.

Each test runs only in the venv that has its client library (.venv-openpi / .venv-lerobot).
"""

import pickle
import threading
from concurrent import futures

import numpy as np
import pytest

from ur5vla.types import Frame, Observation

Q = np.deg2rad([0.0, -90.0, 90.0, -90.0, -90.0, 0.0])


def _obs(prompt='pick up the cube'):
    rgb = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    frames = {'base': Frame(rgb, None, 10.0, 1), 'wrist': Frame(rgb, None, 10.0, 1)}
    return Observation(frames, Q.copy(), 0.25, None, prompt, 12.5)


def test_openpi_backend_against_fake_server():
    pytest.importorskip('openpi_client')
    from openpi_client import msgpack_numpy
    from websockets.sync.server import serve

    from ur5vla.policy.openpi import OpenPIBackend

    received = []

    def handler(ws):
        ws.send(msgpack_numpy.packb({'policy': 'fake'}))
        for msg in ws:
            received.append(msgpack_numpy.unpackb(msg))
            actions = np.zeros((10, 7))
            actions[:, :6] = 0.01 * np.arange(10)[:, None]
            actions[:, 6] = 0.5
            ws.send(msgpack_numpy.packb({'actions': actions, 'server_timing': {'infer_ms': 3.0}}))

    server = serve(handler, 'localhost', 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        backend = OpenPIBackend(action_hz=10, host='localhost', port=server.socket.getsockname()[1],
                                camera_map={'base': 'exterior_image_1_left', 'wrist': 'wrist_image_left'},
                                action_mode='delta')
        info = backend.connect()
        assert info.metadata['server'] == {'policy': 'fake'} and info.cameras == ['base', 'wrist']
        obs = _obs()
        chunk = backend.infer(obs)
        backend.close()
    finally:
        server.shutdown()

    req = received[0]
    assert set(req) == {'observation/exterior_image_1_left', 'observation/wrist_image_left',
                        'observation/joint_position', 'observation/gripper_position', 'prompt'}
    img = req['observation/exterior_image_1_left']
    assert img.shape == (224, 224, 3) and img.dtype == np.uint8
    assert (img[:28] == 0).all()                    # 640x480 letterboxed: padded top and bottom
    np.testing.assert_allclose(req['observation/joint_position'], Q, rtol=1e-6)
    np.testing.assert_allclose(req['observation/gripper_position'], [0.25])
    assert req['prompt'] == 'pick up the cube'
    np.testing.assert_allclose(chunk.q_targets, Q + 0.01 * np.arange(10)[:, None])   # delta -> absolute
    np.testing.assert_allclose(chunk.gripper, 0.5)
    assert chunk.dt == pytest.approx(0.1) and chunk.t_obs == obs.t
    assert chunk.info['server_timing'] == {'infer_ms': 3.0}


def test_lerobot_backend_against_fake_server():
    grpc = pytest.importorskip('grpc')
    pytest.importorskip('lerobot')
    import torch
    from lerobot.async_inference.helpers import TimedAction, raw_observation_to_observation
    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.transport import services_pb2, services_pb2_grpc
    from lerobot.transport.utils import receive_bytes_in_chunks

    from ur5vla.policy.lerobot import LeRobotBackend

    class FakeServer(services_pb2_grpc.AsyncInferenceServicer):
        spec = None
        observations = []

        def Ready(self, request, context):  # noqa: N802
            return services_pb2.Empty()

        def SendPolicyInstructions(self, request, context):  # noqa: N802
            FakeServer.spec = pickle.loads(request.data)
            return services_pb2.Empty()

        def SendObservations(self, request_iterator, context):  # noqa: N802
            data = receive_bytes_in_chunks(request_iterator, None, threading.Event())
            FakeServer.observations.append(pickle.loads(data))
            return services_pb2.Empty()

        def GetActions(self, request, context):  # noqa: N802
            o = FakeServer.observations[-1]
            chunk = [TimedAction(timestamp=o.get_timestamp() + i / 30, timestep=o.get_timestep() + i,
                                 action=torch.full((7,), 0.1 * i)) for i in range(5)]
            return services_pb2.Actions(data=pickle.dumps(chunk))

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(FakeServer(), server)
    port = server.add_insecure_port('localhost:0')
    server.start()
    try:
        backend = LeRobotBackend(action_hz=30, port=port, pretrained_name_or_path='user/ckpt',
                                 camera_map={'base': 'front', 'wrist': 'wrist'}, actions_per_chunk=5)
        info = backend.connect()
        assert info.chunk_len == 5
        obs = _obs()
        chunk = backend.infer(obs)
        backend.close()
    finally:
        server.stop(0)

    spec, sent = FakeServer.spec, FakeServer.observations[0]
    assert spec.policy_type == 'pi05' and spec.pretrained_name_or_path == 'user/ckpt'
    assert spec.lerobot_features['observation.state']['shape'] == (7,)
    assert spec.lerobot_features['observation.images.front']['shape'] == (480, 640, 3)
    assert sent.must_go and sent.get_timestep() == 1

    # What the real server does with it: state vector in our joint order, images resized.
    policy_images = {k: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224))
                     for k in ('observation.images.front', 'observation.images.wrist')}
    converted = raw_observation_to_observation(sent.get_observation(), spec.lerobot_features, policy_images)
    np.testing.assert_allclose(converted['observation.state'][0, :6].numpy(), Q, rtol=1e-6)
    assert converted['observation.state'][0, 6].item() == pytest.approx(0.25)
    assert tuple(converted['observation.images.front'].shape) == (1, 3, 224, 224)
    assert converted['task'] == 'pick up the cube'

    np.testing.assert_allclose(chunk.q_targets[:, 0], 0.1 * np.arange(5), atol=1e-6)
    np.testing.assert_allclose(chunk.gripper, np.clip(0.1 * np.arange(5), 0, 1), atol=1e-6)
    assert chunk.dt == pytest.approx(1 / 30) and chunk.t_obs == obs.t
