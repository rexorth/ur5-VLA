"""LeaderSimView: it renders the leader's pose on a UR5 in MuJoCo as frames for a second --view
window. Needs mujoco, the ur5-workspace scene and a display for the offscreen renderer."""

import os
import time
from pathlib import Path

import numpy as np
import pytest

def test_fold_into_range_keeps_the_physical_angle_within_limits():
    from ur5vla.leadersim import _fold_into_range, TURN
    deg = np.deg2rad
    lo = deg([-360, -360, -180, -360, -360, -360])
    hi = deg([360, 360, 180, 360, 360, 360])
    ref = deg([0, -90, 90, -90, -90, 0])
    # The elbow leader reading 259 deg is out of [-180, 180]; it must fold to -101 (same angle),
    # not clip to 180. Every result stays within range and physically equals the input (mod 360).
    q = deg([-355, -79, 259, -95, -251, 14])
    out = _fold_into_range(q, ref, lo, hi)
    assert np.all(out >= lo - 1e-9) and np.all(out <= hi + 1e-9)
    assert np.allclose((np.rad2deg(out) - np.rad2deg(q)) % 360, 0, atol=1e-6)
    assert np.isclose(np.rad2deg(out)[2], -101, atol=1e-6)      # elbow folded, not clipped to 180


pytest.importorskip('mujoco')

SCENE = Path(__file__).resolve().parents[2] / 'ur5-workspace' / 'scene_c5.xml'
HOME = [0.0, -90.0, 90.0, -90.0, -90.0, 0.0]


def _view(get_q):
    from ur5vla.config import BaseFrame
    from ur5vla.leadersim import LeaderSimView
    return LeaderSimView(SCENE, BaseFrame(), HOME, get_q, width=160, height=120, fps=30)


@pytest.mark.skipif(not SCENE.exists(), reason='ur5-workspace scene not present')
@pytest.mark.skipif(not os.environ.get('DISPLAY'), reason='no display for the offscreen renderer')
def test_renders_frames_and_follows_the_leader_joints():
    fed = {'q': None}
    view = _view(lambda: fed['q'])
    try:
        view.start()
        deadline = time.monotonic() + 10
        while view.latest() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        frame = view.latest()
        assert frame is not None
        assert frame.rgb.shape == (120, 160, 3) and frame.rgb.dtype == np.uint8
        seq = frame.seq

        fed['q'] = np.deg2rad([30.0, -60.0, 60.0, -90.0, -90.0, 0.0])
        time.sleep(0.5)
        expected = np.clip(fed['q'], view._arm_lo, view._arm_hi)
        assert np.allclose(view.data.ctrl[view._arm_act], expected)
        assert view.latest().seq > seq          # it kept rendering new frames
    finally:
        view.stop()


@pytest.mark.skipif(not SCENE.exists(), reason='ur5-workspace scene not present')
@pytest.mark.skipif(not os.environ.get('DISPLAY'), reason='no display for the offscreen renderer')
def test_a_get_q_that_raises_does_not_kill_the_render_thread():
    def boom():
        raise RuntimeError('leader read failed')
    view = _view(boom)
    try:
        view.start()
        deadline = time.monotonic() + 10
        while view.latest() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        first = view.latest()
        assert first is not None                # a frame despite get_q raising every tick
        time.sleep(0.3)
        assert view.latest().seq > first.seq     # and it keeps going
    finally:
        view.stop()
