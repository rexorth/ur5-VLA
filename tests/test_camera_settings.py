"""Fixed camera settings: config checks, what each driver sets (against stand-ins for depthai,
librealsense and V4L2), `cameras --measure`, and the settings saved with each episode."""

import json
import types
from pathlib import Path

import pytest

from ur5vla import config as config_mod
from ur5vla.cameras import v4l2
from ur5vla.cameras.oak import OakCamera
from ur5vla.cameras.realsense import RealSenseCamera
from ur5vla.cameras.synthetic import SyntheticCamera
from ur5vla.cameras.uvc import UVCCamera
from ur5vla.config import ConfigError
from ur5vla.recording import CAMERAS, EpisodeRecorder, read_index

DRY_RUN = Path(__file__).parents[1] / 'configs' / 'dry_run.yaml'     # synthetic base and wrist


@pytest.mark.parametrize('cls, params, words', [
    (OakCamera, {'exposure_us': 20000}, 'exposure_us and iso are set together'),
    (OakCamera, {'focus': 300}, 'focus 300 outside'),
    (OakCamera, {'exposure_us': 50000, 'iso': 200}, 'exposure_us 50000 outside'),   # > 1/30 s
    (OakCamera, {'exposure_us': 20000, 'iso': 200, 'max_exposure_us': 20000}, 'max_exposure_us limits'),
    (OakCamera, {'mains_hz': 55}, 'mains_hz 55: 50 or 60'),
    (RealSenseCamera, {'max_exposure_us': 20000}, 'unknown key'),                   # it can't cap
    (RealSenseCamera, {'gain': 64}, 'gain is fixed only with exposure_us'),
    (RealSenseCamera, {'focus': 100}, 'unknown key'),                               # fixed-focus lens
    (RealSenseCamera, {'white_balance_k': 9000}, 'white_balance_k 9000 outside'),
    (UVCCamera, {'gain': 10}, 'gain is fixed only with exposure_us'),
])
def test_settings_that_cant_work_are_refused_in_the_config(cls, params, words):
    with pytest.raises(ConfigError, match=words):
        cls('cam', **params)


def test_v4l2_structs_and_ioctl_numbers_match_the_kernel_headers():
    assert v4l2._QUERYCTRL.size == 68 and v4l2._CONTROL.size == 8
    assert (v4l2.VIDIOC_QUERYCTRL, v4l2.VIDIOC_G_CTRL, v4l2.VIDIOC_S_CTRL) == (0xc0445624, 0xc008561b, 0xc008561c)


class Recorder:
    """Records method calls, by name, on whatever attribute is asked for."""
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        return lambda *args: self.calls.append((name, *args))


AF = types.SimpleNamespace(AutoFocusMode=types.SimpleNamespace(OFF='OFF', CONTINUOUS_VIDEO='CONT'),
                           AntiBandingMode=types.SimpleNamespace(MAINS_50_HZ='50Hz', MAINS_60_HZ='60Hz'))


def test_oak_fixes_focus_exposure_and_white_balance_and_autofocus_stays_off():
    cam = OakCamera('base', focus=107, exposure_us=20000, iso=146, white_balance_k=3800)
    cam._autofocus = True
    ctrl = Recorder()
    cam._fix_settings(ctrl, AF)
    assert ctrl.calls == [('setAutoFocusMode', 'OFF'), ('setManualFocus', 107),
                          ('setManualExposure', 20000, 146), ('setManualWhiteBalance', 3800)]
    cam.automatic = True                                # --measure: everything automatic
    ctrl = Recorder()
    cam._fix_settings(ctrl, AF)
    assert ctrl.calls == [('setAutoFocusMode', 'CONT')]


def test_oak_on_automatic_exposure_is_capped_and_flicker_free_measuring_too():
    cam = OakCamera('base', focus=107, max_exposure_us=20000, mains_hz=50)
    cam._autofocus = True
    for automatic, focus_calls in ((False, [('setAutoFocusMode', 'OFF'), ('setManualFocus', 107)]),
                                   (True, [('setAutoFocusMode', 'CONT')])):
        cam.automatic = automatic
        ctrl = Recorder()
        cam._fix_settings(ctrl, AF)
        assert ctrl.calls == [('setAntiBandingMode', '50Hz'), ('setAutoExposureLimit', 20000), *focus_calls]


class FakeColourSensor:
    """librealsense's colour sensor: options with defaults, as another program left them."""
    def __init__(self, option):
        o = option
        self.defaults = {o.brightness: 0, o.saturation: 64, o.enable_auto_exposure: 1, o.exposure: 166,
                         o.gain: 64, o.enable_auto_white_balance: 1, o.white_balance: 4600,
                         o.global_time_enabled: 1, o.frames_queue_size: 16, o.power_line_frequency: 3}
        self.values = {**self.defaults, o.brightness: 30, o.saturation: 90, o.gain: 10}

    def get_supported_options(self):
        return list(self.defaults)

    def is_option_read_only(self, opt):
        return False

    def get_option_range(self, opt):
        return types.SimpleNamespace(default=self.defaults[opt])

    def get_option(self, opt):
        return self.values[opt]

    def set_option(self, opt, value):
        self.values[opt] = value


def _realsense(**params):
    cam = RealSenseCamera('wrist', **params)
    names = ('brightness', 'saturation', 'enable_auto_exposure', 'exposure', 'gain', 'enable_auto_white_balance',
             'white_balance', 'global_time_enabled', 'frames_queue_size', 'power_line_frequency')
    cam._rs = types.SimpleNamespace(option=types.SimpleNamespace(**{n: n for n in names}))
    return cam, FakeColourSensor(cam._rs.option)


def test_realsense_resets_what_others_left_then_fixes_exposure_gain_and_white_balance():
    cam, sensor = _realsense(exposure_us=20000, white_balance_k=5500, mains_hz=50)
    cam._fix_settings(sensor)
    v = sensor.values
    assert (v['brightness'], v['saturation']) == (0, 64)                    # back to the defaults
    assert v['power_line_frequency'] == 1                                   # 50 Hz
    assert (v['enable_auto_exposure'], v['exposure'], v['gain']) == (0, 200, 64)   # 100 us steps; default gain
    assert (v['enable_auto_white_balance'], v['white_balance']) == (0, 5500)
    assert cam.settings() == {'exposure_us': 20000, 'gain': 64, 'white_balance_k': 5500}


def test_realsense_on_automatic_reports_nothing_it_cant_read():
    cam, sensor = _realsense(exposure_us=20000)
    cam.automatic = True
    cam._fix_settings(sensor)
    assert sensor.values['enable_auto_exposure'] == 1 and sensor.values['enable_auto_white_balance'] == 1
    assert cam.settings() is None


def test_uvc_turns_autofocus_off_and_asks_for_a_focus(monkeypatch):
    values = {v4l2.FOCUS_AUTO: 1, v4l2.FOCUS_ABSOLUTE: 0}
    monkeypatch.setattr(v4l2, 'reset', lambda fd, have: [])
    monkeypatch.setattr(v4l2, 'set', lambda fd, cid, value: values.__setitem__(cid, value))
    monkeypatch.setattr(v4l2, 'get', lambda fd, cid: values[cid])
    lens = v4l2.Control(v4l2.FOCUS_ABSOLUTE, 'focus_absolute', 1, 0, 255, 0, 0)
    have = {v4l2.FOCUS_AUTO: v4l2.Control(v4l2.FOCUS_AUTO, 'focus_automatic_continuous', 2, 0, 1, 1, 0),
            v4l2.FOCUS_ABSOLUTE: lens}
    cam = UVCCamera('side')
    cam._fd, cam._controls = 99, have
    with pytest.raises(ConfigError, match=r'autofocus, which is kept off: set focus .*0\.\.255'):
        cam._fix_settings()
    assert values[v4l2.FOCUS_AUTO] == 0
    cam = UVCCamera('side', focus=40)
    cam._fd, cam._controls = 99, have
    cam._fix_settings()
    assert values[v4l2.FOCUS_AUTO] == 0 and values[v4l2.FOCUS_ABSOLUTE] == 40
    assert cam._read_settings() == {'focus': 40}


class ChoosingCamera(SyntheticCamera):
    """Reports settings only when left to choose them, like an OAK under --measure."""
    def settings(self):
        return {'focus': 103.6, 'exposure_us': 19989} if self.automatic else None


def test_measure_prints_what_each_camera_chose_as_config_keys(monkeypatch, capsys):
    from ur5vla import cli
    built = []

    def build(cfg):
        built.extend(cfg)
        return {name: (ChoosingCamera if name == 'base' else SyntheticCamera)(name, **{k: v for k, v in c.items()
                                                                                      if k != 'type'})
                for name, c in cfg.items()}
    monkeypatch.setattr(cli, 'build_cameras', build)
    assert cli.main(['cameras', '-c', str(DRY_RUN), '--measure', '--seconds', '1.2']) == 0
    out = capsys.readouterr().out
    assert '  focus 103.6  exposure_us 19989' in out                  # in the once-a-second report
    assert 'base: focus: 104, exposure_us: 19989' in out
    assert "wrist: it doesn't report what it chose" in out


class ReportingCamera(SyntheticCamera):
    """Its exposure follows the light, and each frame says what it was."""
    def _grab(self):
        frame = super()._grab()
        frame.settings = {'focus': 107, 'exposure_us': 10000 + frame.seq, 'iso': 146}
        return frame

    def settings(self):
        return (self.latest() or types.SimpleNamespace(settings=None)).settings


def test_each_episode_saves_its_cameras_settings_at_start_and_per_frame(tmp_path):
    import time
    cams = {'base': ReportingCamera('base', width=8, height=6, fps=30), 'wrist': SyntheticCamera('wrist', width=8, height=6)}
    for cam in cams.values():
        cam.start()
    try:
        time.sleep(0.2)
        rec = EpisodeRecorder(tmp_path / 'ep', cams, config_mod.LoggingConfig())
        rec.start({'kind': 'demo'})
        time.sleep(0.4)
        rec.finish({})
    finally:
        for cam in cams.values():
            cam.stop()
    meta = json.loads((tmp_path / 'ep' / 'meta.json').read_text())
    assert meta['camera_settings']['base']['iso'] == 146 and meta['camera_settings']['wrist'] is None
    rows = read_index(tmp_path / 'ep' / CAMERAS / 'base')
    assert len(rows) >= 5 and all(int(r['exposure_us']) == 10000 + r['seq'] and r['focus'] == '107' for r in rows)
    assert all(r['exposure_us'] == '' for r in read_index(tmp_path / 'ep' / CAMERAS / 'wrist'))
