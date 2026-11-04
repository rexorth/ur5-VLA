"""OakCamera teardown and connect, without the depthai library or a device (both are only touched
through the fakes here). These guard the shutdown path that was leaving the OAK's watchdog to reset
it on quit, and the start-of-session reconnect to a device still re-enumerating."""

import pytest

from ur5vla.cameras.oak import OakCamera


def test_close_always_closes_the_device_even_if_pipeline_stop_fails():
    cam = OakCamera('base')
    closed = []

    class Pipe:
        def stop(self):
            raise RuntimeError('pipeline stop boom')

    class Dev:
        def close(self):
            closed.append(True)

    cam._pipeline, cam._device = Pipe(), Dev()
    cam._close()
    assert closed == [True]                                      # device closed despite the error
    assert cam._pipeline is None and cam._device is None and cam._queue is None
    cam._close()                                                 # a second close is a no-op
    assert closed == [True]


def test_connect_retries_until_the_device_is_ready(monkeypatch):
    import ur5vla.cameras.oak as oak
    monkeypatch.setattr(oak.time, 'sleep', lambda s: None)       # don't wait out the backoff
    cam = OakCamera('base')

    class FakeDai:
        calls = 0

        class DeviceInfo:
            def __init__(self, _):
                pass

        @staticmethod
        def Device(info=None):
            FakeDai.calls += 1
            if FakeDai.calls < 3:                                # fails twice (still re-enumerating)
                raise RuntimeError('X_LINK_DEVICE_NOT_FOUND')
            return 'device'

    assert cam._connect(FakeDai, tries=5, delay_s=0.0) == 'device'
    assert FakeDai.calls == 3


def test_connect_gives_up_and_raises_the_last_error(monkeypatch):
    import ur5vla.cameras.oak as oak
    monkeypatch.setattr(oak.time, 'sleep', lambda s: None)
    cam = OakCamera('base')

    class AlwaysFail:
        class DeviceInfo:
            def __init__(self, _):
                pass

        @staticmethod
        def Device(info=None):
            raise RuntimeError('still not there')

    with pytest.raises(RuntimeError, match='still not there'):
        cam._connect(AlwaysFail, tries=2, delay_s=0.0)
