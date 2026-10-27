"""The session log copies stderr, our log lines and native code's, to a file, and still to the
terminal."""

import logging
import os
import re

import pytest

from ur5vla.sessionlog import session_log

STAMP = r'\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3}'


def test_our_lines_and_native_ones_are_saved_and_still_shown(tmp_path, capfd):
    # A handler on file descriptor 2, as basicConfig's is outside pytest (which swaps sys.stderr).
    handler = logging.StreamHandler(open(2, 'w', closefd=False, buffering=1))
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname).1s %(name)s: %(message)s'))
    logger = logging.getLogger('test.sessionlog')
    logger.addHandler(handler)
    logger.propagate = False
    path = tmp_path / 'run' / 'session.log'
    try:
        with session_log(path):
            logger.warning('teleop paused: Protective stop')
            os.write(2, b'RTDEControlInterface: Robot is disconnected, reconnecting...\n')
            os.write(2, b'Could not ')                  # C++ writes a line in pieces
            os.write(2, b'reconnect\n')
    finally:
        logger.removeHandler(handler)
        handler.stream.close()
    text = path.read_text()
    assert re.search(rf'^{STAMP} W test.sessionlog: teleop paused: Protective stop$', text, re.M)
    assert re.search(rf'^{STAMP} - stderr: RTDEControlInterface: Robot is disconnected', text, re.M)
    assert re.search(rf'^{STAMP} - stderr: Could not reconnect$', text, re.M)
    err = capfd.readouterr().err
    assert 'teleop paused' in err and 'Could not reconnect' in err
    os.write(2, b'after\n')
    assert 'after' not in path.read_text()              # stderr is back as it was


def test_an_error_leaving_the_session_is_saved(tmp_path):
    path = tmp_path / 'session.log'
    with pytest.raises(RuntimeError):
        with session_log(path):
            raise RuntimeError('ur_rtde: Failed to start control script')
    assert 'RuntimeError: ur_rtde: Failed to start control script' in path.read_text()
