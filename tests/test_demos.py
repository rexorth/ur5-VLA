"""Demo scenarios: parsing a demos file, and how `ur5vla demo` schedules scenarios and rounds."""

import argparse

import pytest

from ur5vla import cli
from ur5vla import config as config_mod
from ur5vla.demos import DemoError, Scenario, load_demos


def test_scenarios_parse_strings_and_mappings(tmp_path):
    f = tmp_path / 'd.yaml'
    f.write_text('rounds: 20\n'
                 'scenarios:\n'
                 '  - "put the cube in the box"\n'
                 '  - prompt: "take the cube out and put it on the table"\n'
                 '    name: reset\n'
                 '    start: pick_table\n')
    scenarios, rounds = load_demos(f)
    assert rounds == 20
    assert scenarios[0] == Scenario('put the cube in the box')
    assert scenarios[1] == Scenario('take the cube out and put it on the table', 'pick_table', 'reset')
    assert scenarios[0].label == 'put the cube in the box' and scenarios[1].label == 'reset'


def test_demos_file_errors(tmp_path):
    def load(text):
        p = tmp_path / 'x.yaml'
        p.write_text(text)
        return load_demos(p)

    with pytest.raises(DemoError, match='non-empty'):
        load('rounds: 2\n')
    with pytest.raises(DemoError, match='unknown top-level'):
        load('scenarios: ["a"]\nreps: 3\n')
    with pytest.raises(DemoError, match='unknown key'):
        load('scenarios:\n  - {prompt: a, stat: pick}\n')
    with pytest.raises(DemoError, match='non-empty prompt'):
        load('scenarios:\n  - {name: reset}\n')
    with pytest.raises(DemoError, match='positive integer'):
        load('rounds: 0\nscenarios: ["a"]\n')
    with pytest.raises(DemoError, match='not found'):
        load_demos(tmp_path / 'nope.yaml')


def _cfg():
    return config_mod.parse({'robot': {'start_poses': {'pick_table': [0, -90, 90, -90, -90, 0]}}})


def _args(**kw):
    base = dict(demos=None, scenario=None, prompt=None, start=None, rounds=None, episodes=1)
    return argparse.Namespace(**{**base, **kw})


def test_schedule_from_scenarios():
    scenarios, rounds, scheduled = cli._demo_schedule(_args(scenario=['task A', 'reset B'], rounds=5), _cfg())
    assert [s.prompt for s in scenarios] == ['task A', 'reset B'] and rounds == 5 and scheduled


def test_schedule_from_file_uses_its_rounds(tmp_path):
    f = tmp_path / 'd.yaml'
    f.write_text('rounds: 3\nscenarios:\n  - "x"\n  - {prompt: "y", start: pick_table, name: reset}\n')
    scenarios, rounds, scheduled = cli._demo_schedule(_args(demos=str(f)), _cfg())
    assert rounds == 3 and scheduled and scenarios[1].start == 'pick_table'
    # --rounds overrides the file
    _, rounds2, _ = cli._demo_schedule(_args(demos=str(f), rounds=7), _cfg())
    assert rounds2 == 7


def test_schedule_legacy_single_prompt_uses_episodes():
    scenarios, rounds, scheduled = cli._demo_schedule(_args(prompt='solo', episodes=4), _cfg())
    assert [s.prompt for s in scenarios] == ['solo'] and rounds == 4 and not scheduled


def test_schedule_rejects_an_unknown_start_pose(tmp_path):
    f = tmp_path / 'bad.yaml'
    f.write_text('scenarios:\n  - {prompt: z, start: nope}\n')
    with pytest.raises(SystemExit, match="start pose 'nope'"):
        cli._demo_schedule(_args(demos=str(f)), _cfg())


def test_schedule_needs_a_prompt():
    with pytest.raises(SystemExit, match='a prompt is required'):
        cli._demo_schedule(_args(), _cfg())
