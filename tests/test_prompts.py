"""The y/n and outcome prompts: Enter is affirmative, --yes skips them, and an unrecognized key
re-asks instead of defaulting."""

import builtins

from ur5vla import cli


def _answers(monkeypatch, *replies):
    it = iter(replies)
    monkeypatch.setattr(builtins, 'input', lambda *a, **k: next(it))


def _no_prompt(monkeypatch):
    def boom(*a, **k):
        raise AssertionError('should not prompt with assume_yes')
    monkeypatch.setattr(builtins, 'input', boom)


def test_confirm_enter_is_yes(monkeypatch):
    _answers(monkeypatch, '')
    assert cli._confirm('go?', assume_yes=False) is True


def test_confirm_y_and_n(monkeypatch):
    _answers(monkeypatch, 'y')
    assert cli._confirm('go?', False) is True
    _answers(monkeypatch, 'n')
    assert cli._confirm('go?', False) is False


def test_confirm_reprompts_until_valid(monkeypatch, capsys):
    _answers(monkeypatch, 'maybe', 'x', 'n')
    assert cli._confirm('go?', False) is False
    assert capsys.readouterr().out.count('please answer') == 2


def test_confirm_assume_yes_skips_prompt(monkeypatch):
    _no_prompt(monkeypatch)
    assert cli._confirm('go?', assume_yes=True) is True


def test_outcome_enter_is_success(monkeypatch):
    _answers(monkeypatch, '')
    assert cli._ask_label(assume_yes=False) == 'success'


def test_outcome_letters(monkeypatch):
    for key, label in (('s', 'success'), ('f', 'fail'), ('d', 'discard')):
        _answers(monkeypatch, key)
        assert cli._ask_label(False) == label


def test_outcome_reprompts_until_valid(monkeypatch, capsys):
    _answers(monkeypatch, 'z', 'f')
    assert cli._ask_label(False) == 'fail'
    assert 'please answer s, f, or d' in capsys.readouterr().out


def test_outcome_assume_yes_is_unlabelled(monkeypatch):
    _no_prompt(monkeypatch)
    assert cli._ask_label(assume_yes=True) == 'unlabelled'
