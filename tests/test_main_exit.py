"""main() must force-exit on every outcome, with a non-zero code on failure."""

import pytest

import zotero_arxiv_daily.main as main_module


@pytest.fixture()
def wired(monkeypatch, config):
    codes = []
    monkeypatch.setattr(main_module, "load_config", lambda config_dir: config)
    monkeypatch.setattr(main_module, "configure_logging", lambda debug: None)
    monkeypatch.setattr("zotero_arxiv_daily.main.os._exit", codes.append)
    return codes


def test_failure_exits_nonzero_and_prints_traceback(monkeypatch, wired, capsys):
    def boom(config):
        raise RuntimeError("boom")

    monkeypatch.setattr(main_module, "run", boom)
    main_module.main()
    assert wired == [1]
    assert "boom" in capsys.readouterr().err


def test_success_exits_zero(monkeypatch, wired):
    monkeypatch.setattr(main_module, "run", lambda config: None)
    main_module.main()
    assert wired == [0]
