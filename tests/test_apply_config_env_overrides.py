from omegaconf import OmegaConf

from scripts.apply_config_env_overrides import apply_overrides


def test_apply_overrides_uses_smtp_env_vars(tmp_path, monkeypatch):
    config_path = tmp_path / "custom.yaml"
    config_path.write_text(
        "email:\n"
        "  sender: ${oc.env:SENDER}\n"
        "  receiver: ${oc.env:RECEIVER}\n"
        "  smtp_server: smtp.qq.com\n"
        "  smtp_port: 465\n"
        "  sender_password: ${oc.env:SENDER_PASSWORD}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("SMTP_SERVER", "smtp.example.com")
    monkeypatch.setenv("SMTP_PORT", "587")

    assert apply_overrides(config_path) is True

    config = OmegaConf.load(config_path)
    assert config.email.smtp_server == "smtp.example.com"
    assert config.email.smtp_port == 587


def test_apply_overrides_uses_email_smtp_fallback_names(tmp_path, monkeypatch):
    config_path = tmp_path / "custom.yaml"
    config_path.write_text("email:\n  smtp_server: old.example.com\n  smtp_port: 25\n", encoding="utf-8")
    monkeypatch.delenv("SMTP_SERVER", raising=False)
    monkeypatch.delenv("SMTP_PORT", raising=False)
    monkeypatch.setenv("EMAIL_SMTP_SERVER", "smtp.fallback.example.com")
    monkeypatch.setenv("EMAIL_SMTP_PORT", "2525")

    assert apply_overrides(config_path) is True

    config = OmegaConf.load(config_path)
    assert config.email.smtp_server == "smtp.fallback.example.com"
    assert config.email.smtp_port == 2525


def test_apply_overrides_leaves_config_when_env_absent(tmp_path, monkeypatch):
    config_path = tmp_path / "custom.yaml"
    original = "email:\n  smtp_server: smtp.qq.com\n  smtp_port: 465\n"
    config_path.write_text(original, encoding="utf-8")
    for name in ("SMTP_SERVER", "SMTP_PORT", "EMAIL_SMTP_SERVER", "EMAIL_SMTP_PORT"):
        monkeypatch.delenv(name, raising=False)

    assert apply_overrides(config_path) is False
    assert config_path.read_text(encoding="utf-8") == original
