"""Apply selected environment variable overrides to a Hydra custom config."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

from omegaconf import OmegaConf, open_dict


def _first_nonempty(names: Iterable[str]) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def apply_overrides(config_path: str | Path) -> bool:
    path = Path(config_path)
    config = OmegaConf.load(path)
    changed = False

    smtp_server = _first_nonempty(("SMTP_SERVER", "EMAIL_SMTP_SERVER"))
    smtp_port = _first_nonempty(("SMTP_PORT", "EMAIL_SMTP_PORT"))

    if smtp_server or smtp_port:
        with open_dict(config):
            if "email" not in config or config.email is None:
                config.email = {}
            if smtp_server:
                config.email.smtp_server = smtp_server
                changed = True
            if smtp_port:
                config.email.smtp_port = int(smtp_port)
                changed = True

    if changed:
        OmegaConf.save(config, path)

    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    args = parser.parse_args()
    changed = apply_overrides(args.config_path)
    print("Applied email config env overrides" if changed else "No email config env overrides found")


if __name__ == "__main__":
    main()
