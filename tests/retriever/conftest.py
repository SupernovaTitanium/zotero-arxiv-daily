"""Retriever-specific fixtures."""

import pytest


@pytest.fixture()
def mock_biorxiv_api(monkeypatch):
    """Patch requests.get to return the canned bioRxiv API response."""
    import requests
    from types import SimpleNamespace

    from tests.canned_responses import SAMPLE_BIORXIV_API_RESPONSE

    original_get = requests.get

    def _patched(url, **kwargs):
        if "api.biorxiv.org" in url:
            resp = SimpleNamespace()
            resp.status_code = 200
            resp.json = lambda: SAMPLE_BIORXIV_API_RESPONSE
            resp.raise_for_status = lambda: None
            return resp
        return original_get(url, **kwargs)

    monkeypatch.setattr(requests, "get", _patched)
