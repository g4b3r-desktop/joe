import pytest

from joe.services.colab_api import ColabAPIClient


def test_colab_client_requires_https_url():
    with pytest.raises(ValueError, match="HTTPS"):
        ColabAPIClient("http://example.com", "x" * 40)


def test_colab_client_requires_token():
    with pytest.raises(ValueError, match="token"):
        ColabAPIClient("https://example.trycloudflare.com", "")


def test_colab_client_normalizes_url():
    client = ColabAPIClient("https://example.trycloudflare.com/", "x" * 40)
    assert client.base_url == "https://example.trycloudflare.com"
