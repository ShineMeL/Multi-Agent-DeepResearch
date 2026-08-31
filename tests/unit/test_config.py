import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from deepresearch.config import Settings


def _serialized_validation_error(error: ValidationError) -> str:
    return "\n".join(
        (
            str(error),
            repr(error),
            repr(error.errors()),
            error.json(),
            json.dumps(error.errors(), default=str, sort_keys=True),
            repr(error.__cause__),
            repr(error.__context__),
        )
    )


def test_settings_load_typed_environment_without_exposing_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEEPRESEARCH_MODEL_BASE_URL", "https://model.test/v1")
    monkeypatch.setenv("DEEPRESEARCH_MODEL_ID", "fixture-model")
    monkeypatch.setenv("DEEPRESEARCH_MODEL_API_KEY", "TOP-SECRET-MODEL")
    monkeypatch.setenv("DEEPRESEARCH_TAVILY_API_KEY", "TOP-SECRET-SEARCH")
    monkeypatch.setenv("DEEPRESEARCH_ARTIFACT_ROOT", "custom-artifacts")

    settings = Settings(_env_file=None)

    assert str(settings.model_base_url) == "https://model.test/v1"
    assert settings.model_id == "fixture-model"
    assert settings.model_api_key.get_secret_value() == "TOP-SECRET-MODEL"
    assert settings.tavily_api_key is not None
    assert settings.tavily_api_key.get_secret_value() == "TOP-SECRET-SEARCH"
    assert settings.artifact_root == Path("custom-artifacts")
    assert "TOP-SECRET" not in repr(settings)
    assert "TOP-SECRET" not in settings.model_dump_json()


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("DEEPRESEARCH_MODEL_BASE_URL", "https://user:password@model.test/v1"),
        ("DEEPRESEARCH_CONNECT_TIMEOUT_SECONDS", "0"),
        ("DEEPRESEARCH_READ_TIMEOUT_SECONDS", "nan"),
    ),
)
def test_settings_reject_credentialed_urls_and_invalid_timeouts(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    monkeypatch.setenv("DEEPRESEARCH_MODEL_BASE_URL", "https://model.test/v1")
    monkeypatch.setenv("DEEPRESEARCH_MODEL_ID", "fixture-model")
    monkeypatch.setenv("DEEPRESEARCH_MODEL_API_KEY", "TOP-SECRET-MODEL")
    monkeypatch.setenv(name, value)

    with pytest.raises(ValidationError) as error:
        Settings(_env_file=None)

    assert error.value.errors()[0]["loc"] == (
        name.removeprefix("DEEPRESEARCH_").casefold(),
    )


@pytest.mark.parametrize(
    ("overrides", "secrets"),
    (
        (
            {"model_id": ""},
            ("TOP-SECRET-MODEL", "TOP-SECRET-SEARCH"),
        ),
        (
            {"model_base_url": "https://user:URL-SECRET@model.test/v1"},
            ("URL-SECRET", "TOP-SECRET-MODEL", "TOP-SECRET-SEARCH"),
        ),
        (
            {"model_api_key": {"raw": "FIELD-SECRET"}},
            ("FIELD-SECRET", "TOP-SECRET-SEARCH"),
        ),
    ),
)
def test_settings_validation_errors_never_serialize_raw_credentials(
    overrides: dict[str, object], secrets: tuple[str, ...]
) -> None:
    values: dict[str, object] = {
        "model_base_url": "https://model.test/v1",
        "model_id": "fixture-model",
        "model_api_key": "TOP-SECRET-MODEL",
        "tavily_api_key": "TOP-SECRET-SEARCH",
        **overrides,
    }

    with pytest.raises(ValidationError) as caught:
        Settings(_env_file=None, **values)

    serialized = _serialized_validation_error(caught.value)
    assert all(secret not in serialized for secret in secrets)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
