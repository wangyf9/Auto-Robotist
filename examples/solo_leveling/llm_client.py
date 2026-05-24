import os
import json
from openai import OpenAI

from solo_leveling.config import LLM_MODEL, LLM_TEMPERATURE

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        from solo_leveling.config import OPENAI_API_KEY as CONFIG_KEY
        api_key = os.environ.get("OPENAI_API_KEY") or CONFIG_KEY
        if not api_key:
            raise EnvironmentError(
                "No API key found. Set OPENAI_API_KEY env var or fill in OPENAI_API_KEY in solo_leveling/config.py"
            )
        _client = OpenAI(api_key=api_key)
    return _client


def call_llm(messages: list[dict], model: str = None) -> dict:
    """Call LLM with JSON mode. Returns parsed dict.

    Args:
        messages: Chat messages.
        model: Override model (e.g. LLM_MODEL_STRONG for consolidation).
               Defaults to LLM_MODEL from config.
    """
    client = _get_client()
    effective_model = model or LLM_MODEL

    kwargs = dict(
        model=effective_model,
        messages=messages,
        response_format={"type": "json_object"},
    )
    if _supports_custom_temperature(effective_model):
        kwargs["temperature"] = LLM_TEMPERATURE

    response = client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content
    return json.loads(content)


def _supports_custom_temperature(model: str) -> bool:
    """Some reasoning models only accept the provider default temperature."""
    return not model.startswith("gpt-5")
