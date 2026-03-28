"""
Unified LLM client with automatic fallback.

Fallback chain: Z AI → OpenAI → Gemini
Falls back automatically on 429 (rate limit), 503 (unavailable), or connection errors.
"""

import os
import httpx
from typing import Optional

from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
load_dotenv(os.path.join(os.path.dirname(__file__), '..', 'sim', '.env.local'))
load_dotenv()

# Errors that trigger fallback
_FALLBACK_CODES = {429, 500, 503}


# ──────────────────────────────────────────
# Gemini client
# ──────────────────────────────────────────

_gemini_client = None


def _get_gemini_client() -> Optional[genai.Client]:
    global _gemini_client
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return None
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


async def _call_gemini(
    model: str,
    contents,
    system_instruction: str = "",
    temperature: float = 0.2,
    response_json: bool = False,
) -> str:
    """Call Google Gemini.  `contents` may be a plain str or a list of Parts."""
    gemini = _get_gemini_client()
    if not gemini:
        raise RuntimeError("GEMINI_API_KEY not set.")
    config = types.GenerateContentConfig(temperature=temperature)
    if system_instruction:
        config.system_instruction = system_instruction
    if response_json:
        config.response_mime_type = "application/json"
    response = await gemini.aio.models.generate_content(
        model=model, contents=contents, config=config,
    )
    return response.text


# ──────────────────────────────────────────
# Z AI client (OpenAI-compatible)
# ──────────────────────────────────────────

ZAI_BASE_URL = "https://api.z.ai/api/paas/v4"
ZAI_DEFAULT_MODEL = "glm-5"
ZAI_MODEL_MAP = {
    "gemini-2.5-flash": "glm-5",
    "gemini-2.5-pro": "glm-5",
    "gemini-2.0-flash": "glm-5",
}


def _get_zai_key() -> Optional[str]:
    return os.environ.get("ZAI_API_KEY") or os.environ.get("Z_AI_API_KEY")


def _extract_text(contents) -> str:
    """Extract plain text from contents (may be str or list of Parts)."""
    if isinstance(contents, str):
        return contents
    if isinstance(contents, list):
        return "\n".join(str(p) for p in contents if isinstance(p, str))
    return str(contents)


async def _call_zai(
    model: str,
    contents,
    system_instruction: str = "",
    temperature: float = 0.2,
    response_json: bool = False,
) -> str:
    """Call Z AI's OpenAI-compatible chat completions API."""
    api_key = _get_zai_key()
    if not api_key:
        raise RuntimeError("ZAI_API_KEY not set.")
    zai_model = ZAI_MODEL_MAP.get(model, ZAI_DEFAULT_MODEL)
    text = _extract_text(contents)
    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": text})
    payload = {"model": zai_model, "messages": messages, "temperature": temperature}
    if response_json:
        payload["response_format"] = {"type": "json_object"}
    headers = {
        "Content-Type": "application/json",
        "Accept-Language": "en-US,en",
        "Authorization": f"Bearer {api_key}",
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(f"{ZAI_BASE_URL}/chat/completions", json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


# ──────────────────────────────────────────
# OpenAI client
# ──────────────────────────────────────────

OPENAI_DEFAULT_MODEL = "gpt-4o"
OPENAI_MODEL_MAP = {
    "gemini-2.5-flash": "gpt-4o",
    "gemini-2.5-pro": "gpt-4o",
    "gemini-2.0-flash": "gpt-4o-mini",
}


def _get_openai_key() -> Optional[str]:
    return os.environ.get("OPENAI_API_KEY")


async def _call_openai(
    model: str,
    contents,
    system_instruction: str = "",
    temperature: float = 0.2,
    response_json: bool = False,
) -> str:
    """Call OpenAI's chat completions API."""
    api_key = _get_openai_key()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set.")
    oai_model = OPENAI_MODEL_MAP.get(model, OPENAI_DEFAULT_MODEL)
    text = _extract_text(contents)
    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": text})
    payload = {"model": oai_model, "messages": messages, "temperature": temperature}
    if response_json:
        payload["response_format"] = {"type": "json_object"}
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


# ──────────────────────────────────────────
# Unified generate with fallback
# ──────────────────────────────────────────

# Models that must always go through the Gemini API (no fallback to other providers)
_GEMINI_ONLY_MODELS = {"gemini-robotics-er-1.5-preview"}

# (name, key_checker, caller)
_PROVIDERS = [
    ("Z AI", _get_zai_key, _call_zai),
    ("OpenAI", _get_openai_key, _call_openai),
    ("Gemini", lambda: os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"), _call_gemini),
]


async def generate(
    model: str,
    contents,
    system_instruction: str = "",
    temperature: float = 0.2,
    response_json: bool = True,
) -> str:
    """
    Generate text from LLM with automatic fallback.
    Tries Z AI → OpenAI → Gemini. Falls back on 429/500/503.
    """
    # Gemini-only models skip the fallback chain and go directly to Gemini.
    if model in _GEMINI_ONLY_MODELS:
        return await _call_gemini(
            model=model, contents=contents,
            system_instruction=system_instruction,
            temperature=temperature, response_json=response_json,
        )

    last_error = None
    for name, has_key, caller in _PROVIDERS:
        if not has_key():
            continue
        try:
            print(f"[llm_client] Using {name}")
            return await caller(
                model=model,
                contents=contents,
                system_instruction=system_instruction,
                temperature=temperature,
                response_json=response_json,
            )
        except Exception as e:
            last_error = e
            error_str = str(e)
            is_fallback = any(str(code) in error_str for code in _FALLBACK_CODES)
            if not is_fallback:
                raise
            print(f"[llm_client] {name} failed ({error_str[:100]}), trying next...")

    raise RuntimeError(
        f"All LLM providers failed. Last error: {last_error}. "
        "Set ZAI_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY in .env"
    )
