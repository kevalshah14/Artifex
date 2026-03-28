"""
Unified LLM client with automatic fallback.

Primary:  Google Gemini (via google-genai SDK)
Fallback: Z AI (OpenAI-compatible API at api.z.ai)

Falls back automatically on 429 (rate limit), 503 (unavailable), or connection errors.
"""

import json
import os
import httpx
from typing import Optional

from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', 'sim', '.env'))
load_dotenv()

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


# ──────────────────────────────────────────
# Z AI client (OpenAI-compatible)
# ──────────────────────────────────────────

ZAI_BASE_URL = "https://api.z.ai/api/paas/v4"
ZAI_DEFAULT_MODEL = "glm-5"

# Model mapping: Gemini model name → Z AI equivalent
ZAI_MODEL_MAP = {
    "gemini-2.5-flash": "glm-5",
    "gemini-2.5-pro": "glm-5",
    "gemini-2.0-flash": "glm-5",
}


def _get_zai_key() -> Optional[str]:
    return os.environ.get("ZAI_API_KEY") or os.environ.get("Z_AI_API_KEY")


async def _call_zai(
    model: str,
    contents: str,
    system_instruction: str = "",
    temperature: float = 0.2,
    response_json: bool = False,
) -> str:
    """Call Z AI's OpenAI-compatible chat completions API."""
    api_key = _get_zai_key()
    if not api_key:
        raise RuntimeError("ZAI_API_KEY not set. Add it to .env for fallback.")

    zai_model = ZAI_MODEL_MAP.get(model, ZAI_DEFAULT_MODEL)

    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": contents})

    payload = {
        "model": zai_model,
        "messages": messages,
        "temperature": temperature,
    }
    if response_json:
        payload["response_format"] = {"type": "json_object"}

    headers = {
        "Content-Type": "application/json",
        "Accept-Language": "en-US,en",
        "Authorization": f"Bearer {api_key}",
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{ZAI_BASE_URL}/chat/completions",
            json=payload,
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


# ──────────────────────────────────────────
# Unified generate with fallback
# ──────────────────────────────────────────

# Errors that trigger fallback
_FALLBACK_CODES = {429, 500, 503}


async def generate(
    model: str,
    contents: str,
    system_instruction: str = "",
    temperature: float = 0.2,
    response_json: bool = True,
) -> str:
    """
    Generate text from LLM with automatic fallback.

    Tries Gemini first. If it fails with rate limit (429), unavailable (503),
    or missing key, falls back to Z AI.

    Returns the raw text response.
    """
    # Try Gemini first
    gemini = _get_gemini_client()
    if gemini:
        try:
            config = types.GenerateContentConfig(
                temperature=temperature,
            )
            if system_instruction:
                config.system_instruction = system_instruction
            if response_json:
                config.response_mime_type = "application/json"

            response = await gemini.aio.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            return response.text
        except Exception as e:
            error_str = str(e)
            # Check if this is a retryable error
            is_fallback = any(str(code) in error_str for code in _FALLBACK_CODES)
            if not is_fallback:
                raise  # Non-retryable error, don't fallback

            print(f"[llm_client] Gemini failed ({error_str[:100]}), falling back to Z AI...")

    # Fallback to Z AI
    if _get_zai_key():
        print(f"[llm_client] Using Z AI ({ZAI_MODEL_MAP.get(model, ZAI_DEFAULT_MODEL)})")
        return await _call_zai(
            model=model,
            contents=contents,
            system_instruction=system_instruction,
            temperature=temperature,
            response_json=response_json,
        )

    raise RuntimeError(
        "Both Gemini and Z AI failed. Set GEMINI_API_KEY or ZAI_API_KEY in .env"
    )
