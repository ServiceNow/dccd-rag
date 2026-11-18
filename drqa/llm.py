"""
LLM API wrapper for conflict dataset generation.
Supports OpenAI models and OpenRouter models (prefix "anthropic/", "google/", etc.).
"""

import logging
import os
import time

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI, APIError, RateLimitError, APIConnectionError

# Default model
DEFAULT_MODEL = "anthropic/claude-opus-4-5"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

logger = logging.getLogger(__name__)


def _make_client(model: str) -> OpenAI:
    """Return an OpenAI-compatible client for the given model."""
    # Route through OpenRouter for any model that contains a provider prefix (e.g. "anthropic/...")
    if "/" in model:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENROUTER_API_KEY not set for OpenRouter model")
        return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
    return OpenAI()


def prompt_llm(
    prompt: str,
    model: str = DEFAULT_MODEL,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> str:
    """
    Call LLM with the given prompt, with retry logic for robustness.

    Use OpenRouter for any model name containing "/" (e.g. "anthropic/claude-opus-4-5").
    Use the OpenAI API for bare model names (e.g. "gpt-4o-mini").

    Args:
        prompt: The prompt to send to the LLM
        model: The model to use
        max_retries: Maximum number of retry attempts
        retry_delay: Base delay between retries (exponential backoff)

    Returns:
        The LLM's response text

    Raises:
        Exception: If all retries fail
    """
    client = _make_client(model)
    last_error = None

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )

            content = response.choices[0].message.content
            if content is None:
                raise ValueError("LLM returned empty response")
            return content

        except RateLimitError as e:
            last_error = e
            wait_time = retry_delay * (2 ** attempt)
            logger.warning(f"Rate limit hit, waiting {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(wait_time)

        except APIConnectionError as e:
            last_error = e
            wait_time = retry_delay * (2 ** attempt)
            logger.warning(f"Connection error, retrying in {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(wait_time)

        except APIError as e:
            last_error = e
            if e.status_code and e.status_code >= 500:
                wait_time = retry_delay * (2 ** attempt)
                logger.warning(f"Server error {e.status_code}, retrying in {wait_time:.1f}s")
                time.sleep(wait_time)
            else:
                raise

        except Exception as e:
            last_error = e
            logger.error(f"Unexpected error: {e}")
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                time.sleep(wait_time)
            else:
                raise

    raise Exception(f"Failed after {max_retries} retries. Last error: {last_error}")

