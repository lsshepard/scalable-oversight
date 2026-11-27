"""Claude backend for the repeated-round game, through the official Anthropic SDK.

Credentials.  The key is read from EXPERIMENT_ANTHROPIC_API_KEY and from nowhere else,
and the client is pinned to api.anthropic.com.  Every ambient ANTHROPIC_* variable is
removed from this process before the client is built, so that the credentials, base
URL or headers of a surrounding tool session can never be used for the experiment, and
the experiment key is never sent anywhere but the Claude API.  To use a key that lives
in ANTHROPIC_API_KEY on your own machine:

    EXPERIMENT_ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY python experiments/exp6_llm_rounds.py ...

Request shape (current models: Claude Sonnet 5.5, Haiku 5.5, Opus 5.5).
  * No sampling parameters: the API rejects temperature / top_p / top_k, so every run
    uses the model's default sampling.
  * Thinking is adaptive and cannot be switched off on these models.  Thinking tokens
    are billed as output tokens.  `display: summarized` returns a readable summary of
    the reasoning, which is stored with each record.
  * `effort` controls how much the model thinks (low | medium | high | xhigh | max).
    The model default differs by model (Sonnet 5.5: high, Haiku 5.5: medium), so set it
    explicitly when comparing models.
  * No fallback model.  A reply declined by the API's safety classifiers comes back
    with stop_reason "refusal" and is recorded as such.  Routing it to another model
    would change which model produced the data.
"""
from __future__ import annotations

import os
from typing import Optional

from .rounds import Complete, FatalBackendError, Reply

KEY_ENV = "EXPERIMENT_ANTHROPIC_API_KEY"
API_URL = "https://api.anthropic.com"

# USD per million tokens (input, output).  Anthropic list prices, checked 2026-11-26
# against a price table cached 2026-11-03.  Used only for the spending cap and the cost
# printout; re-check before relying on them.
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
    # earlier generations, same price table
    "claude-sonnet-5": (2.00, 10.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-opus-4-6": (5.00, 25.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


def request_shape(model: str, effort: Optional[str], max_tokens: int) -> dict:
    """Model-dependent request fields.

    Claude Haiku 4.5 has neither adaptive thinking nor `effort`; it is called without
    thinking and reasons in its visible reply.  The 4.6 models take adaptive thinking
    without the `display` field (their summaries are on by default).  Later models take
    adaptive thinking with summaries switched on."""
    req = dict(model=model, max_tokens=max_tokens)
    if model.startswith("claude-haiku-4-5"):
        return req
    if model in ("claude-opus-4-6", "claude-sonnet-4-6"):
        req["thinking"] = {"type": "adaptive"}
    else:
        req["thinking"] = {"type": "adaptive", "display": "summarized"}
    if effort:
        req["output_config"] = {"effort": effort}
    return req


def claude_backend(model: str, effort: Optional[str] = None, max_tokens: int = 16000,
                   api_key: Optional[str] = None, base_url: str = API_URL,
                   max_retries: int = 10, timeout: float = 300.0) -> Complete:
    """Returns complete(system, user) -> Reply.  `api_key` and `base_url` exist for
    tests; real runs take the key from EXPERIMENT_ANTHROPIC_API_KEY."""
    import anthropic

    key = api_key or os.environ.get(KEY_ENV)
    if not key:
        raise FatalBackendError(
            f"{KEY_ENV} is not set. The experiment reads its Claude API key from this "
            f"variable only (never from ANTHROPIC_API_KEY or the session's own login).")
    for name in [n for n in os.environ if n.startswith("ANTHROPIC_")]:
        del os.environ[name]
    client = anthropic.Anthropic(api_key=key, base_url=base_url,
                                 max_retries=max_retries, timeout=timeout)
    request = request_shape(model, effort, max_tokens)

    def complete(system: str, user: str) -> Reply:
        try:
            msg = client.messages.create(
                system=system, messages=[{"role": "user", "content": user}], **request)
        except anthropic.AuthenticationError as e:
            raise FatalBackendError(f"The Claude API rejected the key in {KEY_ENV}.") from e
        except anthropic.PermissionDeniedError as e:
            raise FatalBackendError(
                f"The key in {KEY_ENV} may not use {model!r}: {e.message}") from e
        except anthropic.NotFoundError as e:
            raise FatalBackendError(f"Unknown model {model!r}: {e.message}") from e
        except anthropic.BadRequestError as e:
            raise FatalBackendError(f"The Claude API rejected the request: {e.message}") from e
        # Rate limits, overload and network errors are retried by the SDK; if they
        # persist they propagate and fail this trajectory only.
        text = "".join(b.text for b in msg.content if b.type == "text")
        thinking = "\n".join(b.thinking for b in msg.content
                             if b.type == "thinking" and b.thinking)
        return Reply(text=text, thinking=thinking, stop_reason=msg.stop_reason or "",
                     in_tokens=msg.usage.input_tokens, out_tokens=msg.usage.output_tokens)

    return complete
