import json
import re
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from .config import BACKGROUND_KEEP_ALIVE, CODER_MODEL, OLLAMA_GENERATE_URL


ANSI_PATTERN = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def strip_ansi(text: str) -> str:
    if not text:
        return ""
    return ANSI_PATTERN.sub("", text)


class LocalModel:
    """
    Ollama-backed local model adapter.

    Step 13 changes model access from `ollama run` subprocess calls to the
    Ollama HTTP API with stream=false. This avoids terminal/TTY rendering and
    prevents cursor-control escape sequences from being captured as model text.
    """

    def __init__(
        self,
        model: str = CODER_MODEL,
        endpoint: str = OLLAMA_GENERATE_URL,
        timeout: int = 600,
        temperature: float = 0.15,
        num_ctx: int = 16384,
        num_predict: int | None = None,
    ):
        self.model = model
        self.endpoint = endpoint
        self.timeout = timeout
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.num_predict = num_predict

    def generate(self, prompt: str, response_format: str | None = None) -> str:
        options: Dict[str, Any] = {
            "temperature": self.temperature,
            "num_ctx": self.num_ctx,
        }
        if self.num_predict is not None:
            options["num_predict"] = self.num_predict
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": BACKGROUND_KEEP_ALIVE,
            "options": options,
        }
        if response_format:
            # Ollama supports format="json" to constrain the model to emit a
            # syntactically valid JSON value. Use sparingly — it slows
            # generation but eliminates a whole class of parse failures.
            payload["format"] = response_format

        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8", errors="replace")
        except urllib.error.URLError as exc:
            raise RuntimeError(
                "Ollama HTTP API request failed. Make sure Ollama is running "
                "on localhost:11434. Original error: {}".format(exc)
            )
        except TimeoutError:
            raise RuntimeError("model generation timed out")

        try:
            result = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Ollama HTTP API returned invalid JSON: {}\n\nRAW BODY:\n{}".format(
                    exc,
                    body[:4000],
                )
            )

        if result.get("error"):
            raise RuntimeError(str(result.get("error")))

        text = result.get("response") or ""
        return strip_ansi(text)
