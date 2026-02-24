"""HuggingFace Space adapter base — calls Gradio /predict endpoint.

Uses gradio_client.Client instead of InferenceClient.  The Space wraps
a model behind a simple message → response Gradio interface.
Inherits all prompt-building and response-parsing logic from HFBaseAdapter.
"""

import logging
import re
import time
import requests
from typing import Optional

from src.adapters.hf_base import HFBaseAdapter, _resolve_hf_token

logger = logging.getLogger(__name__)


class SpaceBaseAdapter(HFBaseAdapter):
    """Adapter for models deployed as HuggingFace Spaces (Gradio).

    Overrides ``_get_client()`` and ``_chat_completion()`` to use
    ``gradio_client.Client``.  All prompt building and response parsing
    is inherited from :class:`HFBaseAdapter`.
    """

    def __init__(
        self,
        model_id: str,
        model_name: str,
        supported_stages: list[str],
        hf_model_id: str,
        space_id: str,
        api_name: str = "/doctor_infer",
        timeout_seconds: int = 30,
        max_tokens: int = 2048,
        space_model_name: str = "MedGemma 4B"
    ):
        super().__init__(
            model_id=model_id,
            model_name=model_name,
            supported_stages=supported_stages,
            hf_model_id=hf_model_id,
            timeout_seconds=timeout_seconds,
            max_tokens=max_tokens,
        )
        self._space_id = space_id
        self._api_name = api_name
        self._space_model_name = space_model_name
        self._client = None
        self._current_token: Optional[str] = None

    @property
    def space_id(self) -> str:
        return self._space_id

    def _get_client(self):
        """Lazily initialize the gradio_client.Client with intense logging."""
        from gradio_client import Client

        token = _resolve_hf_token()
        if not token:
            raise RuntimeError("HF_TOKEN not set.")
            
        if self._client is None or token != self._current_token:
            print(f"DEBUG: [1/4] Starting connection to {self._space_id}...")
            
            # Dedicated GPU Spaces might be slow to respond. 
            # We use raw requests to 'ping' the space first and wake it up.
            ping_url = f"https://{self._space_id.replace('/', '-')}.hf.space/"
            print(f"DEBUG: [2/4] Pinging Space URL: {ping_url}")
            try:
                # 30s timeout for the ping
                requests.get(ping_url, headers={"Authorization": f"Bearer {token}"}, timeout=30)
                print("DEBUG: [3/4] Space responded to ping.")
            except Exception as e:
                print(f"DEBUG: [!] Ping warning (non-fatal): {e}")

            print(f"DEBUG: [4/4] Finalizing Gradio Client init...")
            for attempt in range(2):
                try:
                    # No timeout arg here as it's not supported by user's gradio_client version
                    self._client = Client(self._space_id, token=token)
                    self._current_token = token
                    print(f"DEBUG: [DONE] Connected to {self._space_id} successfully.")
                    return self._client
                except Exception as e:
                    print(f"DEBUG: [!] Init attempt {attempt+1} failed: {e}")
                    if attempt < 1:
                        print("DEBUG: Retrying in 5s...")
                        time.sleep(5)
            
            raise RuntimeError(f"Could not connect to Space {self._space_id} after retries.")
            
        return self._client

    def _chat_completion(self, messages: list[dict[str, str]]) -> str:
        """Call the Space with (Model Choice, Image, Text) inputs."""
        client = self._get_client()
        prompt = self._flatten_messages(messages)

        print(f"DEBUG: [INFERENCE] Calling {self._space_id} [{self._space_model_name}]...")
        start_time = time.time()

        try:
            result = client.predict(
                self._space_model_name,
                None,
                prompt,
                api_name=self._api_name
            )
            elapsed = time.time() - start_time
            raw = str(result)
            print(f"DEBUG: [SUCCESS] Received response in {elapsed:.1f}s ({len(raw)} chars)")

            # Strip echoed prompt from response.
            # Space models (especially MedGemma) echo input + generation.
            cleaned = self._strip_echo(raw, prompt)
            return cleaned
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"DEBUG: [ERROR] Space call failed after {elapsed:.1f}s: {e}")
            logger.error(f"Space call failed for {self.model_id}: {e}")
            raise

    @staticmethod
    def _flatten_messages(messages: list[dict[str, str]]) -> str:
        """Flatten chat messages into a single prompt string.

        Uses plain text concatenation instead of [System]/[User] markers,
        because Space models treat the input as raw text completion and
        don't understand chat role markers.
        """
        system_parts = []
        user_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system_parts.append(content)
            else:
                user_parts.append(content)

        parts = []
        if system_parts:
            parts.append("Instructions:\n" + "\n".join(system_parts))
        if user_parts:
            parts.append("\n".join(user_parts))
        return "\n\n".join(parts)

    @staticmethod
    def _strip_echo(response: str, prompt: str) -> str:
        """Strip the echoed prompt from the model's response.

        Space models (MedGemma batch_decode, BioMistral, Qwen) often return
        the full input followed by the generation. We detect and strip it.
        """
        if not response or not prompt:
            return response

        # 1. Direct echo: response starts with the prompt text
        if response.startswith(prompt):
            stripped = response[len(prompt):].strip()
            if stripped:
                return stripped

        # 2. Partial echo: response starts with a significant chunk of the prompt
        # Use the last 100 chars of the prompt as an anchor
        anchor = prompt[-100:].strip() if len(prompt) > 100 else prompt.strip()
        idx = response.find(anchor)
        if idx >= 0:
            after = response[idx + len(anchor):].strip()
            if after:
                return after

        # 3. No echo detected — return as-is
        return response