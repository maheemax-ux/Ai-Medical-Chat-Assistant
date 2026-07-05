
import requests

from config import (
    LLM_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    LLM_TIMEOUT_SECONDS,
)

NO_LLM_MARKER = "[NO_LLM]"
LLM_ERROR_PREFIX = "[LLM ERROR]"


class LLMClient:
    def __init__(self, provider: str = LLM_PROVIDER):
        self.provider = provider

    @property
    def is_configured(self) -> bool:
        if self.provider == "openai":
            return bool(OPENAI_API_KEY)
        if self.provider == "anthropic":
            return bool(ANTHROPIC_API_KEY)
        if self.provider == "gemini":
            return bool(GEMINI_API_KEY)
        return False  # "none" or unrecognized provider

    def is_reachable(self) -> bool:
        
        return self.is_configured

    def chat(self, messages: list, temperature: float = LLM_TEMPERATURE,
             max_tokens: int = LLM_MAX_TOKENS) -> str:
        
        if not self.is_configured:
            return f"{NO_LLM_MARKER} No LLM provider configured (LLM_PROVIDER='{self.provider}')."

        try:
            if self.provider == "openai":
                return self._chat_openai(messages, temperature, max_tokens)
            elif self.provider == "anthropic":
                return self._chat_anthropic(messages, temperature, max_tokens)
            elif self.provider == "gemini":
                return self._chat_gemini(messages, temperature, max_tokens)
            else:
                return f"{NO_LLM_MARKER} Unknown LLM_PROVIDER '{self.provider}'."
        except requests.exceptions.ConnectionError:
            return f"{LLM_ERROR_PREFIX} Could not connect to the {self.provider} API. Check your internet connection."
        except Exception as e:
            return f"{LLM_ERROR_PREFIX} {e}"


    def _chat_openai(self, messages, temperature, max_tokens) -> str:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": OPENAI_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=LLM_TIMEOUT_SECONDS)
        if not resp.ok:
            try:
                detail = resp.json()
            except ValueError:
                detail = resp.text
            return f"{LLM_ERROR_PREFIX} {resp.status_code} from OpenAI: {detail}"
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

    def _chat_anthropic(self, messages, temperature, max_tokens) -> str:
 
        system_prompt = ""
        chat_messages = []
        for m in messages:
            if m["role"] == "system":
                system_prompt += (m["content"] + "\n")
            else:
                chat_messages.append(m)

        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        payload = {
            "model": ANTHROPIC_MODEL,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system_prompt.strip(),
            "messages": chat_messages,
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=LLM_TIMEOUT_SECONDS)
        if not resp.ok:
            try:
                detail = resp.json()
            except ValueError:
                detail = resp.text
            return f"{LLM_ERROR_PREFIX} {resp.status_code} from Anthropic: {detail}"
        data = resp.json()
        return "".join(block.get("text", "") for block in data.get("content", [])).strip()

    def _chat_gemini(self, messages, temperature, max_tokens) -> str:
        system_prompt = ""
        contents = []
        for m in messages:
            if m["role"] == "system":
                system_prompt += (m["content"] + "\n")
            else:
                role = "model" if m["role"] == "assistant" else "user"
                contents.append({"role": role, "parts": [{"text": m["content"]}]})

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
        )
        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system_prompt.strip():
            payload["systemInstruction"] = {"parts": [{"text": system_prompt.strip()}]}

        headers = {"Content-Type": "application/json"}
        resp = requests.post(url, json=payload, headers=headers, timeout=LLM_TIMEOUT_SECONDS)
        if not resp.ok:
            try:
                detail = resp.json()
            except ValueError:
                detail = resp.text
            return f"{LLM_ERROR_PREFIX} {resp.status_code} from Gemini: {detail}"

        data = resp.json()
        try:
            candidate = data["candidates"][0]
            parts = candidate["content"]["parts"]
            return "".join(p.get("text", "") for p in parts).strip()
        except (KeyError, IndexError):
            # Common cause: response was blocked by safety filters and has
            # no "content" field, only a finishReason.
            finish_reason = data.get("candidates", [{}])[0].get("finishReason", "unknown")
            return f"{LLM_ERROR_PREFIX} Gemini returned no content (finishReason: {finish_reason})."
