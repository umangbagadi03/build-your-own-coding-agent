"""Connect a notebook to a model: a real provider, or a local scripted server.

You don't need to read this file to follow the course. The notebooks build
everything that matters themselves, using plain HTTP requests.

Pick a provider with environment variables (or Colab secrets for the keys):

    (nothing set)                     offline scripted server, no key, no cost
    GROQ_API_KEY=...                  Groq
    LLM_PROVIDER=ollama               local Ollama (first chat model it lists)
    OPENAI_API_KEY=... LLM_MODEL=...  OpenAI
    LLM_PROVIDER=custom LLM_BASE_URL=... LLM_MODEL=... [LLM_API_KEY=...]
                                      any OpenAI-compatible endpoint

LLM_MODEL overrides the model for any provider. LLM_PROVIDER=offline forces
offline mode even when keys are present.

Offline mode starts a small HTTP server on 127.0.0.1 that speaks the same wire
format as the real APIs. The notebook code is identical in both modes; only the
URL changes. The server is NOT a model: it follows a fixed script written for
these lessons, so use a real provider to see real behaviour.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import requests

GROQ_DEFAULT_MODEL = "llama-3.3-70b-versatile"
OLLAMA_URL = "http://localhost:11434"


@dataclass
class LLM:
    provider: str
    base_url: str
    model: str
    api_key: str = field(default="", repr=False)  # never echo keys into notebook output
    offline: bool = False
    timeout: int = 300


def connect(verbose: bool = True) -> LLM:
    """Choose a provider from the environment and return its connection details."""
    llm = _choose_provider()
    if verbose:
        if llm.offline:
            print("OFFLINE MODE: a scripted local server, not a real model.")
            print("The HTTP calls are real. The answers are canned for these lessons.")
            print("Token counts are rough estimates and timings are simulated.")
            print("Set GROQ_API_KEY (free tier) or LLM_PROVIDER=ollama to use a real model.")
        else:
            print(f"Connected to {llm.provider}, model: {llm.model}")
    return llm


def _secret(name: str) -> str:
    """Read a key from the environment, then from Colab secrets if available."""
    value = os.environ.get(name, "")
    if value:
        return value
    try:
        from google.colab import userdata  # type: ignore

        return userdata.get(name) or ""
    except Exception:  # not on Colab, or the secret doesn't exist or isn't shared
        return ""


def _choose_provider() -> LLM:
    provider = os.environ.get("LLM_PROVIDER", "").strip().lower()
    model = os.environ.get("LLM_MODEL", "").strip()

    if not provider:
        if _secret("GROQ_API_KEY"):
            provider = "groq"
        elif _secret("OPENAI_API_KEY") and model:
            provider = "openai"
        else:
            provider = "offline"

    if provider == "offline":
        return LLM("offline", _start_offline_server(), "offline-script", offline=True)

    if provider == "groq":
        key = _secret("GROQ_API_KEY")
        if not key:
            raise RuntimeError("LLM_PROVIDER=groq needs GROQ_API_KEY to be set.")
        return LLM("groq", "https://api.groq.com/openai/v1", model or GROQ_DEFAULT_MODEL, key)

    if provider == "openai":
        key = _secret("OPENAI_API_KEY")
        if not key or not model:
            raise RuntimeError("LLM_PROVIDER=openai needs OPENAI_API_KEY and LLM_MODEL.")
        return LLM("openai", "https://api.openai.com/v1", model, key)

    if provider == "ollama":
        return LLM("ollama", f"{OLLAMA_URL}/v1", model or _first_ollama_model())

    if provider == "custom":
        base_url = os.environ.get("LLM_BASE_URL", "").rstrip("/")
        if not base_url or not model:
            raise RuntimeError("LLM_PROVIDER=custom needs LLM_BASE_URL and LLM_MODEL.")
        return LLM("custom", base_url, model, _secret("LLM_API_KEY"))

    raise RuntimeError(
        f"Unknown LLM_PROVIDER {provider!r}. Use offline, groq, ollama, openai or custom."
    )


def _first_ollama_model() -> str:
    try:
        tags = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5).json()
    except requests.RequestException as error:
        raise RuntimeError("Can't reach Ollama. Is `ollama serve` running?") from error
    names = [m["name"] for m in tags.get("models", []) if "embed" not in m["name"]]
    if not names:
        raise RuntimeError("Ollama has no chat models. Try: ollama pull <model>")
    return names[0]


# ---------------------------------------------------------------------------
# Offline scripted server
# ---------------------------------------------------------------------------

_server_url = None
_server_lock = threading.Lock()
_call_counter = 0


def _start_offline_server() -> str:
    global _server_url
    with _server_lock:
        if _server_url is None:
            server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            _server_url = f"http://127.0.0.1:{server.server_address[1]}/v1"
    return _server_url


def _text(content: str) -> dict:
    return {"content": content, "tool_calls": []}


def _calls(*calls: tuple) -> dict:
    return {"content": None, "tool_calls": list(calls)}


def _tool_history(messages: list) -> list:
    """Return (tool name, arguments, result) for every completed tool call."""
    results = {m.get("tool_call_id"): m.get("content", "") for m in messages if m["role"] == "tool"}
    history = []
    for message in messages:
        for call in message.get("tool_calls") or []:
            function = call["function"]
            try:
                arguments = json.loads(function["arguments"])
            except (TypeError, ValueError):
                arguments = {}
            history.append((function["name"], arguments, results.get(call["id"], "")))
    return history


def _script(messages: list, tools: list) -> dict:
    """Decide the next scripted reply. The lessons' prompts are all this understands."""
    names = {t["function"]["name"] for t in tools or []}
    if "run_tests" in names:
        return _script_bug_fix(messages)
    if {"list_files", "read_file"} <= names:
        return _script_research(messages)
    if {"multiply", "count_words"} & names:
        reply = _script_tools(messages, names)
        if reply:
            return reply
    return _script_chat(messages)


def _script_bug_fix(messages: list) -> dict:
    history = _tool_history(messages)
    used = [name for name, _, _ in history]
    if "list_files" not in used:
        return _calls(("list_files", {}))
    if "run_tests" not in used:
        return _calls(("run_tests", {}))
    if "read_file" not in used:
        return _calls(("read_file", {"path": "cart.py"}), ("read_file", {"path": "test_cart.py"}))
    if "write_file" not in used:
        source = next(r for n, a, r in history if n == "read_file" and a.get("path") == "cart.py")
        fixed = source.replace('item["price"] for item', 'item["price"] * item["quantity"] for item')
        fixed = fixed.replace("if total > 50", "if total >= 50")
        return _calls(("write_file", {"path": "cart.py", "content": fixed}))
    if used.count("run_tests") < 2:
        return _calls(("run_tests", {}))
    last_result = [r for n, _, r in history if n == "run_tests"][-1]
    if last_result.startswith("PASSED"):
        return _text(
            "Fixed two bugs in cart.py: subtotal ignored quantity, and shipping charged "
            "orders of exactly 50. All tests pass now."
        )
    return _text("I applied my scripted fix, but the tests still fail, so this isn't solved.")


def _script_research(messages: list) -> dict:
    history = _tool_history(messages)
    listing = next((r for n, _, r in history if n == "list_files"), None)
    if listing is None:
        return _calls(("list_files", {}))
    files = json.loads(listing)
    read = {a.get("path") for n, a, _ in history if n == "read_file"}
    unread = [f for f in files if f not in read]
    if unread and not read:
        return _calls(("read_file", {"path": unread[0]}))  # look at one file first
    if unread:
        return _calls(*[("read_file", {"path": f}) for f in unread])  # then read the rest together
    notes = " ".join(r for n, _, r in history if n == "read_file")
    deadline = re.search(r"deadline[^.]*?((?:January|February|March|April|May|June) \d+)", notes)
    owner = re.search(r"owner: (\w+)", notes)
    return _text(
        f"The launch deadline is {deadline.group(1) if deadline else 'unknown'}, "
        f"and the owner is {owner.group(1) if owner else 'unknown'}."
    )


def _script_tools(messages: list, names: set) -> dict | None:
    last = messages[-1]
    if last["role"] == "tool":
        return _text(f"The answer is {last['content']}.")
    prompt = last.get("content") or ""
    numbers = re.findall(r"\d+", prompt)
    if "multiply" in names and len(numbers) >= 2:
        return _calls(("multiply", {"a": int(numbers[0]), "b": int(numbers[1])}))
    quoted = re.search(r"['\"](.+?)['\"]", prompt)
    if "count_words" in names and "word" in prompt.lower() and quoted:
        return _calls(("count_words", {"text": quoted.group(1)}))
    return None


def _script_chat(messages: list) -> dict:
    prompt = (messages[-1].get("content") or "").lower()
    system = " ".join(m.get("content") or "" for m in messages if m["role"] == "system").lower()
    users = [m.get("content") or "" for m in messages if m["role"] == "user"]

    if "pirate" in system:
        return _text("Arr, an API be a contract between programs, matey!")
    if "what is my name" in prompt or "what's my name" in prompt:
        for text in users:
            match = re.search(r"my name is (\w+)", text, re.IGNORECASE)
            if match:
                return _text(f"Your name is {match.group(1)}.")
        return _text("I don't know your name. I can't see any earlier messages.")
    match = re.search(r"my name is (\w+)", prompt, re.IGNORECASE)
    if match:
        return _text(f"Nice to meet you, {match.group(1).capitalize()}!")
    if "large language model" in prompt:
        return _text(
            "A large language model is a neural network trained on huge amounts of text "
            "to predict what comes next, which lets it generate useful language."
        )
    if "streaming" in prompt:
        return _text(
            "Streaming sends each piece of the answer as soon as it is generated, so you "
            "see the first words almost immediately instead of staring at a blank screen. "
            "The full answer takes just as long, but the wait feels far shorter because "
            "something is happening the whole time."
        )
    if "hello" in prompt:
        return _text("Bonjour !")
    if "tip" in prompt:
        return _text("Keep each function small enough to explain in a single sentence.")
    return _text(
        "(offline script) I only know the prompts used in these lessons. "
        "Connect a real model to ask me anything else."
    )


def _next_call_id() -> str:
    global _call_counter
    with _server_lock:
        _call_counter += 1
        return f"call_{_call_counter}"


def _estimate_tokens(value) -> int:
    """Rough stand-in for a real tokenizer: about four characters per token."""
    return max(1, len(json.dumps(value)) // 4)


def _pieces(text: str, size: int = 12) -> list:
    return [text[i : i + size] for i in range(0, len(text), size)]


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # keep notebook output clean
        pass

    def _send_json(self, status: int, body: dict) -> None:
        raw = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _error(self, status: int, message: str) -> None:
        self._send_json(status, {"error": {"message": message, "type": "invalid_request_error"}})

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            return self._error(404, f"Unknown path {self.path}")
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
        except ValueError:
            return self._error(400, "Request body is not valid JSON.")
        messages = body.get("messages")
        if not messages:
            return self._error(400, "'messages' is required and can't be empty.")

        tools = body.get("tools") or []
        reply = _script(messages, tools)
        tool_calls = [
            {
                "id": _next_call_id(),
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
            for name, arguments in reply["tool_calls"]
        ]
        usage = {
            "prompt_tokens": _estimate_tokens([messages, tools]),
            "completion_tokens": _estimate_tokens([reply["content"], tool_calls]),
        }
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
        finish_reason = "tool_calls" if tool_calls else "stop"

        if body.get("stream"):
            self._stream(body, reply["content"], tool_calls, finish_reason, usage)
        else:
            # A non-streaming caller waits for the whole answer, so charge it the full time.
            pieces = len(_pieces(reply["content"] or "")) + len(tool_calls)
            time.sleep(0.3 + 0.04 * pieces)
            message = {"role": "assistant", "content": reply["content"]}
            if tool_calls:
                message["tool_calls"] = tool_calls
            self._send_json(
                200,
                {
                    "id": "offline-response",
                    "object": "chat.completion",
                    "model": body.get("model"),
                    "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
                    "usage": usage,
                },
            )

    def _stream(self, body, content, tool_calls, finish_reason, usage) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        def send(delta: dict, reason=None, extra=None) -> None:
            chunk = {
                "id": "offline-response",
                "object": "chat.completion.chunk",
                "model": body.get("model"),
                "choices": [{"index": 0, "delta": delta, "finish_reason": reason}],
            }
            chunk.update(extra or {})
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()

        try:
            time.sleep(0.3)  # simulated time to first token
            send({"role": "assistant", "content": ""})
            for piece in _pieces(content or ""):
                time.sleep(0.04)
                send({"content": piece})
            for index, call in enumerate(tool_calls):
                send(
                    {
                        "tool_calls": [
                            {
                                "index": index,
                                "id": call["id"],
                                "type": "function",
                                "function": {"name": call["function"]["name"], "arguments": ""},
                            }
                        ]
                    }
                )
                for fragment in _pieces(call["function"]["arguments"], 8):
                    time.sleep(0.02)
                    send({"tool_calls": [{"index": index, "function": {"arguments": fragment}}]})
            send({}, reason=finish_reason)
            if (body.get("stream_options") or {}).get("include_usage"):
                send({}, extra={"choices": [], "usage": usage})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # the client stopped reading early, which is allowed
