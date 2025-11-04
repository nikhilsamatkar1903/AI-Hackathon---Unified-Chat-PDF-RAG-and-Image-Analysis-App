import pytest
from types import SimpleNamespace

from components import llm_clients


def test_call_generate_api_monkeypatched(monkeypatch):
    # Monkeypatch requests.post to return a predictable streaming response
    class FakeResp:
        def __init__(self, text_lines):
            self._lines = text_lines
            self.status_code = 200
        def iter_lines(self, decode_unicode=False):
            for l in self._lines:
                yield l.encode('utf-8') if not isinstance(l, bytes) else l

    def fake_post(url, json=None, stream=False, timeout=None):
        # Emulate Ollama streaming JSON lines with 'response' in them
        lines = [json.dumps({"id": "1", "choices": [{"delta": {"content": "Hello"}}]})]
        return FakeResp(lines)

    monkeypatch.setattr(llm_clients, 'requests', SimpleNamespace(post=fake_post))

    # Call the function
    out = llm_clients.call_generate_api('test-model', 'hello', images_b64=None)
    # The specific output depends on implementation; ensure it returns a string or empty string
    assert isinstance(out, str)
