"""
Tests for Task 4.1 — ollama_client.generate().

All HTTP calls mocked via responses library (or unittest.mock).
No Ollama server needed.
"""

from unittest.mock import patch, MagicMock

import pytest
import requests

from backend.app.llm.ollama_client import generate, OllamaUnavailable, _build_prompt


# ─── helpers ──────────────────────────────────────────────────────────────────


def _mock_response(text: str = "The penalty is 2% per month.", status: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = {"response": text}
    resp.raise_for_status = MagicMock()
    if status >= 400:
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=resp
        )
        resp.text = "Model not found"
    return resp


# ─── generate() happy path ────────────────────────────────────────────────────


class TestGenerateHappyPath:
    def test_returns_string(self):
        with patch("requests.post", return_value=_mock_response()):
            result = generate("What is the penalty?")
        assert isinstance(result, str)

    def test_response_stripped(self):
        with patch("requests.post", return_value=_mock_response("  hello world  ")):
            result = generate("question")
        assert result == "hello world"

    def test_model_in_payload(self):
        with patch("requests.post", return_value=_mock_response()) as mock_post:
            generate("q", model="phi4")
        payload = mock_post.call_args.kwargs["json"]
        assert payload["model"] == "phi4"

    def test_stream_false(self):
        """Non-streaming must always be set — streaming adds complexity."""
        with patch("requests.post", return_value=_mock_response()) as mock_post:
            generate("q")
        payload = mock_post.call_args.kwargs["json"]
        assert payload["stream"] is False

    def test_context_included_in_prompt(self):
        with patch("requests.post", return_value=_mock_response()) as mock_post:
            generate("What is the penalty?", context=["Clause 8.3: 2% monthly."])
        payload = mock_post.call_args.kwargs["json"]
        assert "Clause 8.3" in payload["prompt"]
        assert "What is the penalty?" in payload["prompt"]

    def test_no_context_sends_bare_prompt(self):
        with patch("requests.post", return_value=_mock_response()) as mock_post:
            generate("Simple question")
        payload = mock_post.call_args.kwargs["json"]
        assert payload["prompt"] == "Simple question"

    def test_timeout_passed_to_requests(self):
        with patch("requests.post", return_value=_mock_response()) as mock_post:
            generate("q", timeout=90)
        assert mock_post.call_args.kwargs["timeout"] == 90


# ─── error handling ──────────────────────────────────────────────────────────


class TestErrors:
    def test_connection_error_raises_ollama_unavailable(self):
        with patch("requests.post", side_effect=requests.exceptions.ConnectionError()):
            with pytest.raises(OllamaUnavailable, match="ollama serve"):
                generate("question")

    def test_timeout_raises_timeout_error(self):
        with patch("requests.post", side_effect=requests.exceptions.Timeout()):
            with pytest.raises(TimeoutError, match="timed out"):
                generate("question")

    def test_http_error_raises_ollama_unavailable(self):
        with patch("requests.post", return_value=_mock_response(status=404)):
            with pytest.raises(OllamaUnavailable, match="HTTP"):
                generate("question")

    def test_ollama_unavailable_is_runtime_error(self):
        assert issubclass(OllamaUnavailable, RuntimeError)


# ─── _build_prompt helper ────────────────────────────────────────────────────


class TestBuildPrompt:
    def test_no_context_returns_bare_question(self):
        result = _build_prompt("What is the penalty?", [])
        assert result == "What is the penalty?"

    def test_context_prepended(self):
        result = _build_prompt("Question?", ["Chunk A content."])
        assert "Chunk A content." in result
        assert "Question?" in result

    def test_multiple_chunks_numbered(self):
        result = _build_prompt("Q?", ["chunk 1", "chunk 2"])
        assert "excerpt 1" in result
        assert "excerpt 2" in result

    def test_question_after_context(self):
        result = _build_prompt("My question", ["some chunk"])
        # Question should appear after the context block
        ctx_pos = result.find("some chunk")
        q_pos = result.find("My question")
        assert q_pos > ctx_pos
