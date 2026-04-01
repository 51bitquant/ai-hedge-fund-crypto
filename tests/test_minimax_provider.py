"""Unit tests for MiniMax provider integration in get_llm()."""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock

# Add src to path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Mock out problematic imports that may fail on certain Python versions
for mod in [
    "langchain_google_genai",
    "langchain_ollama",
    "langchain.output_parsers",
    "langchain.output_parsers.json",
]:
    sys.modules.setdefault(mod, MagicMock())


class TestMiniMaxProvider:
    """Tests for MiniMax provider in the LLM factory."""

    def setup_method(self):
        """Clear LRU cache before each test."""
        # Remove cached module to allow re-import with fresh mocks
        if "llm" in sys.modules:
            del sys.modules["llm"]

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-minimax-key"})
    def test_minimax_returns_chat_openai_instance(self):
        """MiniMax provider should create a ChatOpenAI instance."""
        with patch("langchain_openai.ChatOpenAI") as mock_cls:
            # Re-import to pick up the mock
            if "llm" in sys.modules:
                del sys.modules["llm"]
            from llm import get_llm

            get_llm.cache_clear()
            get_llm(provider="minimax", model="MiniMax-M2.7")
            mock_cls.assert_called()

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-minimax-key"})
    def test_minimax_default_base_url(self):
        """MiniMax should use api.minimax.io/v1 as default base URL."""
        with patch("langchain_openai.ChatOpenAI") as mock_cls:
            if "llm" in sys.modules:
                del sys.modules["llm"]
            from llm import get_llm

            get_llm.cache_clear()
            get_llm(provider="minimax", model="MiniMax-M2.7")
            call_kwargs = mock_cls.call_args[1]
            assert call_kwargs["base_url"] == "https://api.minimax.io/v1"

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-minimax-key"})
    def test_minimax_custom_base_url(self):
        """MiniMax should accept a custom base URL override."""
        with patch("langchain_openai.ChatOpenAI") as mock_cls:
            if "llm" in sys.modules:
                del sys.modules["llm"]
            from llm import get_llm

            get_llm.cache_clear()
            custom_url = "https://custom.minimax.io/v1"
            get_llm(provider="minimax", model="MiniMax-M2.7", base_url=custom_url)
            call_kwargs = mock_cls.call_args[1]
            assert call_kwargs["base_url"] == custom_url

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-minimax-key"})
    def test_minimax_api_key_from_env(self):
        """MiniMax should read MINIMAX_API_KEY from environment."""
        with patch("langchain_openai.ChatOpenAI") as mock_cls:
            if "llm" in sys.modules:
                del sys.modules["llm"]
            from llm import get_llm

            get_llm.cache_clear()
            get_llm(provider="minimax", model="MiniMax-M2.7")
            call_kwargs = mock_cls.call_args[1]
            assert call_kwargs["api_key"] == "test-minimax-key"

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-minimax-key"})
    def test_minimax_model_name_passed(self):
        """Model name should be passed through to ChatOpenAI."""
        with patch("langchain_openai.ChatOpenAI") as mock_cls:
            if "llm" in sys.modules:
                del sys.modules["llm"]
            from llm import get_llm

            get_llm.cache_clear()
            get_llm(provider="minimax", model="MiniMax-M2.7-highspeed")
            call_kwargs = mock_cls.call_args[1]
            assert call_kwargs["model"] == "MiniMax-M2.7-highspeed"

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-minimax-key"})
    def test_minimax_temperature_set(self):
        """MiniMax should set temperature to 0.7 (within valid range (0, 1])."""
        with patch("langchain_openai.ChatOpenAI") as mock_cls:
            if "llm" in sys.modules:
                del sys.modules["llm"]
            from llm import get_llm

            get_llm.cache_clear()
            get_llm(provider="minimax", model="MiniMax-M2.7")
            call_kwargs = mock_cls.call_args[1]
            assert call_kwargs["temperature"] == 0.7

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-minimax-key"})
    def test_minimax_timeout_and_retries(self):
        """MiniMax should use the same timeout/retry config as other providers."""
        with patch("langchain_openai.ChatOpenAI") as mock_cls:
            if "llm" in sys.modules:
                del sys.modules["llm"]
            from llm import get_llm

            get_llm.cache_clear()
            get_llm(provider="minimax", model="MiniMax-M2.7")
            call_kwargs = mock_cls.call_args[1]
            assert call_kwargs["timeout"] == 120
            assert call_kwargs["max_retries"] == 3

    def test_unsupported_provider_raises(self):
        """Unsupported providers should raise ValueError."""
        if "llm" in sys.modules:
            del sys.modules["llm"]
        from llm import get_llm

        get_llm.cache_clear()
        with pytest.raises(ValueError, match="Unsupported provider"):
            get_llm(provider="nonexistent", model="some-model")

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-minimax-key"})
    def test_minimax_caching(self):
        """Repeated calls with same args should return cached instance."""
        with patch("langchain_openai.ChatOpenAI") as mock_cls:
            if "llm" in sys.modules:
                del sys.modules["llm"]
            from llm import get_llm

            get_llm.cache_clear()
            result1 = get_llm(provider="minimax", model="MiniMax-M2.7")
            result2 = get_llm(provider="minimax", model="MiniMax-M2.7")
            assert result1 is result2
            # Should only be called once due to caching
            assert mock_cls.call_count == 1

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-openai-key"})
    def test_openai_still_works(self):
        """Existing OpenAI provider should still function correctly."""
        with patch("langchain_openai.ChatOpenAI") as mock_cls:
            if "llm" in sys.modules:
                del sys.modules["llm"]
            from llm import get_llm

            get_llm.cache_clear()
            get_llm(provider="openai", model="gpt-4o-mini")
            call_kwargs = mock_cls.call_args[1]
            assert call_kwargs["base_url"] == "https://api.openai.com/v1"
            assert call_kwargs["api_key"] == "test-openai-key"

    @patch.dict(os.environ, {"GROQ_API_KEY": "test-groq-key"})
    def test_groq_still_works(self):
        """Existing Groq provider should still function correctly."""
        with patch("langchain_groq.ChatGroq") as mock_cls:
            if "llm" in sys.modules:
                del sys.modules["llm"]
            from llm import get_llm

            get_llm.cache_clear()
            get_llm(provider="groq", model="llama3-70b")
            mock_cls.assert_called_once()

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-anthropic-key"})
    def test_anthropic_still_works(self):
        """Existing Anthropic provider should still function correctly."""
        with patch("langchain_anthropic.ChatAnthropic") as mock_cls:
            if "llm" in sys.modules:
                del sys.modules["llm"]
            from llm import get_llm

            get_llm.cache_clear()
            get_llm(provider="anthropic", model="claude-3-sonnet")
            mock_cls.assert_called_once()


class TestMiniMaxConfig:
    """Tests for MiniMax-related configuration."""

    def test_config_example_mentions_minimax(self):
        """config.example.yaml should mention minimax as a provider option."""
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "config.example.yaml"
        )
        with open(config_path) as f:
            content = f.read()
        assert "minimax" in content
        assert "MiniMax-M2.7" in content

    def test_readme_mentions_minimax(self):
        """README.md should mention MiniMax as a supported provider."""
        readme_path = os.path.join(os.path.dirname(__file__), "..", "README.md")
        with open(readme_path) as f:
            content = f.read()
        assert "MiniMax" in content
        assert "MINIMAX_API_KEY" in content

    def test_readme_cn_mentions_minimax(self):
        """README_CN.md should mention MiniMax as a supported provider."""
        readme_path = os.path.join(os.path.dirname(__file__), "..", "README_CN.md")
        with open(readme_path) as f:
            content = f.read()
        assert "MiniMax" in content
        assert "MINIMAX_API_KEY" in content


class TestMiniMaxIntegration:
    """Integration tests for MiniMax provider (require MINIMAX_API_KEY).

    These tests verify MiniMax API connectivity using the OpenAI-compatible
    endpoint directly, independent of the langchain installation.
    """

    @pytest.fixture(autouse=True)
    def check_api_key(self):
        if not os.environ.get("MINIMAX_API_KEY"):
            pytest.skip("MINIMAX_API_KEY not set")

    def _api_call(self, model, content, max_tokens=10):
        """Helper to call MiniMax API with retry on transient errors."""
        import json
        import time
        import urllib.request
        import urllib.error

        api_key = os.environ["MINIMAX_API_KEY"]
        url = "https://api.minimax.io/v1/chat/completions"
        payload = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.7,
            "max_tokens": max_tokens,
        }).encode()

        for attempt in range(3):
            try:
                req = urllib.request.Request(
                    url,
                    data=payload,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {api_key}",
                    },
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as e:
                if e.code in (429, 529) and attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                raise

    def test_minimax_api_connectivity(self):
        """MiniMax API should be reachable and respond to chat completions."""
        data = self._api_call("MiniMax-M2.7", "Say 'hello' and nothing else.")
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert len(data["choices"][0]["message"]["content"]) > 0

    def test_minimax_highspeed_api(self):
        """MiniMax-M2.7-highspeed should also respond via the API."""
        data = self._api_call("MiniMax-M2.7-highspeed", "Reply with 'ok'.")
        assert "choices" in data
        assert len(data["choices"]) > 0

    def test_minimax_temperature_constraint(self):
        """MiniMax API should accept temperature in (0, 1] range."""
        data = self._api_call("MiniMax-M2.7", "Say 'test'.", max_tokens=5)
        assert "choices" in data
