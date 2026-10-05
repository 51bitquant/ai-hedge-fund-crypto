import unittest
from unittest import mock

from src.utils.fxmacrodata_provider import FXMacroDataError, FXMacroDataProvider


def _response(status_code=200, payload=None):
    response = mock.Mock()
    response.status_code = status_code
    response.json.return_value = payload
    return response


class FXMacroDataProviderTest(unittest.TestCase):
    def setUp(self):
        self.provider = FXMacroDataProvider(api_key="test-key")

    @mock.patch("src.utils.fxmacrodata_provider.requests.get")
    def test_redirect_is_not_followed(self, get):
        get.return_value = _response(302)
        with self.assertRaises(FXMacroDataError) as ctx:
            self.provider.calendar("usd")
        self.assertFalse(get.call_args.kwargs["allow_redirects"])
        self.assertNotIn("test-key", str(ctx.exception))

    @mock.patch("src.utils.fxmacrodata_provider.requests.get")
    def test_error_text_never_contains_key(self, get):
        provider = FXMacroDataProvider(api_key="test\nkey")
        with self.assertRaises(FXMacroDataError) as ctx:
            provider.calendar("usd")
        self.assertNotIn("test\nkey", str(ctx.exception))
        get.assert_not_called()

    @mock.patch("src.utils.fxmacrodata_provider.requests.get")
    def test_error_body_with_200_raises_clean_error(self, get):
        get.return_value = _response(200, {"detail": "Invalid API key"})
        with self.assertRaises(FXMacroDataError):
            self.provider.calendar("usd")
        get.return_value = _response(200, ["unexpected"])
        with self.assertRaises(FXMacroDataError):
            self.provider.calendar("usd")


if __name__ == "__main__":
    unittest.main()
