import os
from typing import Any, Dict, Optional
from urllib.parse import urljoin

import requests


class FXMacroDataProvider:
    """Optional macro-data provider for crypto workflows."""

    DEFAULT_BASE_URL = "https://api.fxmacrodata.com/v1/"

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = 30,
    ) -> None:
        self.api_key = (
            api_key
            or os.getenv("FXMACRODATA_API_KEY")
            or os.getenv("FXMD_API_KEY")
            or ""
        )
        self.base_url = base_url.rstrip("/") + "/"
        self.timeout = timeout

    @classmethod
    def from_env(cls) -> "FXMacroDataProvider":
        return cls()

    def request(self, path: str, **params: Any) -> Dict[str, Any]:
        query = {key: value for key, value in params.items() if value is not None}
        if self.api_key:
            query["api_key"] = self.api_key
        url = urljoin(self.base_url, path.lstrip("/"))
        response = requests.get(url, params=query, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def data_catalogue(self, currency: str) -> Dict[str, Any]:
        return self.request(f"data_catalogue/{currency.lower()}")

    def calendar(self, currency: str = "usd", **params: Any) -> Dict[str, Any]:
        return self.request(f"calendar/{currency.lower()}", **params)

    def announcements(
        self,
        currency: str = "usd",
        indicator: str = "policy_rate",
        **params: Any,
    ) -> Dict[str, Any]:
        return self.request(
            f"announcements/{currency.lower()}/{indicator}",
            **params,
        )

    def forex(self, base: str, quote: str = "usd", **params: Any) -> Dict[str, Any]:
        return self.request(
            f"forex/{base.lower()}/{quote.lower()}",
            **params,
        )

    def commodity(self, indicator: str, **params: Any) -> Dict[str, Any]:
        return self.request(f"commodities/{indicator}", **params)

    def commodities_latest(self, **params: Any) -> Dict[str, Any]:
        return self.request("commodities/latest", **params)

    def market_sessions(self, **params: Any) -> Dict[str, Any]:
        return self.request("market_sessions", **params)

    def risk_sentiment(self, **params: Any) -> Dict[str, Any]:
        return self.request("risk_sentiment", **params)

    def market_context(
        self,
        quote_currency: str = "usd",
        indicator: str = "policy_rate",
        limit: int = 10,
    ) -> Dict[str, Any]:
        currency = quote_currency.lower()
        return {
            "catalogue": self.data_catalogue(currency),
            "calendar": self.calendar(currency, limit=limit),
            "announcements": self.announcements(
                currency,
                indicator=indicator,
                limit=limit,
            ),
            "market_sessions": self.market_sessions(),
            "risk_sentiment": self.risk_sentiment(limit=limit),
        }
