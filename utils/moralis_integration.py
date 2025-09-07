# utils/moralis_integration.py
import os

class MoralisProvider:
    def __init__(self, api_key: str | None):
        self.api_key = api_key


def get_moralis_provider() -> MoralisProvider:
    """Return a simple provider that holds the API key from env.
    Real integration can be added later without breaking callers.
    """
    api_key = os.getenv("MORALIS_API_KEY")
    return MoralisProvider(api_key)