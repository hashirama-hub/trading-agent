import pytest
import os
from dotenv import load_dotenv

load_dotenv()

@pytest.fixture
def binance_testnet():
    return os.getenv("BINANCE_TESTNET", "true").lower() == "true"

@pytest.fixture
def deepseek_model():
    return os.getenv("DEEPSEEK_MODEL", "deepseek-v4.1-flash")