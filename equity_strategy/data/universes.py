"""Constituent lists of the supported stock universes."""
import io
import time
from pathlib import Path

import pandas as pd

from equity_strategy.config import dataset_dir
from equity_strategy.definitions import StockUniverses

SP500_WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
STOXXE600_COMPONENT_FILE = dataset_dir(StockUniverses.STOXXE600) / "component_list.txt"


def get_tickers(dataset: str) -> list[str]:
    """Current constituents of ``dataset`` as listed by the index provider."""
    if dataset == StockUniverses.STOXXE600:
        return read_stoxxe600_component_file(STOXXE600_COMPONENT_FILE)
    if dataset == StockUniverses.SP500:
        return fetch_sp500_tickers()
    raise ValueError(f"Unknown stock universe: {dataset!r}")


def to_yahoo_symbol(symbol: str) -> str:
    """Map an index-provider symbol to its Yahoo Finance ticker (Swiss ``.S`` becomes ``.SW``)."""
    if symbol.endswith(".S"):
        return f"{symbol[:-2]}.SW"
    return symbol


def read_stoxxe600_component_file(path: Path) -> list[str]:
    """Parse a STOXX selection list saved as text; the ticker is the second column."""
    with open(path) as file:
        tickers = {line.split()[1] for line in file if line.strip()}
    return sorted(tickers)


def fetch_sp500_tickers() -> list[str]:
    """Scrape the S&P 500 constituents from Wikipedia through a Chrome browser session."""
    # Imported lazily: Selenium is only needed for this download, not for backtesting.
    from selenium import webdriver

    browser = webdriver.Chrome()
    try:
        browser.get(SP500_WIKIPEDIA_URL)
        time.sleep(1)
        page_html = browser.page_source
    finally:
        browser.quit()
    constituents = pd.read_html(io.StringIO(page_html), header=0)[0]
    return constituents["Symbol"].tolist()
