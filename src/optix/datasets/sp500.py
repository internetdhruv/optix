"""Reusable real-market sample data: S&P 500 prices, returns, and sectors.

Fetched once from the network and cached under ``data/`` at the repo root,
so notebooks across the solver arsenal (simplex today, QP/interior-point
later) can build problems from the same real data without re-downloading it.

Polars-native: everything returned here is a ``pl.DataFrame``. yfinance
itself is pandas-based, so its result is converted immediately, without
depending on pyarrow (see the numpy-array construction in ``load_prices``).
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import requests
import yfinance as yf
from lxml import html

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"
UNIVERSE_PATH = DATA_DIR / "sp500_universe.csv"

WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
USER_AGENT = "Mozilla/5.0 (compatible; optix-datasets/0.1)"


def load_universe(refresh: bool = False) -> pl.DataFrame:
    """Ticker -> GICS sector for the S&P 500 constituents.

    Cached to ``data/sp500_universe.csv`` (checked into the repo) so this
    works offline after the first fetch and stays stable across runs.
    """
    if UNIVERSE_PATH.exists() and not refresh:
        return pl.read_csv(UNIVERSE_PATH)

    response = requests.get(WIKIPEDIA_URL, headers={"User-Agent": USER_AGENT}, timeout=15)
    response.raise_for_status()
    tree = html.fromstring(response.text)
    table = tree.get_element_by_id("constituents")

    tickers, sectors = [], []
    for row in table.xpath(".//tbody/tr")[1:]:
        cells = row.xpath("./td")
        if not cells:
            continue
        tickers.append(cells[0].text_content().strip().replace(".", "-"))
        sectors.append(cells[2].text_content().strip())

    universe = pl.DataFrame({"ticker": tickers, "sector": sectors})

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    universe.write_csv(UNIVERSE_PATH)
    return universe


def sample_universe(n: int = 300, seed: int = 0) -> pl.DataFrame:
    """A reproducible random subset of the universe, sized for one problem."""
    return load_universe().sample(n=n, seed=seed)


def _prices_cache_path(start: str, end: str) -> Path:
    return DATA_DIR / f"sp500_prices_{start}_{end}.parquet"


def load_prices(start: str = "2023-01-01", end: str = "2025-01-01", refresh: bool = False) -> pl.DataFrame:
    """Adjusted close prices (one ``date`` column + one column per ticker)
    for the full universe, cached.

    Always fetches the full universe for the given window so one cache file
    is shared across notebooks and problem sizes; slice out the tickers you
    want for a given problem with ``prices.select(["date", *some_tickers])``.
    """
    cache_path = _prices_cache_path(start, end)
    if cache_path.exists() and not refresh:
        return pl.read_parquet(cache_path)

    tickers = load_universe()["ticker"].to_list()
    data = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True, threads=True)
    # how="any", not "all": a ticker that IPO'd/spun off partway through the
    # window (e.g. GEV, RDDT, KVUE) is NaN for the days before it existed,
    # not entirely NaN — and NaN poisons polars' .mean()/.pct_change() rather
    # than being skipped like a null would be. Drop anything without a full
    # history for this window rather than let that leak into the objective.
    close = data["Close"].dropna(axis=1, how="any")

    dates = close.index.to_numpy().astype("datetime64[D]")
    prices = pl.DataFrame({
        "date": dates,
        **{ticker: close[ticker].to_numpy().astype("float64") for ticker in close.columns},
    })

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    prices.write_parquet(cache_path)
    return prices


def load_returns(start: str = "2023-01-01", end: str = "2025-01-01", refresh: bool = False) -> pl.DataFrame:
    """Daily simple returns (one ``date`` column + one column per ticker),
    derived from cached prices."""
    prices = load_prices(start=start, end=end, refresh=refresh)
    tickers = [c for c in prices.columns if c != "date"]
    return prices.select(
        pl.col("date"),
        *[pl.col(t).pct_change().alias(t) for t in tickers],
    ).slice(1)  # first row is null for every ticker after pct_change
