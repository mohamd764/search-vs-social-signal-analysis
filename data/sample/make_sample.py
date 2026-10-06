"""Write a tiny synthetic Trends table and two tweet extracts.

The files follow the column layout of the private inputs used in the original
notebook. Values are generated from NumPy seed 42. They are not Google Trends
indexes and they are not Twitter/X activity.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HEADER = (
    "# SYNTHETIC DATA — generated counts and indexes, not Google Trends or Twitter/X measurements.\n"
    "# Rebuild with: python data/sample/make_sample.py (NumPy seed 42).\n"
)
WEEKS = 60
SEED = 42


def build(seed: int = SEED, n_weeks: int = WEEKS) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    weeks = pd.date_range("2020-01-05", periods=n_weeks, freq="W-SUN")
    t = np.arange(n_weeks, dtype=float)
    trends = pd.DataFrame(
        {
            "date": weeks.strftime("%Y-%m-%d"),
            "Corruption": _index(25 + 8 * np.sin(t / 6) + rng.normal(0, 4, n_weeks)),
            "Car Wash Operations": _index(18 + 0.15 * t + rng.normal(0, 3, n_weeks)),
            "Economy": _index(55 + 4 * np.cos(t / 9) + rng.normal(0, 3.5, n_weeks)),
            "isPartial": np.array([0] * (n_weeks - 1) + [1], dtype=int),
        }
    )
    return {
        "google_trends_weekly.csv": trends,
        "twitter_alpha.csv": _tweets(rng, weeks, like_high=40),
        "twitter_beta.csv": _tweets(rng, weeks, like_high=80),
    }


def write_sample(directory: Path | None = None, seed: int = SEED, n_weeks: int = WEEKS) -> Path:
    directory = Path(__file__).resolve().parent if directory is None else Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, frame in build(seed=seed, n_weeks=n_weeks).items():
        float_format = "%.1f" if name.startswith("google_trends") else None
        body = frame.to_csv(index=False, lineterminator="\n", float_format=float_format)
        (directory / name).write_text(HEADER + body, encoding="utf-8")
    return directory


def _index(values: np.ndarray) -> np.ndarray:
    return np.clip(np.round(values, 1), 0, 100)


def _tweets(rng: np.random.Generator, weeks: pd.DatetimeIndex, like_high: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for week in weeks:
        n_tweets = int(rng.integers(1, 5))
        for step in range(1, n_tweets + 1):
            stamp = week - pd.Timedelta(days=step) + pd.Timedelta(hours=int(rng.integers(8, 20)))
            rows.append(
                {
                    "tweet_created_at": stamp.strftime("%Y-%m-%dT%H:%M:%S") + "+00:00",
                    "retweet_count": int(rng.integers(0, 25)),
                    "like_count": int(rng.integers(0, like_high)),
                    "quote_count": int(rng.integers(0, 6)),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    directory = write_sample()
    print(f"Wrote synthetic sample files to {directory}")


if __name__ == "__main__":
    main()
