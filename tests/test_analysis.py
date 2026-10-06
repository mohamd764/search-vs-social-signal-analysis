"""Checks for the weekly merge, influence score, rolling correlation, and OLS."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import importlib.util

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import analysis

_SAMPLE_PATH = ROOT / "data" / "sample" / "make_sample.py"
_SPEC = importlib.util.spec_from_file_location("make_sample", _SAMPLE_PATH)
make_sample = importlib.util.module_from_spec(_SPEC)
assert _SPEC is not None and _SPEC.loader is not None
_SPEC.loader.exec_module(make_sample)


class InfluenceAndMergeTests(unittest.TestCase):
    def test_influence_score_is_mean_of_weekly_sums(self):
        tweets = pd.DataFrame(
            {
                "tweet_date": pd.to_datetime(["2020-01-06 09:00:00", "2020-01-07 11:00:00"]),
                "retweet_count": [1, 3],
                "like_count": [10, 0],
                "quote_count": [0, 2],
            }
        )
        weekly = analysis.weekly_engagement(tweets)
        self.assertEqual(list(weekly.index), [pd.Timestamp("2020-01-12")])
        row = weekly.iloc[0]
        self.assertEqual(row["retweet_count"], 4)
        self.assertEqual(row["like_count"], 10)
        self.assertEqual(row["quote_count"], 2)
        self.assertAlmostEqual(row["influence_score"], (4 + 10 + 2) / 3)
        self.assertEqual(row["tweet_count"], 2)

    def test_loaders_accept_original_column_names_and_timezones(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            trends_path = directory / "trends.csv"
            tweets_path = directory / "tweets.csv"
            trends_path.write_text(
                "Unnamed: 0,Corruption,Car Wash Operations,Economy,isPartial\n"
                "2020-01-05,10,12,40,0\n"
                "2020-01-12,11,13,41,0\n",
                encoding="utf-8",
            )
            tweets_path.write_text(
                "tweet_created_at,retweet_count,like_count,quote_count\n"
                "2020-01-03T15:00:00+00:00,2,4,1\n"
                "2020-01-09T15:00:00+00:00,3,5,0\n",
                encoding="utf-8",
            )
            trends = analysis.load_trends(trends_path)
            tweets = analysis.load_tweets(tweets_path)
            merged = analysis.merge_weekly(trends, analysis.weekly_engagement(tweets))
        self.assertEqual(len(merged), 2)
        self.assertTrue(str(tweets["tweet_date"].dtype).startswith("datetime64"))
        self.assertIsNone(getattr(tweets["tweet_date"].dt, "tz", None))
        self.assertIn("influence_score", merged.columns)

    def test_merge_does_not_mutate_inputs(self):
        index = pd.date_range("2020-01-05", periods=4, freq="W-SUN")
        trends = pd.DataFrame({"Corruption": [1, 2, 3, 4], "Car Wash Operations": [2, 2, 2, 3]}, index=index)
        weekly = pd.DataFrame(
            {"retweet_count": 1, "like_count": 1, "quote_count": 1, "influence_score": 1.0, "tweet_count": 1},
            index=index,
        )
        trends_before = trends.copy()
        weekly_before = weekly.copy()
        merged = analysis.merge_weekly(trends, weekly)
        pd.testing.assert_frame_equal(trends, trends_before)
        pd.testing.assert_frame_equal(weekly, weekly_before)
        self.assertEqual(len(merged), 4)


class RollingAndOlsTests(unittest.TestCase):
    def _panel(self) -> pd.DataFrame:
        index = pd.date_range("2020-01-05", periods=40, freq="W-SUN")
        time = (index - index[0]).days.astype(float)
        x = np.array([(step % 5) - 2 for step in range(40)], dtype=float)
        y = 3 + 1.5 * x + 0.01 * time
        return pd.DataFrame(
            {
                "Corruption": x,
                "Car Wash Operations": x[::-1],
                "Economy": np.linspace(40, 50, 40),
                "tweet_count": y,
                "influence_score": y + 2,
            },
            index=index,
        )

    def test_rolling_correlation_window_and_panel(self):
        frame = self._panel()
        pairwise = analysis.rolling_correlation(
            frame,
            pairs=(("Corruption", "tweet_count"),),
            window=8,
        )
        self.assertEqual(list(pairwise.columns), ["Corruption vs tweet_count"])
        self.assertTrue(pairwise.iloc[:7, 0].isna().all())
        self.assertTrue(pairwise.iloc[7:].notna().all().all())
        panel = analysis.rolling_correlation(frame, pairs=(("Corruption", "tweet_count"),), window=8, as_panel=True)
        self.assertEqual(panel.index.nlevels, 2)

    def test_ols_recovers_known_coefficients(self):
        frame = self._panel()
        model = analysis.fit_ols_with_time(frame, "Corruption", "tweet_count")
        self.assertEqual(int(model.nobs), 40)
        self.assertAlmostEqual(model.params["const"], 3.0, places=6)
        self.assertAlmostEqual(model.params["Corruption"], 1.5, places=6)
        self.assertAlmostEqual(model.params["time"], 0.01, places=6)
        self.assertAlmostEqual(model.rsquared, 1.0, places=6)
        self.assertNotIn("time", frame.columns)

    def test_period_correlation_accepts_quarter_alias(self):
        frame = self._panel()
        corr = analysis.period_correlation_matrix(frame, rule="Q")
        self.assertIn("Corruption", corr.columns)
        self.assertNotIn("isPartial", corr.columns)


class SampleAndRepoTests(unittest.TestCase):
    def test_committed_sample_matches_generator(self):
        sample_dir = ROOT / "data" / "sample"
        with tempfile.TemporaryDirectory() as tmp:
            make_sample.write_sample(Path(tmp))
            for name in ("google_trends_weekly.csv", "twitter_alpha.csv", "twitter_beta.csv"):
                committed = (sample_dir / name).read_bytes()
                regenerated = (Path(tmp) / name).read_bytes()
                self.assertEqual(committed, regenerated)
                self.assertTrue(committed.decode("utf-8").startswith(make_sample.HEADER.splitlines()[0]))

    def test_sample_pipeline_writes_figures(self):
        trends = analysis.load_trends(analysis.DEFAULT_TRENDS_PATH)
        tweets = analysis.load_tweets(analysis.DEFAULT_TWEET_PATHS[0])
        result = analysis.analyze_series(trends, tweets, window=12)
        self.assertGreaterEqual(len(result["merged"]), 12)
        self.assertEqual(set(result["models"]), set(analysis.HEADLINE_SPECS))
        with tempfile.TemporaryDirectory() as tmp:
            written = analysis.save_series_figures("sample alpha", result, tmp)
        self.assertGreaterEqual(len(written), 3)
        for path in written:
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 0)

    def test_sources_have_no_local_user_paths(self):
        banned = ("C:/Users", "C:\\Users")
        suffixes = {".py", ".md", ".ipynb", ".yml", ".yaml", ".txt", ".csv"}
        offenders = []
        for path in ROOT.rglob("*"):
            if not path.is_file() or path.suffix not in suffixes:
                continue
            if path.resolve() == Path(__file__).resolve():
                continue
            if any(part in {".git", "outputs", ".venv"} for part in path.parts):
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for token in banned:
                if token in text:
                    offenders.append(f"{path}: {token}")
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
