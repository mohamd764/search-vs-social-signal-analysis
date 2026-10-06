"""Weekly Google Trends indexes versus Twitter engagement.

The pipeline:

1. Load a weekly search-interest table and one or more tweet-level extracts.
2. Resample tweets to Sunday-ending weeks and build an influence score.
3. Inner-join the two weekly tables.
4. Estimate rolling Pearson correlations and an OLS fit with a linear time control.

The influence score is the unweighted mean of the weekly sums of retweets, likes,
and quotes. ``time`` in the regression is the number of days since the first week
in that merged panel. That is the specification behind the published fits
(R² = 0.053, n = 366; R² = 0.272, n = 362).
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson

ROOT = Path(__file__).resolve().parent
DEFAULT_TRENDS_PATH = ROOT / "data" / "sample" / "google_trends_weekly.csv"
DEFAULT_TWEET_PATHS = (
    ROOT / "data" / "sample" / "twitter_alpha.csv",
    ROOT / "data" / "sample" / "twitter_beta.csv",
)
DEFAULT_OUTPUT_DIR = ROOT / "outputs"

ENGAGEMENT_COLUMNS = ("retweet_count", "like_count", "quote_count")
TREND_DATE_CANDIDATES = ("date", "GT_data_from", "Unnamed: 0")
TWEET_DATE_CANDIDATES = ("tweet_created_at", "tweet_date")
EXCLUDED_FROM_CORR = ("isPartial", "time")
HEADLINE_SPECS = (
    ("Corruption", "tweet_count"),
    ("Car Wash Operations", "influence_score"),
)
DEFAULT_PAIRS = HEADLINE_SPECS + (
    ("Economy", "influence_score"),
    ("Corruption", "influence_score"),
    ("Car Wash Operations", "tweet_count"),
)
PERIOD_RULES = {"Q": "QE", "QE": "QE", "Y": "YE", "A": "YE", "YE": "YE"}


def _read_table(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No data file at {path}")
    frame = pd.read_csv(path, comment="#")
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def _resolve_column(frame: pd.DataFrame, explicit: str | None, candidates: Sequence[str], kind: str) -> str:
    if explicit:
        if explicit not in frame.columns:
            raise KeyError(f"{kind} date column {explicit!r} is not in {list(frame.columns)}")
        return explicit
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise KeyError(
        f"No {kind} date column found. Looked for {list(candidates)}; "
        f"file columns are {list(frame.columns)}."
    )


def _to_naive_utc(values: pd.Series) -> pd.Series:
    """Parse timestamps and drop timezone information (values are left in UTC)."""
    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    return pd.Series(parsed.dt.tz_convert(None), index=values.index)


def load_trends(path: Path | str, date_column: str | None = None) -> pd.DataFrame:
    """Load a weekly Google Trends table indexed by normalized week labels."""
    frame = _read_table(path)
    column = _resolve_column(frame, date_column, TREND_DATE_CANDIDATES, "Trends")
    frame = frame.rename(columns={column: "week"})
    frame["week"] = _to_naive_utc(frame["week"]).dt.normalize()
    frame = frame.dropna(subset=["week"]).set_index("week").sort_index()
    required = [name for name in ("Corruption", "Car Wash Operations") if name not in frame.columns]
    if required:
        raise KeyError(f"Trends file is missing columns: {required}")
    return frame


def load_tweets(path: Path | str, date_column: str | None = None) -> pd.DataFrame:
    """Load a tweet-level engagement extract with a naive ``tweet_date`` column."""
    frame = _read_table(path)
    column = _resolve_column(frame, date_column, TWEET_DATE_CANDIDATES, "tweet")
    missing = [name for name in ENGAGEMENT_COLUMNS if name not in frame.columns]
    if missing:
        raise KeyError(f"Twitter file is missing columns: {missing}")
    frame = frame.copy()
    frame["tweet_date"] = _to_naive_utc(frame[column])
    frame = frame.dropna(subset=["tweet_date"])
    for name in ENGAGEMENT_COLUMNS:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    return frame


def weekly_engagement(tweets: pd.DataFrame, date_column: str = "tweet_date") -> pd.DataFrame:
    """Sum engagement by Sunday-ending week and add the influence score.

    ``influence_score`` is the arithmetic mean of the weekly sums of
    ``retweet_count``, ``like_count``, and ``quote_count``. ``tweet_count`` is
    the number of tweets in the week.
    """
    if date_column not in tweets.columns:
        raise KeyError(f"Expected a {date_column!r} column")
    missing = [name for name in ENGAGEMENT_COLUMNS if name not in tweets.columns]
    if missing:
        raise KeyError(f"Twitter frame is missing columns: {missing}")
    frame = tweets.dropna(subset=[date_column]).copy()
    if frame.empty:
        raise ValueError("No tweets with a usable timestamp")
    frame = frame.set_index(date_column).sort_index()
    weekly = frame.resample("W-SUN").agg({name: "sum" for name in ENGAGEMENT_COLUMNS})
    weekly["influence_score"] = weekly[list(ENGAGEMENT_COLUMNS)].mean(axis=1)
    weekly["tweet_count"] = frame.resample("W-SUN").size()
    weekly.index.name = "week"
    return weekly


def merge_weekly(trends: pd.DataFrame, weekly_engagement_frame: pd.DataFrame) -> pd.DataFrame:
    """Inner-join weekly Trends and weekly engagement on the week timestamp."""
    left = trends.copy()
    right = weekly_engagement_frame.copy()
    left.index = pd.DatetimeIndex(pd.to_datetime(left.index)).tz_localize(None).normalize()
    right.index = pd.DatetimeIndex(pd.to_datetime(right.index)).tz_localize(None).normalize()
    left.index.name = "week"
    right.index.name = "week"
    merged = left.join(right, how="inner").sort_index()
    merged.index.name = "week"
    return merged


def correlation_matrix(
    frame: pd.DataFrame,
    exclude: Sequence[str] = EXCLUDED_FROM_CORR,
) -> pd.DataFrame:
    """Pearson correlation of numeric columns, dropping flags such as ``isPartial``."""
    numeric = frame.select_dtypes(include="number")
    columns = [column for column in numeric.columns if column not in exclude]
    return numeric.loc[:, columns].corr()


def _existing_pairs(frame: pd.DataFrame, pairs: Sequence[tuple[str, str]]) -> list[tuple[str, str]]:
    return [(left, right) for left, right in pairs if left in frame.columns and right in frame.columns]


def rolling_correlation(
    frame: pd.DataFrame,
    pairs: Sequence[tuple[str, str]] | None = None,
    window: int = 12,
    min_periods: int | None = None,
    as_panel: bool = False,
) -> pd.DataFrame:
    """Trailing-window Pearson correlations.

    Each window has its own correlation matrix (``DataFrame.rolling.corr``).
    By default the return value keeps one column per requested pair, named
    ``"{left} vs {right}"``. Set ``as_panel=True`` to return the full rolling
    correlation matrices instead (MultiIndex of week, variable).
    """
    if window < 3:
        raise ValueError("window must be at least 3 observations")
    if min_periods is None:
        min_periods = window
    selected = DEFAULT_PAIRS if pairs is None else pairs
    use_pairs = _existing_pairs(frame, selected)
    if not use_pairs:
        raise KeyError("None of the requested correlation pairs are present in the frame")
    columns: list[str] = []
    for left, right in use_pairs:
        for name in (left, right):
            if name not in columns:
                columns.append(name)
    use = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce").sort_index()
    if len(use) < min_periods:
        raise ValueError(
            f"Need at least {min_periods} weekly rows for a rolling window of {window}; got {len(use)}"
        )
    panel = use.rolling(window=window, min_periods=min_periods).corr()
    if as_panel:
        return panel
    paired: dict[str, pd.Series] = {}
    for left, right in use_pairs:
        extracted = panel.xs(left, level=-1)[right].reindex(use.index)
        paired[f"{left} vs {right}"] = extracted
    return pd.DataFrame(paired, index=use.index)


def period_correlation_matrix(
    frame: pd.DataFrame,
    rule: str = "QE",
    exclude: Sequence[str] = EXCLUDED_FROM_CORR,
) -> pd.DataFrame:
    """Correlation of period means.

    Quarterly (``QE``) and yearly (``YE``) results match the original notebook,
    which averaged the weekly panel first and then correlated those means. The
    result is one matrix for the whole sample, not a matrix inside each period.
    ``Q`` and ``Y`` are accepted as aliases.
    """
    frequency = PERIOD_RULES.get(rule, rule)
    numeric = frame.select_dtypes(include="number")
    columns = [column for column in numeric.columns if column not in exclude]
    aggregated = numeric.loc[:, columns].resample(frequency).mean()
    return aggregated.dropna(axis=1, how="all").corr()


def fit_ols_with_time(frame: pd.DataFrame, independent: str, dependent: str):
    """OLS of ``dependent`` on ``independent`` plus days since the first week.

    Rows with a missing regressor or outcome are dropped. The caller's frame
    is not modified. The covariance type is classical OLS, matching the
    published notebook summaries.
    """
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("The frame must be indexed by week-ending timestamps")
    for name in (independent, dependent):
        if name not in frame.columns:
            raise KeyError(f"Column {name!r} is not in the frame")
    subset = frame.loc[:, [independent, dependent]].apply(pd.to_numeric, errors="coerce")
    subset = subset.dropna().sort_index()
    if len(subset) < 4:
        raise ValueError(f"Need at least 4 complete weeks to fit OLS; got {len(subset)}")
    time = (subset.index - subset.index[0]).days.astype(float)
    design = pd.DataFrame(
        {independent: subset[independent].astype(float).to_numpy(), "time": time},
        index=subset.index,
    )
    design = sm.add_constant(design, has_constant="add")
    outcome = subset[dependent].astype(float)
    return sm.OLS(outcome, design).fit()


def _style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_correlation_heatmap(frame: pd.DataFrame, title: str, path: Path | str | None = None):
    """Heatmap of :func:`correlation_matrix`."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    corr = correlation_matrix(frame)
    fig, ax = plt.subplots(figsize=(8.2, 6.6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_scatter_with_trendline(
    frame: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str,
    path: Path | str | None = None,
):
    """Scatter with a bivariate trend line (no time control)."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    sns.regplot(
        x=x_column,
        y=y_column,
        data=frame,
        ax=ax,
        scatter_kws={"s": 18, "alpha": 0.75, "color": "#1f4e79"},
        line_kws={"color": "#b22222", "linewidth": 1.5},
    )
    ax.set_title(title)
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    _style_axes(ax)
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_rolling_correlation(pairwise: pd.DataFrame, title: str, path: Path | str | None = None):
    """Line chart of rolling pairwise correlations."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    pairwise.plot(ax=ax, linewidth=1.6)
    ax.axhline(0, color="0.45", linewidth=0.8)
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("Pearson r")
    ax.set_xlabel("Week")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    _style_axes(ax)
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_period_correlation_heatmap(corr: pd.DataFrame, title: str, path: Path | str | None = None):
    """Heatmap for a period-mean correlation matrix."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(8.2, 6.6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    _save(fig, path)
    return fig


def _save(fig, path: Path | str | None) -> None:
    if path is None:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=150, bbox_inches="tight")


def analyze_series(
    trends: pd.DataFrame,
    tweets: pd.DataFrame,
    window: int = 12,
) -> dict:
    """Merge one Twitter series with Trends and fit the headline models."""
    weekly = weekly_engagement(tweets)
    merged = merge_weekly(trends, weekly)
    if merged.empty:
        raise ValueError(
            "The weekly inner join is empty. Trends week labels need to fall on the "
            "same Sunday-ending dates produced by resampling tweets with W-SUN."
        )
    pairs = _existing_pairs(merged, DEFAULT_PAIRS)
    rolling = rolling_correlation(merged, pairs=pairs, window=window)
    models = {}
    for spec in HEADLINE_SPECS:
        independent, dependent = spec
        if independent in merged.columns and dependent in merged.columns:
            models[spec] = fit_ols_with_time(merged, independent, dependent)
    quarterly = period_correlation_matrix(merged, rule="QE")
    return {
        "weekly": weekly,
        "merged": merged,
        "rolling": rolling,
        "models": models,
        "quarterly_corr": quarterly,
        "window": window,
    }


def save_series_figures(name: str, result: dict, directory: Path | str) -> list[Path]:
    """Write the correlation, scatter, rolling, and quarterly-mean charts."""
    import matplotlib.pyplot as plt

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    slug = _slug(name)
    written: list[Path] = []

    def _store(fig, filename: str) -> None:
        destination = directory / filename
        _save(fig, destination)
        written.append(destination)
        plt.close(fig)

    _store(
        plot_correlation_heatmap(result["merged"], f"{name}: weekly correlation"),
        f"{slug}_correlation_heatmap.png",
    )
    _store(
        plot_rolling_correlation(
            result["rolling"],
            f"{name}: {result['window']}-week rolling correlation",
        ),
        f"{slug}_rolling_correlation.png",
    )
    _store(
        plot_period_correlation_heatmap(
            result["quarterly_corr"],
            f"{name}: correlation of quarterly means",
        ),
        f"{slug}_quarterly_mean_correlation.png",
    )
    merged = result["merged"]
    for independent, dependent, filename in (
        ("Corruption", "tweet_count", f"{slug}_corruption_vs_tweet_count.png"),
        ("Car Wash Operations", "influence_score", f"{slug}_carwash_vs_influence.png"),
    ):
        if independent in merged.columns and dependent in merged.columns:
            _store(
                plot_scatter_with_trendline(
                    merged,
                    independent,
                    dependent,
                    f"{name}: {dependent} vs {independent}",
                ),
                filename,
            )
    return written


def _slug(name: str) -> str:
    cleaned = "".join(character.lower() if character.isalnum() else "_" for character in name.strip())
    return "_".join(part for part in cleaned.split("_") if part) or "series"


def model_record(series_name: str, independent: str, dependent: str, model, source: str) -> dict:
    """JSON-friendly summary of one OLS fit."""
    return {
        "series": series_name,
        "independent": independent,
        "dependent": dependent,
        "nobs": int(model.nobs),
        "rsquared": float(model.rsquared),
        "rsquared_adj": float(model.rsquared_adj),
        "fvalue": float(model.fvalue),
        "f_pvalue": float(model.f_pvalue),
        "condition_number": float(model.condition_number),
        "durbin_watson": float(durbin_watson(model.resid)),
        "params": {key: float(value) for key, value in model.params.items()},
        "pvalues": {key: float(value) for key, value in model.pvalues.items()},
        "data_source": source,
    }


def _looks_synthetic(path: Path) -> bool:
    with path.open(encoding="utf-8") as handle:
        head = handle.readline()
    return "SYNTHETIC" in head


def _default_label(path: Path) -> str:
    stem = path.stem
    if stem.startswith("twitter_"):
        stem = stem[len("twitter_") :]
    return stem.replace("_", " ")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge weekly Google Trends with Twitter engagement, then correlate and fit OLS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--trends", type=Path, default=DEFAULT_TRENDS_PATH, help="Weekly Google Trends CSV")
    parser.add_argument(
        "--tweets",
        type=Path,
        nargs="+",
        default=list(DEFAULT_TWEET_PATHS),
        help="Tweet-level CSV extracts, one series per file",
    )
    parser.add_argument("--names", nargs="+", default=None, help="Display name for each tweet file")
    parser.add_argument("--trends-date-column", default=None, help="Date column in the Trends file")
    parser.add_argument("--tweet-date-column", default=None, help="Timestamp column in the tweet files")
    parser.add_argument("--window", type=int, default=12, help="Trailing window (weeks) for rolling correlations")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for figures and OLS JSON")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    os.environ.setdefault("MPLBACKEND", "Agg")
    args = _parse_args(argv)
    names = list(args.names) if args.names is not None else [_default_label(path) for path in args.tweets]
    if len(names) != len(args.tweets):
        raise SystemExit(f"--names has {len(names)} entries for {len(args.tweets)} tweet files")

    synthetic = _looks_synthetic(args.trends)
    if synthetic:
        print(
            "Running on the synthetic sample in data/sample/. "
            "R² values from this run describe that generated data. "
            "The published fits are R² = 0.272 (n = 362) and R² = 0.053 (n = 366); see the README."
        )

    trends = load_trends(args.trends, date_column=args.trends_date_column)
    source = "synthetic" if synthetic else "user"
    records = []
    figure_dir = args.output_dir / "figures"
    for path, name in zip(args.tweets, names):
        tweets = load_tweets(path, date_column=args.tweet_date_column)
        result = analyze_series(trends, tweets, window=args.window)
        written = save_series_figures(name, result, figure_dir)
        print(f"\n{name}: {len(result['merged'])} overlapping weeks. Figures: {len(written)}")
        for (independent, dependent), model in result["models"].items():
            record = model_record(name, independent, dependent, model, source)
            records.append(record)
            print(
                f"  {dependent} ~ {independent} + time | "
                f"n = {record['nobs']}  R² = {record['rsquared']:.4f}  "
                f"adj. R² = {record['rsquared_adj']:.4f}"
            )
            print(model.summary())

    if not records:
        raise SystemExit("No OLS models were fit. Check that the expected columns are present.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "ols_results.json"
    results_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(
        [
            {
                "series": record["series"],
                "independent": record["independent"],
                "dependent": record["dependent"],
                "nobs": record["nobs"],
                "rsquared": record["rsquared"],
                "rsquared_adj": record["rsquared_adj"],
                "fvalue": record["fvalue"],
                "f_pvalue": record["f_pvalue"],
                "condition_number": record["condition_number"],
                "durbin_watson": record["durbin_watson"],
                "data_source": record["data_source"],
            }
            for record in records
        ]
    ).to_csv(args.output_dir / "ols_results.csv", index=False)
    print(f"\nWrote {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
