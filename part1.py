from __future__ import annotations

from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import norm



CAN_BANKS = ["RY.TO", "TD.TO", "BNS.TO", "BMO.TO", "CM.TO"]
US_BANKS = ["JPM", "BAC", "GS", "MS"]
BENCHMARKS = ["^GSPTSE", "^GSPC"]

START_DATE = "2014-01-01"
TRADING_DAYS = 252
RF_ANNUAL = 0.0  

ROOT = Path("bank-intelligence")
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
TAB_DIR = OUT_DIR / "tables"


#Set up
def ensure_dirs() -> None:
    for p in [RAW_DIR, PROC_DIR, FIG_DIR, TAB_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def save_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path)
    elif path.suffix.lower() == ".csv":
        df.to_csv(path, index=True)
    else:
        raise ValueError(f"Unsupported file extension: {path.suffix}")


# download prices
def download_prices(tickers: list[str], start: str = START_DATE) -> pd.DataFrame:
    df = yf.download(
        tickers=tickers,
        start=start,
        auto_adjust=False,
        actions=False,
        progress=False,
        group_by="column",
    )

    if isinstance(df.columns, pd.MultiIndex):
        if "Adj Close" in df.columns.get_level_values(0):
            prices = df["Adj Close"].copy()
        else:
            prices = df["Close"].copy()
    else:
        # single ticker case
        col = "Adj Close" if "Adj Close" in df.columns else "Close"
        prices = df[[col]].rename(columns={col: tickers[0]}).copy()

    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "date"
    prices = prices.sort_index()
    return prices


def data_health(prices: pd.DataFrame) -> pd.DataFrame:
    health = pd.DataFrame(
        {
            "start_date": prices.apply(lambda s: s.first_valid_index()),
            "end_date": prices.apply(lambda s: s.last_valid_index()),
            "n_obs": prices.notna().sum(),
            "pct_missing": prices.isna().mean() * 100,
        }
    ).sort_values("pct_missing", ascending=False)
    return health


# Set features
def compute_daily_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1))


def compute_monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    monthly_prices = prices.resample("M").last()
    return monthly_prices.pct_change()


def compute_rolling_volatility(log_returns: pd.DataFrame) -> pd.DataFrame:
    vol_20 = log_returns.rolling(20).std() * np.sqrt(TRADING_DAYS)
    vol_60 = log_returns.rolling(60).std() * np.sqrt(TRADING_DAYS)
    vol_20.columns = [f"{c}_vol20" for c in vol_20.columns]
    vol_60.columns = [f"{c}_vol60" for c in vol_60.columns]
    return pd.concat([vol_20, vol_60], axis=1)


# performances and Rankings
def cagr(prices: pd.Series) -> float:
    s = prices.dropna()
    if len(s) < 2:
        return np.nan
    start, end = float(s.iloc[0]), float(s.iloc[-1])
    n_days = (s.index[-1] - s.index[0]).days
    n_years = n_days / 365.25
    if n_years <= 0:
        return np.nan
    return (end / start) ** (1 / n_years) - 1


def annualized_vol(log_returns: pd.Series) -> float:
    s = log_returns.dropna()
    if len(s) < 2:
        return np.nan
    return float(s.std()) * np.sqrt(TRADING_DAYS)


def sharpe_ratio(log_returns: pd.Series, rf_annual: float = RF_ANNUAL) -> float:
    s = log_returns.dropna()
    if len(s) < 2:
        return np.nan
    rf_daily = (1 + rf_annual) ** (1 / TRADING_DAYS) - 1
    excess = s - rf_daily
    denom = float(excess.std())
    if denom == 0 or np.isnan(denom):
        return np.nan
    return float(excess.mean() / denom) * np.sqrt(TRADING_DAYS)


def sortino_ratio(log_returns: pd.Series, rf_annual: float = RF_ANNUAL) -> float:
    s = log_returns.dropna()
    if len(s) < 2:
        return np.nan
    rf_daily = (1 + rf_annual) ** (1 / TRADING_DAYS) - 1
    excess = s - rf_daily
    downside = excess[excess < 0]
    denom = float(downside.std())
    if denom == 0 or np.isnan(denom):
        return np.nan
    return float(excess.mean() / denom) * np.sqrt(TRADING_DAYS)


def max_drawdown(prices: pd.Series) -> float:
    s = prices.dropna()
    if len(s) < 2:
        return np.nan
    running_max = s.cummax()
    dd = (s / running_max) - 1.0
    return float(dd.min())


def calmar_ratio(prices: pd.Series) -> float:
    cg = cagr(prices)
    mdd = max_drawdown(prices)
    if np.isnan(cg) or np.isnan(mdd) or mdd == 0:
        return np.nan
    return float(cg / abs(mdd))


def worst_day(log_returns: pd.Series) -> float:
    s = log_returns.dropna()
    if len(s) == 0:
        return np.nan
    return float(s.min())


def zscore(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    mu, sd = s.mean(), s.std()
    if sd == 0 or np.isnan(sd):
        return pd.Series(np.nan, index=s.index)
    return (s - mu) / sd


def performance_table(prices: pd.DataFrame, rets: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    rows = []
    for t in tickers:
        if t not in prices.columns or t not in rets.columns:
            continue
        rows.append(
            {
                "ticker": t,
                "start_date": prices[t].first_valid_index(),
                "end_date": prices[t].last_valid_index(),
                "n_obs": int(prices[t].notna().sum()),
                "cagr": cagr(prices[t]),
                "ann_vol": annualized_vol(rets[t]),
                "sharpe": sharpe_ratio(rets[t]),
                "sortino": sortino_ratio(rets[t]),
                "max_drawdown": max_drawdown(prices[t]),
                "calmar": calmar_ratio(prices[t]),
                "worst_daily_log_return": worst_day(rets[t]),
            }
        )
    return pd.DataFrame(rows).set_index("ticker")


def add_risk_adjusted_ranking(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    out["mdd_mag"] = out["max_drawdown"].abs()

    z_sharpe = zscore(out["sharpe"])
    z_calmar = zscore(out["calmar"])
    z_sortino = zscore(out["sortino"])
    z_vol = zscore(out["ann_vol"])
    z_mdd = zscore(out["mdd_mag"])

    w_sharpe, w_calmar, w_sortino, w_vol, w_mdd = 0.35, 0.25, 0.15, 0.15, 0.10
    out["risk_adjusted_score"] = (
        w_sharpe * z_sharpe
        + w_calmar * z_calmar
        + w_sortino * z_sortino
        - w_vol * z_vol
        - w_mdd * z_mdd
    )
    out["rank_overall"] = out["risk_adjusted_score"].rank(ascending=False, method="min").astype("Int64")
    return out.sort_values(["rank_overall", "ticker"])


def plot_cumulative_growth(rets: pd.DataFrame, tickers: list[str], outpath: Path) -> None:
    data = rets[tickers].dropna(how="all")
    growth = np.exp(data.cumsum())
    plt.figure()
    growth.plot()
    plt.title("Cumulative Growth (Log-Return Compounded)")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_drawdowns(prices: pd.DataFrame, tickers: list[str], outpath: Path) -> None:
    s = prices[tickers].dropna(how="all")
    running_max = s.cummax()
    dd = (s / running_max) - 1.0
    plt.figure()
    dd.plot()
    plt.title("Drawdowns")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# Tail risk
def hist_var(returns: pd.Series, alpha: float) -> float:
    s = returns.dropna()
    if len(s) < 50:
        return np.nan
    return float(np.quantile(s, alpha))


def hist_es(returns: pd.Series, alpha: float) -> float:
    s = returns.dropna()
    if len(s) < 50:
        return np.nan
    v = np.quantile(s, alpha)
    tail = s[s <= v]
    if len(tail) == 0:
        return np.nan
    return float(tail.mean())


def normal_var(returns: pd.Series, alpha: float) -> float:
    s = returns.dropna()
    if len(s) < 50:
        return np.nan
    mu = float(s.mean())
    sigma = float(s.std())
    if sigma == 0 or np.isnan(sigma):
        return np.nan
    z = norm.ppf(alpha)
    return mu + z * sigma


def normal_es(returns: pd.Series, alpha: float) -> float:
    s = returns.dropna()
    if len(s) < 50:
        return np.nan
    mu = float(s.mean())
    sigma = float(s.std())
    if sigma == 0 or np.isnan(sigma):
        return np.nan
    z = norm.ppf(alpha)
    phi = norm.pdf(z)
    return mu - sigma * (phi / alpha)


def var_backtest_breaches(returns: pd.Series, var_value: float) -> tuple[int, float]:
    s = returns.dropna()
    if len(s) == 0 or np.isnan(var_value):
        return 0, np.nan
    breaches = int((s < var_value).sum())
    rate = breaches / len(s)
    return breaches, float(rate)


def build_tail_risk_table(rets: pd.DataFrame, tickers: list[str], alphas: list[float]) -> pd.DataFrame:
    rows = []
    for t in tickers:
        if t not in rets.columns:
            continue
        s = rets[t].dropna()
        row = {"ticker": t, "n_obs": int(s.notna().sum()), "start_date": s.index.min(), "end_date": s.index.max()}

        for a in alphas:
            hv = hist_var(s, a)
            he = hist_es(s, a)
            nv = normal_var(s, a)
            ne = normal_es(s, a)

            pct = int((1 - a) * 100)
            row[f"hist_VaR_{pct}"] = hv
            row[f"hist_ES_{pct}"] = he
            row[f"norm_VaR_{pct}"] = nv
            row[f"norm_ES_{pct}"] = ne

            b, r = var_backtest_breaches(s, hv)
            row[f"breaches_histVaR_{pct}"] = b
            row[f"breach_rate_histVaR_{pct}"] = r

        rows.append(row)

    return pd.DataFrame(rows).set_index("ticker")


def plot_return_hist_with_var(rets: pd.Series, title: str, var95: float, var99: float, outpath: Path) -> None:
    s = rets.dropna()
    plt.figure()
    plt.hist(s.values, bins=60)
    plt.axvline(var95, linestyle="--")
    plt.axvline(var99, linestyle="--")
    plt.title(title)
    plt.xlabel("Daily log return")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

# Set up main pipeline
def main() -> None:
    ensure_dirs()

    print("Current working directory:", Path.cwd())
    print("Project root will be created at:", (Path.cwd() / ROOT).resolve())

    tickers = CAN_BANKS + US_BANKS + BENCHMARKS

    
    prices = download_prices(tickers, start=START_DATE)
    health = data_health(prices)

    save_df(prices, RAW_DIR / "prices.parquet")
    save_df(health, RAW_DIR / "prices_health.csv")

    daily_log_returns = compute_daily_log_returns(prices)
    monthly_returns = compute_monthly_returns(prices)
    rolling_vol = compute_rolling_volatility(daily_log_returns)

    save_df(daily_log_returns, PROC_DIR / "daily_log_returns.parquet")
    save_df(monthly_returns, PROC_DIR / "monthly_returns.parquet")
    save_df(rolling_vol, PROC_DIR / "rolling_volatility.parquet")

 
    summary = performance_table(prices, daily_log_returns, tickers)
    ranked = add_risk_adjusted_ranking(summary)

    save_df(summary, TAB_DIR / "performance_summary.csv")
    save_df(ranked, TAB_DIR / "risk_adjusted_rankings.csv")

    plot_cumulative_growth(daily_log_returns, [t for t in tickers if t in daily_log_returns.columns], FIG_DIR / "cumulative_growth.png")
    plot_drawdowns(prices, [t for t in tickers if t in prices.columns], FIG_DIR / "drawdowns.png")

    print("\nTop 10 (overall risk-adjusted score):")
    cols = ["rank_overall", "risk_adjusted_score", "cagr", "ann_vol", "sharpe", "sortino", "calmar", "max_drawdown"]
    print(ranked[cols].head(10))

   
    alphas = [0.05, 0.01]
    tail = build_tail_risk_table(daily_log_returns, tickers, alphas)
    save_df(tail, TAB_DIR / "var_es_tail_risk.csv")

    for t in ["RY.TO", "TD.TO", "JPM", "^GSPTSE", "^GSPC"]:
        if t in daily_log_returns.columns and t in tail.index:
            var95 = tail.loc[t, "hist_VaR_95"]
            var99 = tail.loc[t, "hist_VaR_99"]
            plot_return_hist_with_var(
                daily_log_returns[t],
                title=f"Return Distribution + Historical VaR (95/99): {t}",
                var95=var95,
                var99=var99,
                outpath=FIG_DIR / f"returns_hist_var_{t.replace('^','')}.png",
            )

    print("\nWorst tail risk (Historical ES 99%):")
    cols2 = ["hist_VaR_99", "hist_ES_99", "norm_VaR_99", "norm_ES_99", "breach_rate_histVaR_99"]
    print(tail[cols2].sort_values("hist_ES_99").head(10))

    print("\nDONE.")
    print("Raw data:", (RAW_DIR / "prices.parquet").resolve())
    print("Processed:", (PROC_DIR / "daily_log_returns.parquet").resolve())
    print("Tables:", TAB_DIR.resolve())
    print("Figures:", FIG_DIR.resolve())


if __name__ == "__main__":
    main()