from __future__ import annotations

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import statsmodels.api as sm

from pandas_datareader import data as pdr
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas



ROOT = Path(r"C:\Users\offic\OneDrive\Desktop\LEARNING\powerbi project\mine\bank-intelligence")

RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
TAB_DIR = OUT_DIR / "tables"
REP_DIR = ROOT / "reports"


# configure
CAN_BANKS = ["RY.TO", "TD.TO", "BNS.TO", "BMO.TO", "CM.TO"]
US_BANKS = ["JPM", "BAC", "GS", "MS"]
BENCHMARKS = ["^GSPTSE", "^GSPC"]

TRADING_DAYS = 252

# Portfolio constraints
NO_SHORTING = True
MAX_WEIGHT_PER_ASSET = 0.30
MIN_CANADA_WEIGHT = 0.60

# Monte Carlo
MC_HORIZON_DAYS = 252
MC_N_PATHS = 5000
MC_METHOD = "mvnormal"  # "mvnormal" or "bootstrap"

# Macro start
MACRO_START = "2014-01-01"

# FRED series IDs (some Canada IDs may fail; replace if needed)
FRED_SERIES = {
    "fed_funds": "FEDFUNDS",
    "cpi_us": "CPIAUCSL",
    "unemp_us": "UNRATE",
    "ust10y": "DGS10",

    # Canada series (may vary on FRED; if they fail, replace)
    "cpi_ca": "CPALTT01CAM659N",
    "unemp_ca": "LRUNTTTTCAM156S",
    "boc_policy": "IRSTCI01CAM156N",
    "can10y": "IRLTLT01CAM156N",
}

STRESS_SCENARIOS = {
    "RatesUp_UnempUp": {"fed_funds": 0.50, "boc_policy": 0.50, "unemp_us": 0.50, "unemp_ca": 0.50},
    "RecessionShock": {"unemp_us": 1.50, "unemp_ca": 1.50, "ust10y": -0.50, "can10y": -0.50},
    "InflationSpike": {"cpi_us": 0.30, "cpi_ca": 0.30, "fed_funds": 0.25, "boc_policy": 0.25},
}


# Setup and loaders
def ensure_dirs() -> None:
    for p in [FIG_DIR, TAB_DIR, REP_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def load_daily_log_returns() -> pd.DataFrame:
    path = PROC_DIR / "daily_log_returns.parquet"
    require_file(path)
    df = pd.read_parquet(path).sort_index()
    df.index = pd.to_datetime(df.index)
    return df


def load_monthly_returns() -> pd.DataFrame:
    path = PROC_DIR / "monthly_returns.parquet"
    require_file(path)
    df = pd.read_parquet(path).sort_index()
    df.index = pd.to_datetime(df.index)
    return df


def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, index_col=0)


def safe_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True)


def zscore(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    mu, sd = s.mean(), s.std()
    if sd == 0 or np.isnan(sd):
        return pd.Series(np.nan, index=s.index)
    return (s - mu) / sd


# macro and feature importance
def fetch_fred(series_id: str, start: str) -> pd.Series | None:
    try:
        df = pdr.DataReader(series_id, "fred", start)
        s = df.iloc[:, 0]
        s.index = pd.to_datetime(s.index)
        s.name = series_id
        return s
    except Exception as e:
        warnings.warn(f"Failed FRED download {series_id}: {e}")
        return None


def build_macro_monthly_diff(start: str = MACRO_START) -> pd.DataFrame:
    series = {}
    for name, sid in FRED_SERIES.items():
        s = fetch_fred(sid, start=start)
        if s is None:
            continue

        # Ensure monthly end-of-month
        # If daily/biz-daily series, resample to month-end last
        s = s.resample("M").last()
        series[name] = s

    if not series:
        raise RuntimeError(
            "No macro series downloaded. Update FRED_SERIES with valid IDs or check your internet/firewall."
        )

    macro = pd.concat(series.values(), axis=1)
    macro.columns = list(series.keys())
    macro = macro.sort_index()

    # Monthly changes (deltas)
    macro_d = macro.diff().dropna(how="all")
    return macro_d


def align_monthly_returns_macro(monthly_returns: pd.DataFrame, macro_d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    common = monthly_returns.index.intersection(macro_d.index)
    y = monthly_returns.loc[common].copy()
    X = macro_d.loc[common].copy()

    X = X.dropna(axis=1, thresh=int(0.8 * len(X)))
    df = pd.concat([y, X], axis=1).dropna()
    return df[y.columns], df[X.columns]


def ols_betas(y: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    Xc = sm.add_constant(X)
    rows = []
    for t in y.columns:
        model = sm.OLS(y[t], Xc).fit()
        params = model.params.drop("const", errors="ignore")
        row = {"ticker": t, "r2": float(model.rsquared), "n_obs": int(model.nobs)}
        for k, v in params.items():
            row[f"beta_{k}"] = float(v)
        rows.append(row)
    return pd.DataFrame(rows).set_index("ticker").sort_values("r2", ascending=False)


def enet_importance(y: pd.Series, X: pd.DataFrame, random_state: int = 42) -> pd.Series:
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("enet", ElasticNetCV(
                l1_ratio=[0.1, 0.5, 0.9, 1.0],
                cv=5,
                random_state=random_state,
                max_iter=20000
            )),
        ]
    )
    pipe.fit(X, y)
    coefs = pipe.named_steps["enet"].coef_
    return pd.Series(np.abs(coefs), index=X.columns).sort_values(ascending=False)


def rf_perm_importance(y: pd.Series, X: pd.DataFrame, random_state: int = 42) -> pd.Series:
    model = RandomForestRegressor(
        n_estimators=600,
        random_state=random_state,
        min_samples_leaf=3,
        n_jobs=-1,
    )
    model.fit(X, y)
    perm = permutation_importance(model, X, y, n_repeats=20, random_state=random_state, n_jobs=-1)
    return pd.Series(perm.importances_mean, index=X.columns).sort_values(ascending=False)


def plot_importance(imp_df: pd.DataFrame, ticker: str, title: str, outpath: Path, top_k: int = 8) -> None:
    d = imp_df[imp_df["ticker"] == ticker].sort_values("importance", ascending=False).head(top_k)
    if d.empty:
        return
    plt.figure()
    plt.bar(d["feature"], d["importance"])
    plt.title(title)
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def stress_test_from_ols_betas(betas: pd.DataFrame, scenarios: dict[str, dict[str, float]]) -> pd.DataFrame:
    beta_cols = [c for c in betas.columns if c.startswith("beta_")]
    if not beta_cols:
        return pd.DataFrame()

    feat_map = {c: c.replace("beta_", "") for c in beta_cols}
    rows = []
    for scen, shocks in scenarios.items():
        for t in betas.index:
            impact = 0.0
            for bcol, feat in feat_map.items():
                if feat in shocks and pd.notna(betas.loc[t, bcol]):
                    impact += float(betas.loc[t, bcol]) * float(shocks[feat])
            rows.append({"scenario": scen, "ticker": t, "est_monthly_return_impact": impact})

    return pd.DataFrame(rows).pivot(index="ticker", columns="scenario", values="est_monthly_return_impact")


def step5() -> None:
    ensure_dirs()

    monthly = load_monthly_returns()
    tickers = [t for t in (CAN_BANKS + US_BANKS + BENCHMARKS) if t in monthly.columns]

    macro_d = build_macro_monthly_diff(start=MACRO_START)
    y, X = align_monthly_returns_macro(monthly[tickers], macro_d)

    betas = ols_betas(y, X)
    safe_write_csv(betas, TAB_DIR / "macro_betas_ols.csv")

    enet_rows, rf_rows = [], []
    for t in y.columns:
        enet_imp = enet_importance(y[t], X)
        for feat, val in enet_imp.items():
            enet_rows.append({"ticker": t, "feature": feat, "importance": float(val)})

        rf_imp = rf_perm_importance(y[t], X)
        for feat, val in rf_imp.items():
            rf_rows.append({"ticker": t, "feature": feat, "importance": float(val)})

    enet_df = pd.DataFrame(enet_rows)
    rf_df = pd.DataFrame(rf_rows)

    safe_write_csv(enet_df, TAB_DIR / "macro_feature_importance_elasticnet.csv")
    safe_write_csv(rf_df, TAB_DIR / "macro_feature_importance_rf_perm.csv")

    for t in ["RY.TO", "TD.TO", "CM.TO", "JPM", "^GSPC"]:
        plot_importance(enet_df, t, f"ElasticNet Macro Feature Importance - {t}", FIG_DIR / f"macro_enet_{t.replace('^','')}.png")
        plot_importance(rf_df, t, f"RF Permutation Macro Importance - {t}", FIG_DIR / f"macro_rf_{t.replace('^','')}.png")

    stress = stress_test_from_ols_betas(betas, STRESS_SCENARIOS)
    if not stress.empty:
        safe_write_csv(stress, TAB_DIR / "macro_stress_test_impacts.csv")

    print("Step 5 complete:", (TAB_DIR / "macro_betas_ols.csv").resolve())


# Monte Carlo
def mc_mvnormal(daily_log_returns: pd.DataFrame, tickers: list[str], horizon: int, n_paths: int) -> pd.DataFrame:
    R = daily_log_returns[tickers].dropna(how="any")
    mu = R.mean().values
    cov = R.cov().values
    sims = np.random.multivariate_normal(mean=mu, cov=cov, size=(n_paths, horizon))
    growth = np.exp(sims.sum(axis=1))
    return pd.DataFrame(growth, columns=tickers)


def mc_bootstrap(daily_log_returns: pd.DataFrame, tickers: list[str], horizon: int, n_paths: int) -> pd.DataFrame:
    R = daily_log_returns[tickers].dropna(how="any").values
    n_obs = R.shape[0]
    idx = np.random.randint(0, n_obs, size=(n_paths, horizon))
    sims = R[idx]
    growth = np.exp(sims.sum(axis=1))
    return pd.DataFrame(growth, columns=tickers)


def summarize_mc(growth: pd.DataFrame) -> pd.DataFrame:
    ret = growth - 1.0
    return pd.DataFrame({
        "mean_return": ret.mean(),
        "median_return": ret.median(),
        "p05": ret.quantile(0.05),
        "p01": ret.quantile(0.01),
        "p95": ret.quantile(0.95),
        "prob_loss": (ret < 0).mean(),
    }).sort_values("mean_return", ascending=False)


def plot_mc_hist(ret: pd.Series, title: str, outpath: Path) -> None:
    plt.figure()
    plt.hist(ret.dropna().values, bins=60)
    plt.title(title)
    plt.xlabel("Simulated 1Y return")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def step6() -> None:
    ensure_dirs()

    daily = load_daily_log_returns()
    tickers = [t for t in (CAN_BANKS + US_BANKS) if t in daily.columns]

    growth = mc_bootstrap(daily, tickers, MC_HORIZON_DAYS, MC_N_PATHS) if MC_METHOD == "bootstrap" else mc_mvnormal(daily, tickers, MC_HORIZON_DAYS, MC_N_PATHS)
    summary = summarize_mc(growth)
    safe_write_csv(summary, TAB_DIR / "monte_carlo_1y_summary.csv")

    for t in ["RY.TO", "TD.TO", "CM.TO", "JPM", "GS"]:
        if t in growth.columns:
            plot_mc_hist(growth[t] - 1.0, f"Monte Carlo 1Y Return Distribution: {t}", FIG_DIR / f"mc_1y_{t}.png")

    print("Step 6 complete:", (TAB_DIR / "monte_carlo_1y_summary.csv").resolve())


# Portfolio optimization 
def annualize_mean_cov(daily: pd.DataFrame, tickers: list[str]) -> tuple[np.ndarray, np.ndarray]:
    R = daily[tickers].dropna(how="any")
    mu_ann = R.mean().values * TRADING_DAYS
    cov_ann = R.cov().values * TRADING_DAYS
    return mu_ann, cov_ann


def portfolio_stats(w: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> tuple[float, float]:
    r = float(w @ mu)
    v = float(np.sqrt(w @ cov @ w))
    return r, v


def neg_sharpe(w: np.ndarray, mu: np.ndarray, cov: np.ndarray, rf: float = 0.0) -> float:
    r, v = portfolio_stats(w, mu, cov)
    return 1e9 if v == 0 else -((r - rf) / v)


def min_var(w: np.ndarray, cov: np.ndarray) -> float:
    return float(w @ cov @ w)


def build_constraints(tickers: list[str]) -> list[dict]:
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    can_idx = [i for i, t in enumerate(tickers) if t in CAN_BANKS]
    if can_idx:
        cons.append({"type": "ineq", "fun": lambda w, idx=can_idx: np.sum(w[idx]) - MIN_CANADA_WEIGHT})
    return cons


def build_bounds(n: int) -> list[tuple[float, float]]:
    if NO_SHORTING:
        return [(0.0, MAX_WEIGHT_PER_ASSET) for _ in range(n)]
    return [(-0.5, MAX_WEIGHT_PER_ASSET) for _ in range(n)]


def efficient_frontier(mu: np.ndarray, cov: np.ndarray, tickers: list[str], n_points: int = 35) -> pd.DataFrame:
    n = len(tickers)
    x0 = np.repeat(1 / n, n)
    bounds = build_bounds(n)
    cons = build_constraints(tickers)

    res_minv = minimize(min_var, x0, args=(cov,), method="SLSQP", bounds=bounds, constraints=cons)
    res_maxr = minimize(lambda w: -(w @ mu), x0, method="SLSQP", bounds=bounds, constraints=cons)

    w_minv = res_minv.x
    w_maxr = res_maxr.x
    r_minv, _ = portfolio_stats(w_minv, mu, cov)
    r_maxr, _ = portfolio_stats(w_maxr, mu, cov)

    targets = np.linspace(r_minv, r_maxr, n_points)

    rows = []
    for rt in targets:
        cons_rt = cons + [{"type": "eq", "fun": lambda w, rt=rt: (w @ mu) - rt}]
        res = minimize(min_var, x0, args=(cov,), method="SLSQP", bounds=bounds, constraints=cons_rt)
        if not res.success:
            continue
        w = res.x
        r, v = portfolio_stats(w, mu, cov)
        rows.append({"target_return": rt, "achieved_return": r, "vol": v, "sharpe_rf0": (r / v if v > 0 else np.nan)})
    return pd.DataFrame(rows)


def plot_frontier(df: pd.DataFrame, outpath: Path) -> None:
    if df.empty:
        return
    plt.figure()
    plt.plot(df["vol"], df["achieved_return"])
    plt.title("Efficient Frontier")
    plt.xlabel("Annualized Volatility")
    plt.ylabel("Annualized Return")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def step7() -> None:
    ensure_dirs()

    daily = load_daily_log_returns()
    tickers = [t for t in (CAN_BANKS + US_BANKS) if t in daily.columns]

    mu, cov = annualize_mean_cov(daily, tickers)
    n = len(tickers)

    x0 = np.repeat(1 / n, n)
    bounds = build_bounds(n)
    cons = build_constraints(tickers)

    res_sh = minimize(neg_sharpe, x0, args=(mu, cov, 0.0), method="SLSQP", bounds=bounds, constraints=cons)
    if not res_sh.success:
        raise RuntimeError(f"Max Sharpe failed: {res_sh.message}")

    res_mv = minimize(min_var, x0, args=(cov,), method="SLSQP", bounds=bounds, constraints=cons)
    if not res_mv.success:
        raise RuntimeError(f"Min Var failed: {res_mv.message}")

    w_sh = res_sh.x
    w_mv = res_mv.x

    weights = pd.DataFrame({"max_sharpe": w_sh, "min_variance": w_mv}, index=tickers)
    safe_write_csv(weights, TAB_DIR / "optimal_portfolio_weights.csv")

    sh_r, sh_v = portfolio_stats(w_sh, mu, cov)
    mv_r, mv_v = portfolio_stats(w_mv, mu, cov)

    can_idx = [i for i, t in enumerate(tickers) if t in CAN_BANKS]
    metrics = pd.DataFrame({
        "portfolio": ["max_sharpe_rf0", "min_variance"],
        "ann_return": [sh_r, mv_r],
        "ann_vol": [sh_v, mv_v],
        "sharpe_rf0": [(sh_r / sh_v if sh_v > 0 else np.nan), (mv_r / mv_v if mv_v > 0 else np.nan)],
        "canada_weight": [float(np.sum(w_sh[can_idx])), float(np.sum(w_mv[can_idx]))],
    }).set_index("portfolio")

    safe_write_csv(metrics, TAB_DIR / "portfolio_metrics.csv")

    frontier = efficient_frontier(mu, cov, tickers, n_points=35)
    safe_write_csv(frontier, TAB_DIR / "efficient_frontier.csv")
    plot_frontier(frontier, FIG_DIR / "efficient_frontier.png")

    print("Step 7 complete:", (TAB_DIR / "optimal_portfolio_weights.csv").resolve())



# =========================
# RUN STEPS 5–8
# =========================
def main() -> None:
    ensure_dirs()
    print("Using ROOT:", ROOT)

    print("\nStep 5...")
    step5()

    print("\nStep 6...")
    step6()

    print("\nStep 7...")
    step7()

    

    print("\nALL DONE.")
    print("Tables:", TAB_DIR.resolve())
    print("Figures:", FIG_DIR.resolve())
    print("Reports:", REP_DIR.resolve())


if __name__ == "__main__":
    main()