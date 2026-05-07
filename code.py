"""
fintual_capm_pipeline.py
Pipeline CAPM: construccion, backtest y visualizacion de portafolios ETF.

Requiere:
    pip install pandas numpy yfinance matplotlib

Estructura de carpetas (se crea automaticamente):
    data/    <- prices_3y_usd.csv, tickers_fintual.csv, tickers_sp500.csv
    output/  <- CSVs, tablas .tex y graficos .png

Uso:
    python fintual_capm_pipeline.py
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.lines import Line2D
from scipy.optimize import minimize

# ─────────────────────────────────────────────────────────────
# CONFIGURACION
# ─────────────────────────────────────────────────────────────

BASE_DIR   = Path(__file__).resolve().parent
DATA_DIR   = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

PRICES_FILE     = DATA_DIR / "prices_3y_usd.csv"
TICKERS_FINTUAL = DATA_DIR / "tickers_fintual.csv"
TICKERS_SP500   = DATA_DIR / "tickers_sp500.csv"

INITIAL_CAPITAL      = 10_000.0
RF_ANNUAL            = 0.04
TEST_DAYS            = 252
MARKET_CANDIDATES    = ["SPY", "VOO", "IVV", "VTI"]

TOP_N        = 30     # Activos max por portafolio
MAX_WEIGHT   = 0.15   # Peso maximo por activo

REBALANCE_FREQ       = "M"
DRIFT_TOLERANCE      = 0.03
TRANSACTION_COST_BPS = 10

# Paleta oscura y sobria (coherente entre todos los graficos)
C = {
    "conservative": "#1A5C2A",   # Rojo vino
    "moderate":     "#1B3A6B",   # Azul marino
    "aggressive": "#7B1C2E",   # Verde bosque
    "market":       "#2C2C2C",   # Casi negro
    "other":        "#A8B5C8",   # Gris azulado claro
}
LABELS_ES = {
    "conservative": "Conservador",
    "moderate":     "Moderado",
    "aggressive":   "Agresivo",
}

# Fallback de tickers si no existen CSVs en data/
FINTUAL_FALLBACK = [
    "VT","VTI","ITOT","VOO","IVV","SPY","ESGU","ESGV","DSI","ETHO",
    "VUG","VTV","VNQ","BND","BLV","LQD","JNK","HYG","EMB","GLD",
    "IAU","IAUM","QQQ","QQQM","VGT","XLK","FTEC","SOXX","SOXQ","SMH",
    "XSD","KOMP","CLOU","ESPO","ARKK","ARKW","EEM","EWJ","EWY","EPP",
    "CXSE","FLCH","FLIN","FLJP","FLKR","ACWI","CNRG","XLV","XLF","XLI","IWM","EFA",
]
SP500_FALLBACK = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","LLY","JPM","V","UNH",
    "XOM","AVGO","MA","JNJ","HD","PG","MRK","ABBV","COST","TSLA",
    "CVX","NEE","AMD","MCD","NFLX","QCOM","TXN","LIN","DHR","ADBE",
    "CMCSA","WMT","PEP","CAT","INTU","CRM","AMGN","GS","BA","RTX",
    "HON","SPGI","DE","GILD","BKNG","TJX","GE","C","BLK","LOW",
    "AXP","ISRG","SYK","REGN","MDT","EOG","VRTX","SCHW","PLD","MMM",
    "CB","USB","CME","ETN","ZTS","KLAC","LRCX","ICE","MDLZ","MO",
    "SO","APD","ITW","SHW","NOC","TGT","EL","AON","DUK","WM",
    "CSX","NSC","PNC","MCO","EMR","FCX","FTNT","OXY","ROP","BR",
    "LMT","ORCL","ACN","BAC","T","COP","ADP","FIS",
]


# ─────────────────────────────────────────────────────────────
# RESOLUCION DE TICKERS
# ─────────────────────────────────────────────────────────────

def resolve_tickers() -> tuple:
    """Lee tickers desde data/. Si no existen los CSV, usa listas internas."""
    def load_or_fallback(path: Path, fallback: list, label: str) -> list:
        if path.exists():
            df  = pd.read_csv(path)
            col = "ticker" if "ticker" in df.columns else df.columns[0]
            out = df[col].dropna().str.strip().tolist()
            print(f"  {label}: {len(out)} tickers desde {path.name}")
            return out
        print(f"  {label}: {path.name} no encontrado, usando lista interna ({len(fallback)})")
        return fallback

    return (
        load_or_fallback(TICKERS_FINTUAL, FINTUAL_FALLBACK, "Fintual"),
        load_or_fallback(TICKERS_SP500,   SP500_FALLBACK,   "SP500"),
    )


# ─────────────────────────────────────────────────────────────
# DESCARGA AUTOMATICA
# ─────────────────────────────────────────────────────────────

def download_prices_if_missing(fintual: list, sp500: list):
    """Descarga precios ajustados desde yfinance si prices_3y_usd.csv no existe."""
    if PRICES_FILE.exists():
        print(f"CSV encontrado: {PRICES_FILE.name}")
        return

    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: yfinance no instalado. Ejecuta: pip install yfinance")
        sys.exit(1)

    print("Descargando precios (3 anos)...")
    all_tickers = list(set(fintual + sp500))
    source_map  = {t: "fintual" for t in fintual}
    source_map.update({t: "sp500" for t in sp500 if t not in source_map})

    end, start = pd.Timestamp.today().normalize(), None
    start = end - pd.DateOffset(years=3)

    chunks = []
    for i in range(0, len(all_tickers), 100):
        batch = all_tickers[i: i + 100]
        try:
            raw   = yf.download(batch, start=start, end=end, auto_adjust=True, progress=False)
            close = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw
            chunks.append(close)
            print(f"  Batch {i // 100 + 1}: {len(batch)} tickers OK")
        except Exception as e:
            print(f"  Warning batch {i // 100 + 1}: {e}")

    if not chunks:
        sys.exit("ERROR: descarga fallida.")

    px_wide = pd.concat(chunks, axis=1)
    px_wide = px_wide.T.groupby(level=0).first().T   # Elimina duplicados entre batches

    px_long = (
        px_wide.stack(future_stack=True)
        .reset_index()
        .rename(columns={"Date": "date", "Ticker": "ticker", 0: "close"})
    )
    px_long["source"] = px_long["ticker"].map(source_map).fillna("other")
    px_long.dropna(subset=["close"]).to_csv(PRICES_FILE, index=False)
    print(f"  Guardado: {PRICES_FILE} ({len(px_long):,} filas)")


# ─────────────────────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────────────────────

def annual_to_daily(rate: float, n: int = 252) -> float:
    return (1 + rate) ** (1 / n) - 1


def latex_escape(x) -> str:
    s = str(x)
    for k, v in {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
                 "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
                 "^": r"\textasciicircum{}"}.items():
        s = s.replace(k, v)
    return s


def write_latex_table(df: pd.DataFrame, path: Path, caption: str = "", label: str = ""):
    """Tabla LaTeX estandar. Nombres de archivo con guiones, sin guion bajo."""
    cols  = list(df.columns)
    align = "l" + "r" * (len(cols) - 1)
    lines = [r"\begin{table}[htbp]", r"\centering",
             rf"\begin{{tabular}}{{{align}}}", r"\hline",
             " & ".join(latex_escape(c) for c in cols) + r" \\", r"\hline"]
    for _, row in df.iterrows():
        vals = [f"{row[c]:,.4f}" if isinstance(row[c], float) else latex_escape(row[c]) for c in cols]
        lines.append(" & ".join(vals) + r" \\")
    lines += [r"\hline", r"\end{tabular}"]
    if caption: lines.append(rf"\caption{{{latex_escape(caption)}}}")
    if label:   lines.append(rf"\label{{{latex_escape(label)}}}")
    lines.append(r"\end{table}")
    path.write_text("\n".join(lines), encoding="utf-8")


# ─────────────────────────────────────────────────────────────
# CARGA Y PREPARACION
# ─────────────────────────────────────────────────────────────

def load_prices() -> pd.DataFrame:
    df = pd.read_csv(PRICES_FILE)
    missing = {"date", "ticker", "close"} - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas: {missing}")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date", "ticker"]).drop_duplicates(["date", "ticker"], keep="last")


def prepare_price_matrix(df: pd.DataFrame) -> pd.DataFrame:
    px = df.pivot(index="date", columns="ticker", values="close").sort_index()
    return px.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")


def choose_market_proxy(returns: pd.DataFrame) -> str:
    for t in MARKET_CANDIDATES:
        if t in returns.columns:
            return t
    raise ValueError(f"No se encontro proxy de mercado. Candidatos: {MARKET_CANDIDATES}")


def split_train_test(returns: pd.DataFrame):
    """~2 anos entrenamiento + 1 ano test. Sin data leakage."""
    if len(returns) <= TEST_DAYS + 30:
        raise ValueError("Pocos datos para separar entrenamiento y backtest.")
    return returns.iloc[:-TEST_DAYS].copy(), returns.iloc[-TEST_DAYS:].copy()


# ─────────────────────────────────────────────────────────────
# ESTIMACION CAPM
# ─────────────────────────────────────────────────────────────

def estimate_capm(train: pd.DataFrame, source_map: pd.Series, market: str) -> pd.DataFrame:
    """
    E[R_i] = Rf + beta_i * (E[Rm] - Rf)
    Solo usa datos de entrenamiento (sin data leakage).
    """
    rf_d       = annual_to_daily(RF_ANNUAL)
    mkt        = train[market].dropna()
    mkt_exc    = mkt - rf_d
    mkt_var    = mkt_exc.var(ddof=1)
    mkt_prem   = mkt_exc.mean() * 252

    rows = []
    for t in train.columns:
        tmp = pd.concat([train[t], mkt], axis=1, join="inner").dropna()
        if len(tmp) < 126:
            continue
        a_exc = tmp.iloc[:, 0] - rf_d
        m_exc = tmp.iloc[:, 1] - rf_d
        beta  = np.cov(a_exc, m_exc, ddof=1)[0, 1] / mkt_var if mkt_var > 0 else np.nan
        vol   = tmp.iloc[:, 0].std(ddof=1) * np.sqrt(252)
        er    = RF_ANNUAL + beta * mkt_prem
        sharpe = (er - RF_ANNUAL) / vol if vol > 0 else np.nan
        rows.append({"ticker": t, "source": source_map.get(t, "other"),
                     "beta": beta, "returnannual": tmp.iloc[:, 0].mean() * 252,
                     "expectedreturncapm": er, "volannual": vol,
                     "sharpecapm": sharpe, "nobs": len(tmp)})

    return (pd.DataFrame(rows)
            .replace([np.inf, -np.inf], np.nan)
            .dropna(subset=["beta", "expectedreturncapm", "volannual", "sharpecapm"])
            .sort_values("sharpecapm", ascending=False)
            .reset_index(drop=True))


# ─────────────────────────────────────────────────────────────
# CONSTRUCCION DE PORTAFOLIOS
# ─────────────────────────────────────────────────────────────

def _mv_base(capm: pd.DataFrame, train_returns: pd.DataFrame):
    """Prepara mu, cov, betas y df filtrado para optimizacion media-varianza."""
    df = capm.dropna(subset=["beta", "sharpecapm", "expectedreturncapm", "volannual"]).copy()
    tickers = [t for t in df["ticker"].tolist() if t in train_returns.columns]
    df = df[df["ticker"].isin(tickers)].reset_index(drop=True)
    idx_r = [train_returns.columns.tolist().index(t) for t in df["ticker"]]
    mu  = df["expectedreturncapm"].values
    cov = train_returns.iloc[:, idx_r].dropna().cov().values * 252
    betas = df["beta"].values
    return df, mu, cov, betas


def _finalize(df, w_opt, name):
    """Trunca a TOP_N activos con mayor peso, renormaliza y etiqueta."""
    n = len(w_opt)
    order   = np.argsort(w_opt)[::-1][:TOP_N]
    w_trunc = np.zeros(n)
    w_trunc[order] = w_opt[order]
    w_trunc /= w_trunc.sum()
    df_out = df.iloc[order].copy()
    df_out["capmweight"] = w_trunc[order]
    df_out["portfolio"]  = name
    return df_out[df_out["capmweight"] > 1e-8].reset_index(drop=True)


def build_portfolio_min_vol(capm, train_returns):
    """Conservador: minimiza varianza de portafolio (beta objetivo ~0.6)."""
    df, mu, cov, betas = _mv_base(capm, train_returns)
    n = len(mu)
    def port_vol(w): return float(np.sqrt(w @ cov @ w))
    constraints = [
        {"type": "eq", "fun": lambda w: w.sum() - 1.0},
        {"type": "ineq", "fun": lambda w: 0.65 - (w @ betas)},
        {"type": "ineq", "fun": lambda w: (w @ betas) - 0.55}
    ]
    bounds = [(0.0, MAX_WEIGHT)] * n
    w0 = np.full(n, 1.0 / n)
    res = minimize(port_vol, w0, method="SLSQP",
                   bounds=bounds, constraints=constraints,
                   options={"ftol": 1e-9, "maxiter": 1000})
    w_opt = res.x if res.success else w0
    if not res.success:
        print(f"  [WARN] min_vol: {res.message}")
    return _finalize(df, w_opt, "conservative")


def build_portfolio_max_sharpe(capm, train_returns):
    """Moderado: maximiza Sharpe (sin restriccion de beta)."""
    df, mu, cov, betas = _mv_base(capm, train_returns)
    n  = len(mu)
    rf = RF_ANNUAL
    def neg_sharpe(w):
        vol = float(np.sqrt(w @ cov @ w))
        return -(float(w @ mu) - rf) / vol if vol > 1e-9 else 1e9
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(0.0, MAX_WEIGHT)] * n
    w0 = np.full(n, 1.0 / n)
    res = minimize(neg_sharpe, w0, method="SLSQP",
                   bounds=bounds, constraints=constraints,
                   options={"ftol": 1e-9, "maxiter": 1000})
    w_opt = res.x if res.success else w0
    if not res.success:
        print(f"  [WARN] max_sharpe: {res.message}")
    return _finalize(df, w_opt, "moderate")


def build_portfolio_max_return(capm, train_returns):
    """Agresivo: maximiza retorno esperado (beta objetivo ~1.4)."""
    df, mu, cov, betas = _mv_base(capm, train_returns)
    n = len(mu)
    def neg_ret(w): return -float(w @ mu)
    constraints = [
        {"type": "eq", "fun": lambda w: w.sum() - 1.0},
        {"type": "ineq", "fun": lambda w: 1.45 - (w @ betas)},
        {"type": "ineq", "fun": lambda w: (w @ betas) - 1.35}
    ]
    bounds = [(0.0, MAX_WEIGHT)] * n
    w0 = np.full(n, 1.0 / n)
    res = minimize(neg_ret, w0, method="SLSQP",
                   bounds=bounds, constraints=constraints,
                   options={"ftol": 1e-9, "maxiter": 1000})
    w_opt = res.x if res.success else w0
    if not res.success:
        print(f"  [WARN] max_return: {res.message}")
    return _finalize(df, w_opt, "aggressive")


def build_portfolios(capm: pd.DataFrame, train_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Tres portafolios sobre la FPP (misma optimizacion media-varianza, sin restriccion de beta):
      - Conservador : minima volatilidad
      - Moderado    : maximo Sharpe
      - Agresivo    : maximo retorno esperado
    """
    builders = [
        ("conservative", build_portfolio_min_vol),
        ("moderate",     build_portfolio_max_sharpe),
        ("aggressive",   build_portfolio_max_return),
    ]
    parts = []
    for name, fn in builders:
        p = fn(capm, train_returns)
        b_port = (p["beta"] * p["capmweight"]).sum()
        v_port = float(np.sqrt(
            p["capmweight"].values @
            (train_returns[[t for t in p["ticker"] if t in train_returns.columns]]
             .dropna().cov().values * 252) @
            p["capmweight"].values
        ))
        r_port = (p["expectedreturncapm"] * p["capmweight"]).sum()
        print(f"  {LABELS_ES[name]}: {len(p)} activos | "
              f"σ={v_port*100:.1f}% | E[R]={r_port*100:.1f}% | β={b_port:.3f}")
        parts.append(p)
    return pd.concat(parts, ignore_index=True)


# ─────────────────────────────────────────────────────────────
# BACKTEST
# ─────────────────────────────────────────────────────────────

def get_review_dates(px: pd.DataFrame) -> set:
    if REBALANCE_FREQ == "M":
        return set(px.resample("ME").last().index)
    if REBALANCE_FREQ == "W":
        return set(px.resample("W-FRI").last().index)
    raise ValueError("REBALANCE_FREQ debe ser 'M' o 'W'.")



class Portfolio:
    """
    Maneja el estado y rebalanceo de un portafolio.
    """
    def __init__(self, name: str, initial_capital: float, target_weights: pd.Series, initial_prices: pd.Series):
        self.name = name
        self.target_weights = target_weights
        self.tickers = target_weights.index.tolist()
        self.cash = 0.0
        self.cost_rate = TRANSACTION_COST_BPS / 10_000

        # Compras iniciales
        self.holdings = (initial_capital * self.target_weights / initial_prices).astype(float)
        self.nav = initial_capital

    def current_weights(self, current_prices: pd.Series) -> pd.Series:
        asset_val = self.holdings * current_prices
        nav = asset_val.sum() + self.cash
        return asset_val / nav if nav > 0 else asset_val * 0.0

    def get_nav(self, current_prices: pd.Series) -> float:
        return (self.holdings * current_prices).sum() + self.cash

    def needs_rebalance(self, current_prices: pd.Series, tolerance: float) -> bool:
        act_w = self.current_weights(current_prices)
        return (act_w - self.target_weights).abs().max() > tolerance

    def rebalance(self, current_prices: pd.Series, date, op_rows: list):
        pre_val = self.holdings * current_prices
        nav = pre_val.sum() + self.cash
        act_w = self.current_weights(current_prices)

        # Calcular trades teoricos
        trades = nav * self.target_weights - pre_val
        cost = trades.abs().sum() * self.cost_rate
        net_nav = nav - cost

        # Ejecutar operaciones
        self.holdings = (net_nav * self.target_weights) / current_prices
        self.cash = 0.0
        self.nav = net_nav

        post_val = self.holdings * current_prices
        post_w = post_val / net_nav if net_nav > 0 else post_val * 0.0

        # Registrar operaciones
        for t in self.tickers:
            tv = (net_nav * self.target_weights[t]) - pre_val[t]
            if abs(tv) > 1e-8:
                op_rows.append({
                    "date": date, "ticker": t,
                    "portfolio": self.name,
                    "price": current_prices[t], "tradeusd": tv,
                    "weightbefore": act_w[t], "weighttarget": self.target_weights[t],
                    "weightafter": post_w[t],
                    "transactioncostusd": abs(tv) * self.cost_rate
                })


def backtest_portfolio(test_prices: pd.DataFrame, portfolio_df: pd.DataFrame):
    """
    Simula el portafolio dia a dia usando la clase Portfolio.
    """
    tickers = portfolio_df["ticker"].tolist()
    weights = portfolio_df.set_index("ticker")["capmweight"].reindex(tickers)
    px = test_prices[tickers].dropna().copy()

    if px.empty:
        raise ValueError(f"Sin datos de test para: {tickers}")

    name = portfolio_df["portfolio"].iloc[0]
    first_date = px.index[0]
    review_dates = get_review_dates(px)

    # Inicializar Portfolio
    port = Portfolio(name, INITIAL_CAPITAL, weights, px.iloc[0])

    daily_rows, op_rows = [], []

    for dt, px_row in px.iterrows():
        # Decidir si rebalancear
        if dt in review_dates and dt != first_date and port.needs_rebalance(px_row, DRIFT_TOLERANCE):
            port.rebalance(px_row, dt, op_rows)

        nav = port.get_nav(px_row)

        daily_rows.append({
            "date": dt,
            "portfolio": name,
            "nav": nav, 
            "dailyreturn": np.nan
        })

    daily = pd.DataFrame(daily_rows).sort_values("date").reset_index(drop=True)
    daily["dailyreturn"] = daily["nav"].pct_change()
    return daily, pd.DataFrame(op_rows)


def summarize_backtest(daily: pd.DataFrame) -> dict:
    """Retorno total, anualizado, volatilidad, Sharpe y maximo drawdown."""
    rets = daily["dailyreturn"].dropna()
    nav  = daily["nav"]

    if rets.empty:
        return {"portfolio": daily["portfolio"].iloc[0],
                "navstart": nav.iloc[0], "navend": nav.iloc[-1],
                "totalreturn": np.nan, "returnannual": np.nan,
                "volannual": np.nan, "sharpe": np.nan,
                "maxdrawdown": np.nan, "ndays": len(daily)}

    total  = nav.iloc[-1] / nav.iloc[0] - 1
    ann    = (1 + total) ** (252 / len(rets)) - 1
    vol    = rets.std(ddof=1) * np.sqrt(252)
    sharpe = (ann - RF_ANNUAL) / vol if vol > 0 else np.nan
    max_dd = (nav / nav.cummax() - 1).min()

    return {"portfolio": daily["portfolio"].iloc[0],
            "navstart": nav.iloc[0], "navend": nav.iloc[-1],
            "totalreturn": total, "returnannual": ann,
            "volannual": vol, "sharpe": sharpe,
            "maxdrawdown": max_dd, "ndays": len(daily)}


# ─────────────────────────────────────────────────────────────
# GRAFICOS
# ─────────────────────────────────────────────────────────────

def _fig_style():
    """Parametros comunes de estilo para todos los graficos."""
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.grid": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10,
    })


def plot_capm_frontier(capm: pd.DataFrame, portfolios: pd.DataFrame, market: str):
    """LMV + activos individuales (gris semitransparente) + portafolios marcados con X."""
    _fig_style()
    fig, ax = plt.subplots(figsize=(11, 6))

    # Tickers seleccionados por algun portafolio
    sel_tickers = set(portfolios["ticker"])

    # Activos no seleccionados: gris claro semitransparente
    other = capm[~capm["ticker"].isin(sel_tickers) & (capm["ticker"] != market)]
    ax.scatter(other["beta"], other["expectedreturncapm"] * 100,
               color=C["other"], alpha=0.35, s=18, label="Activos ajenos a portafolios", zorder=2)

    # Activos seleccionados (coloreados por portafolio, circulo pequeno)
    for name in ["conservative", "moderate", "aggressive"]:
        sub = portfolios[portfolios["portfolio"] == name]
        ax.scatter(sub["beta"], sub["expectedreturncapm"] * 100,
                   color=C[name], alpha=0.5, s=28, zorder=3)

    # Linea del Mercado de Valores (LMV)
    beta_range  = np.linspace(capm["beta"].min() - 0.1, capm["beta"].max() + 0.1, 300)
    mkt_premium = (capm[capm["ticker"] == market]["expectedreturncapm"].values[0]
                   if market in capm["ticker"].values
                   else capm["expectedreturncapm"].mean())
    lmv_y = (RF_ANNUAL + beta_range * (mkt_premium - RF_ANNUAL)) * 100
    ax.plot(beta_range, lmv_y, color="#222222", linestyle="--", linewidth=1.2,
            label="LMV (Linea del Mercado de Valores)", zorder=4)

    # Marcadores X grandes por portafolio (centroide ponderado)
    for name in ["conservative", "moderate", "aggressive"]:
        sub   = portfolios[portfolios["portfolio"] == name]
        b_avg = (sub["beta"] * sub["capmweight"]).sum()
        r_avg = (sub["expectedreturncapm"] * sub["capmweight"]).sum() * 100
        ax.scatter(b_avg, r_avg, marker="X", s=200, color=C[name],
                   edgecolors="white", linewidths=0.8, zorder=6,
                   label=LABELS_ES[name])
        ax.annotate(LABELS_ES[name], (b_avg, r_avg),
                    textcoords="offset points", xytext=(10, -2),
                    ha="left", va="center", fontsize=9, fontweight="bold", color=C[name])

    ax.axvline(x=1.0, color="#888888", linestyle=":", linewidth=0.9, alpha=0.7)
    ax.set_xlabel("Beta (\u03b2)")
    ax.set_ylabel("Retorno esperado anual CAPM (%)")
    ax.set_title("CAPM: Riesgo sistematico vs Retorno esperado", fontsize=12, fontweight="bold")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.legend(fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "grafico-capm-frontera.png", dpi=150)
    plt.close(fig)


def plot_fpp(capm: pd.DataFrame, portfolios: pd.DataFrame, train_returns: pd.DataFrame, n_sim: int = 10_000):
    """
    Frontera de Posibilidades de Portafolio (simulacion Monte Carlo).
    FPP y portafolios usan exactamente la misma mu/cov y restricciones
    (k=TOP_N, peso max MAX_WEIGHT) para garantizar consistencia.
    Frontera eficiente suavizada via interpolacion monotona (PCHIP).
    """
    from scipy.interpolate import PchipInterpolator

    _fig_style()

    # ── universo comun: tickers CAPM que estan en train_returns ──
    df, mu, cov, betas = _mv_base(capm, train_returns)
    n = len(mu)

    # ── simulacion Monte Carlo con mismas restricciones que portafolios ──
    rng = np.random.default_rng(42)
    sim_ret, sim_vol = [], []
    for _ in range(n_sim):
        k   = min(TOP_N, n)
        idx = rng.choice(n, size=k, replace=False)
        w   = rng.dirichlet(np.ones(k))
        w   = np.clip(w, 0.0, MAX_WEIGHT)
        w  /= w.sum()
        mu_sub  = mu[idx]
        cov_sub = cov[np.ix_(idx, idx)]
        sim_ret.append(float(w @ mu_sub))
        sim_vol.append(float(np.sqrt(w @ cov_sub @ w)))

    sim_ret = np.array(sim_ret) * 100
    sim_vol = np.array(sim_vol) * 100

    # ── frontera eficiente: envolvente superior + suavizado PCHIP ──
    n_bins  = 80
    bins    = np.percentile(sim_vol, np.linspace(0, 100, n_bins + 1))
    fe_vol, fe_ret = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (sim_vol >= lo) & (sim_vol < hi)
        if mask.sum() >= 3:
            fe_vol.append(np.median(sim_vol[mask]))
            fe_ret.append(sim_ret[mask].max())
    fe_vol = np.array(fe_vol)
    fe_ret = np.array(fe_ret)
    ord_   = np.argsort(fe_vol)
    fe_vol, fe_ret = fe_vol[ord_], fe_ret[ord_]

    # Solo la parte creciente (frontera eficiente real)
    keep = np.concatenate([[True], fe_ret[1:] >= fe_ret[:-1]])
    fe_vol_plot = fe_vol[keep]
    fe_ret_plot = fe_ret[keep]

    # Interpolacion suave PCHIP (monotona, sin overshooting)
    if len(fe_vol_plot) >= 4:
        interp   = PchipInterpolator(fe_vol_plot, fe_ret_plot)
        vol_fine = np.linspace(fe_vol_plot[0], fe_vol_plot[-1], 400)
        ret_fine = interp(vol_fine)
    else:
        vol_fine, ret_fine = fe_vol_plot, fe_ret_plot

    # ── portafolios: mismo mu/cov que la simulacion ──
    col_map = {t: i for i, t in enumerate(df["ticker"].tolist())}

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.scatter(sim_vol, sim_ret, color="#6B9AC4", alpha=0.15, s=6,
               label=f"Portafolios simulados (MC, k={TOP_N}, max. {int(MAX_WEIGHT*100)}% c/u)",
               zorder=2)
    ax.plot(vol_fine, ret_fine, color="#111111", linewidth=2,
            label="Frontera eficiente", zorder=4)

    for name in ["conservative", "moderate", "aggressive"]:
        sub  = portfolios[portfolios["portfolio"] == name]
        cols = [t for t in sub["ticker"] if t in col_map]
        if not cols:
            continue
        idx2 = [col_map[t] for t in cols]
        w2   = sub.set_index("ticker").loc[cols, "capmweight"].values
        w2   = w2 / w2.sum()
        mu2  = mu[idx2]
        cov2 = cov[np.ix_(idx2, idx2)]
        r    = float(w2 @ mu2) * 100
        v    = float(np.sqrt(w2 @ cov2 @ w2)) * 100
        ax.scatter(v, r, marker="X", s=220, color=C[name],
                   edgecolors="white", linewidths=0.8, zorder=6,
                   label=LABELS_ES[name])
        ax.annotate(LABELS_ES[name], (v, r),
                    textcoords="offset points", xytext=(10, -2),
                    ha="left", va="center",
                    fontsize=9, fontweight="bold", color=C[name])

    ax.set_xlabel("Volatilidad anualizada (\u03c3)")
    ax.set_ylabel("Retorno esperado anual (CAPM)")
    ax.set_title(
        f"Frontera de Posibilidades de Portafolio\n"
        f"(universo CAPM completo | k={TOP_N} activos | peso max. {int(MAX_WEIGHT*100)}%)",
        fontsize=11, fontweight="bold")
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.legend(fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "grafico-fpp.png", dpi=150)
    plt.close(fig)

def plot_backtest_nav(daily_list: list, market_series: pd.Series, market: str):
    """NAV normalizado (base 100) de los tres portafolios vs benchmark."""
    _fig_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, daily in daily_list:
        nav_n = daily["nav"] / daily["nav"].iloc[0] * 100
        ax.plot(daily["date"], nav_n, label=LABELS_ES[name],
                color=C[name], linewidth=1.8)

    if market_series is not None and len(market_series) > 0:
        mkt_n = market_series / market_series.iloc[0] * 100
        ax.plot(mkt_n.index, mkt_n.values,
                label=f"Mercado (proxy: {market})",
                color=C["market"], linewidth=1.2, linestyle="--", alpha=0.75)

    ax.set_xlabel("Fecha")
    ax.set_ylabel("NAV normalizado (base 100)")
    ax.set_title("Backtest: evolucion del valor del portafolio", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "grafico-backtest-nav.png", dpi=150)
    plt.close(fig)


def plot_drawdown(daily_list: list):
    """
    Drawdown acumulado desde el maximo historico previo.
    DD_t = NAV_t / max(NAV_s, s<=t) - 1
    Sin relleno bajo la curva.
    """
    _fig_style()
    fig, ax = plt.subplots(figsize=(10, 5.625))

    for name, daily in daily_list:
        nav = daily["nav"]
        dd = (nav / nav.cummax() - 1) * 100
        ax.plot(daily["date"], dd, label=LABELS_ES[name],
                color=C[name], linewidth=1.5)

    ax.set_xlabel("Fecha")
    ax.set_ylabel("Drawdown (%)")
    ax.grid(False)
    ax.set_title("Drawdown del portafolio (periodo de test)",
                 fontsize=12, fontweight="bold")
    ax.yaxis.set_major_formatter(
        mtick.FuncFormatter(lambda v, _: f"{v:.1f}%")
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "grafico-drawdown.png", dpi=150)
    plt.close(fig)


def _plot_weights_single(portfolios: pd.DataFrame, name: str):
    """Genera grafico de barras horizontales de pesos, top-30, una imagen por portafolio."""
    _fig_style()
    df = (portfolios[portfolios["portfolio"] == name]
          .sort_values("capmweight", ascending=False)
          .head(30)
          .sort_values("capmweight", ascending=True))
    n = len(df)
    height = min(6.0, max(4.0, n * 0.35))
    fig, ax = plt.subplots(figsize=(10, height))
    bars = ax.barh(df["ticker"], df["capmweight"] * 100,
                   color=C[name], alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.set_title(f"{LABELS_ES[name]} — Top 30 activos",
                 fontsize=12, fontweight="bold", color=C[name], pad=10)
    ax.set_xlabel("Peso (%)")
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v:.1f}%"))
    ax.bar_label(bars, fmt="%.1f%%", fontsize=8, padding=3, color="#333333")
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"grafico-pesos-{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_weights(portfolios: pd.DataFrame):
    """Genera tres imagenes separadas: una por portafolio, top-30 activos."""
    for name in ["conservative", "moderate", "aggressive"]:
        _plot_weights_single(portfolios, name)



def plot_metrics_bar(summary: pd.DataFrame):
    """Barras comparativas de Sharpe, retorno anualizado y max drawdown."""
    _fig_style()
    names  = summary["portfolio"].tolist()
    labels = [LABELS_ES[n] for n in names]
    colors = [C[n] for n in names]

    metrics = [
        ("Ratio de Sharpe",         "sharpe",        False),
        ("Retorno anualizado (%)",   "returnannual",  True),
        ("Maximo Drawdown (%)",      "maxdrawdown",   True),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(10, 5.625))
    for ax, (title, col, as_pct) in zip(axes, metrics):
        vals = summary[col] * (100 if as_pct else 1)
        bars = ax.bar(labels, vals, color=colors, alpha=0.85,
                      edgecolor="white", linewidth=0.8)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.axhline(0, color="#888888", linewidth=0.7)
        fmt = "%.2f%%" if as_pct else "%.2f"
        ax.bar_label(bars, fmt=fmt, fontsize=9, padding=4, color="#222222")
        ax.grid(False)
        ax.tick_params(axis="x", labelsize=8)
        if as_pct:
            ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v:.1f}%"))

    fig.suptitle("Metricas comparativas por portafolio (periodo de test)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "grafico-metricas-comparativas.png", dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# EXPORTACIONES CSV + LaTeX
# ─────────────────────────────────────────────────────────────

def export_all(capm, portfolios, backtest_summary, train, test, market):
    tex_p = ["ticker", "source", "beta", "expectedreturncapm", "volannual", "sharpecapm", "capmweight"]
    tex_b = ["portfolio", "navstart", "navend", "totalreturn", "returnannual", "volannual",
             "sharpe", "maxdrawdown"]

    capm.to_csv(OUTPUT_DIR / "capm-stats.csv", index=False)
    write_latex_table(capm.head(25)[["ticker","source","beta","expectedreturncapm",
                                     "volannual","sharpecapm","nobs"]],
                      OUTPUT_DIR / "tabla-capm-top.tex",
                      caption="Top 25 activos por Sharpe CAPM", label="tab:capm-top")

    portfolios.to_csv(OUTPUT_DIR / "portfolios-capm.csv", index=False)
    for name in ["conservative", "moderate", "aggressive"]:
        df = portfolios[portfolios["portfolio"] == name]
        df.to_csv(OUTPUT_DIR / f"portfolio-{name}.csv", index=False)
        write_latex_table(df[tex_p], OUTPUT_DIR / f"tabla-portafolio-{name}.tex",
                          caption=f"Portafolio {LABELS_ES[name]}",
                          label=f"tab:portafolio-{name}")

    backtest_summary.to_csv(OUTPUT_DIR / "backtest-summary.csv", index=False)
    write_latex_table(backtest_summary[tex_b], OUTPUT_DIR / "tabla-backtest-resumen.tex",
                      caption="Resumen backtest portafolios", label="tab:backtest-resumen")
    for _, row in backtest_summary.iterrows():
        n = row["portfolio"]
        write_latex_table(pd.DataFrame([row])[tex_b],
                          OUTPUT_DIR / f"tabla-backtest-{n}.tex",
                          caption=f"Backtest {LABELS_ES[n]}",
                          label=f"tab:backtest-{n}")

    pd.DataFrame([{
        "trainstart": str(train.index.min().date()),
        "trainend":   str(train.index.max().date()),
        "teststart":  str(test.index.min().date()),
        "testend":    str(test.index.max().date()),
        "marketproxy": market, "rfannual": RF_ANNUAL, 
        "topn": TOP_N, "rebalfreq": REBALANCE_FREQ,
        "drifttolerance": DRIFT_TOLERANCE, "transactioncostbps": TRANSACTION_COST_BPS,
    }]).to_csv(OUTPUT_DIR / "run-info.csv", index=False)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Tickers y precios
    print("Resolviendo tickers...")
    fintual_t, sp500_t = resolve_tickers()
    download_prices_if_missing(fintual_t, sp500_t)

    # 2. Preparacion de datos
    print("Cargando precios...")
    raw        = load_prices()
    source_map = (raw.drop_duplicates("ticker").set_index("ticker")["source"]
                  if "source" in raw.columns else pd.Series(dtype=object))
    prices  = prepare_price_matrix(raw)
    returns = prices.pct_change().dropna(how="all")

    market = choose_market_proxy(returns)
    print(f"Proxy de mercado: {market}")

    # 3. Filtra tickers con historia suficiente (max 30 dias faltantes)
    min_obs   = len(returns) - 30
    good      = returns.notna().sum()
    good      = good[good >= min_obs].index
    returns   = returns[good].dropna(axis=0, how="any")
    prices    = prices[returns.columns].loc[returns.index]
    print(f"Tickers validos: {len(good)}")

    # 4. Split train / test (sin data leakage)
    train, test = split_train_test(returns)
    test_prices = prices.loc[test.index]
    print(f"Train: {train.index.min().date()} -> {train.index.max().date()}")
    print(f"Test : {test.index.min().date()}  -> {test.index.max().date()}")

    # 5. Estimacion CAPM y construccion de portafolios
    print("Estimando CAPM...")
    capm       = estimate_capm(train, source_map, market)
    portfolios = build_portfolios(capm, train)

    # 6. Backtest
    print("Ejecutando backtest...")
    backtest_rows, daily_list = [], []
    market_test = test_prices[market] if market in test_prices.columns else None

    for name in ["conservative", "moderate", "aggressive"]:
        p = portfolios[portfolios["portfolio"] == name].copy()
        print(f"  {LABELS_ES[name]}: {len(p)} activos")

        daily, ops = backtest_portfolio(test_prices, p)
        daily.to_csv(OUTPUT_DIR / f"backtest-daily-{name}.csv", index=False)
        ops.to_csv(OUTPUT_DIR   / f"backtest-operations-{name}.csv", index=False)

        summary = summarize_backtest(daily)
        backtest_rows.append(summary)
        daily_list.append((name, daily))

    backtest_summary = (pd.DataFrame(backtest_rows)
                        .sort_values("portfolio").reset_index(drop=True))

    # 7. Exporta CSVs y tablas LaTeX
    export_all(capm, portfolios, backtest_summary, train, test, market)

    # 8. Graficos
    print("Generando graficos...")
    plot_capm_frontier(capm, portfolios, market)
    plot_fpp(capm, portfolios, train)
    plot_backtest_nav(daily_list, market_test, market)
    plot_drawdown(daily_list)
    plot_weights(portfolios)
    plot_metrics_bar(backtest_summary)

    print("\n=== RESUMEN BACKTEST ===")
    print(backtest_summary[["portfolio","totalreturn","returnannual",
                             "volannual","sharpe","maxdrawdown"]].to_string(index=False))
    print(f"\nArchivos en: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()