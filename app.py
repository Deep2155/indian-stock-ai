 import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Indian Stock AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# CUSTOM THEME
# =========================================================
st.markdown(
    """
    <style>
    .stApp { background: #0b1220; color: #e5edf8; }
    [data-testid="stSidebar"] {
        background: #111c2f;
        border-right: 1px solid #26354d;
    }
    .block-container {
        max-width: 1500px;
        padding-top: 1.3rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3, h4 { color: #f8fafc !important; }
    [data-testid="stMetric"] {
        background: #17243a;
        border: 1px solid #26354d;
        padding: 16px;
        border-radius: 12px;
    }
    [data-testid="stMetricValue"] { color: #38bdf8; }
    .stButton > button {
        border-radius: 8px;
        min-height: 2.6rem;
        font-weight: 600;
    }
    [data-testid="stDataFrame"] {
        border: 1px solid #26354d;
        border-radius: 10px;
    }
    .small-note { color: #9fb0c7; font-size: 0.88rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# MODEL AND COMMON FUNCTIONS
# =========================================================
@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "stock_ai_model.joblib"
    package = joblib.load(model_path)
    return package["model"], package["features"]


def download_history(symbol, period="5y", interval="1d"):
    ticker = symbol.strip().upper()
    if not ticker:
        raise ValueError("Please enter an NSE stock symbol.")
    if not ticker.startswith("^") and not ticker.endswith(".NS"):
        ticker += ".NS"

    data = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    if data.empty:
        raise ValueError(f"No data available for {ticker}. Check the symbol.")
    return ticker, data


def calculate_indicators(df):
    d = df.copy()
    d["EMA20"] = d["Close"].ewm(span=20, adjust=False).mean()
    d["EMA50"] = d["Close"].ewm(span=50, adjust=False).mean()
    d["EMA200"] = d["Close"].ewm(span=200, adjust=False).mean()

    delta = d["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    d["RSI"] = 100 - (100 / (1 + rs))

    ema12 = d["Close"].ewm(span=12, adjust=False).mean()
    ema26 = d["Close"].ewm(span=26, adjust=False).mean()
    d["MACD"] = ema12 - ema26
    d["MACD_SIGNAL"] = d["MACD"].ewm(span=9, adjust=False).mean()

    previous_close = d["Close"].shift(1)
    true_range = pd.concat(
        [
            d["High"] - d["Low"],
            (d["High"] - previous_close).abs(),
            (d["Low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    d["ATR"] = true_range.rolling(14).mean()

    typical_price = (d["High"] + d["Low"] + d["Close"]) / 3
    volume_sum = d["Volume"].rolling(20).sum()
    d["VWAP_20"] = (
        (typical_price * d["Volume"]).rolling(20).sum()
        / volume_sum.replace(0, np.nan)
    )

    for days in (1, 5, 10, 20):
        d[f"Return_{days}D"] = d["Close"].pct_change(days)

    d["EMA20_Distance"] = d["Close"] / d["EMA20"] - 1
    d["EMA50_Distance"] = d["Close"] / d["EMA50"] - 1
    d["EMA200_Distance"] = d["Close"] / d["EMA200"] - 1
    d["MACD_Histogram"] = d["MACD"] - d["MACD_SIGNAL"]
    d["ATR_Percent"] = d["ATR"] / d["Close"]
    d["Volatility_20D"] = d["Return_1D"].rolling(20).std()
    d["Average_Volume_20"] = d["Volume"].rolling(20).mean()
    d["Volume_Ratio"] = (
        d["Volume"] / d["Average_Volume_20"].replace(0, np.nan)
    )
    d["Above_VWAP"] = (d["Close"] > d["VWAP_20"]).astype(float)
    return d


def detect_patterns(df):
    d = df.copy()
    o, h, low, c = d["Open"], d["High"], d["Low"], d["Close"]
    body = (c - o).abs()
    candle_range = (h - low).replace(0, np.nan)
    upper = h - pd.concat([o, c], axis=1).max(axis=1)
    lower = pd.concat([o, c], axis=1).min(axis=1) - low
    prev_o, prev_c = o.shift(1), c.shift(1)
    found_patterns = []

    for i in range(len(d)):
        found = []
        if pd.isna(candle_range.iloc[i]):
            found_patterns.append("No pattern")
            continue
        current_body = body.iloc[i]
        current_range = candle_range.iloc[i]

        if current_body <= current_range * 0.10:
            found.append("Doji")
        if (
            lower.iloc[i] >= current_body * 2
            and upper.iloc[i] <= current_body
            and current_body > 0
        ):
            found.append("Hammer")
        if (
            upper.iloc[i] >= current_body * 2
            and lower.iloc[i] <= current_body
            and current_body > 0
        ):
            found.append("Shooting Star")
        if i > 0:
            if (
                prev_c.iloc[i] < prev_o.iloc[i]
                and c.iloc[i] > o.iloc[i]
                and o.iloc[i] <= prev_c.iloc[i]
                and c.iloc[i] >= prev_o.iloc[i]
            ):
                found.append("Bullish Engulfing")
            if (
                prev_c.iloc[i] > prev_o.iloc[i]
                and c.iloc[i] < o.iloc[i]
                and o.iloc[i] >= prev_c.iloc[i]
                and c.iloc[i] <= prev_o.iloc[i]
            ):
                found.append("Bearish Engulfing")
        found_patterns.append(", ".join(found) if found else "No pattern")

    d["Candlestick_Pattern"] = found_patterns
    return d


def get_prediction(data, model, features):
    enriched = calculate_indicators(data)
    latest_features = (
        enriched[features]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .tail(1)
    )
    if latest_features.empty:
        raise ValueError("Latest technical indicators could not be calculated.")
    prediction = model.predict(latest_features)[0]
    probabilities = model.predict_proba(latest_features)[0]
    return enriched, prediction, probabilities


def probability_table(model, probabilities):
    return pd.DataFrame(
        {
            "Market Condition": model.classes_,
            "Model Probability (%)": (probabilities * 100).round(2),
        }
    )


def show_candlestick_chart(df, title, rows=180):
    chart = df.tail(rows)
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=chart.index,
            open=chart["Open"],
            high=chart["High"],
            low=chart["Low"],
            close=chart["Close"],
            name="Price",
        )
    )
    for column in ("EMA20", "EMA50", "EMA200"):
        if column in chart:
            fig.add_trace(
                go.Scatter(
                    x=chart.index,
                    y=chart[column],
                    mode="lines",
                    name=column,
                )
            )
    fig.update_layout(
        title=title,
        height=560,
        template="plotly_dark",
        paper_bgcolor="#111c2f",
        plot_bgcolor="#111c2f",
        xaxis_rangeslider_visible=False,
        margin=dict(l=20, r=20, t=55, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


def format_market_cap(value):
    if not isinstance(value, (int, float)) or pd.isna(value):
        return "Not available"
    if value >= 1e7:
        return f"₹{value / 1e7:,.2f} Cr"
    return f"₹{value:,.0f}"


# =========================================================
# SIDEBAR AND APP HEADER
# =========================================================
with st.sidebar:
    st.markdown("## 📈 INDIAN STOCK AI")
    st.caption("Personal Market Assistant")
    st.divider()
    page = st.radio(
        "NAVIGATION",
        [
            "Dashboard",
            "AI Stock Analysis",
            "Market Overview",
            "Stock Comparison",
            "Fundamentals",
            "Quarterly Results",
            "Historical Data",
            "Price Alerts",
        ],
        label_visibility="visible",
    )
    st.divider()
    if st.button("🔄 Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.caption("Data source: Yahoo Finance")
    st.caption("Data may be delayed or incomplete.")

header_left, header_right = st.columns([3, 1])
with header_left:
    st.markdown(
        "<h1 style='margin-bottom:0;'>📈 Indian Stock AI</h1>",
        unsafe_allow_html=True,
    )
    st.caption("Your Intelligent Indian Stock Market Analysis Platform")
with header_right:
    st.markdown(
        "<div style='text-align:right;padding-top:15px;'>"
        "<span style='background:#123d35;color:#4ade80;padding:8px 14px;"
        "border-radius:20px;font-size:13px;'>● AI SYSTEM READY</span></div>",
        unsafe_allow_html=True,
    )
st.divider()

try:
    model, features = load_model()
except Exception as exc:
    st.error(
        "AI model could not be loaded. Confirm that "
        "`stock_ai_model.joblib` is in the same GitHub folder as `app.py`."
    )
    st.exception(exc)
    st.stop()

# =========================================================
# PAGE: DASHBOARD
# =========================================================
if page == "Dashboard":
    st.subheader("Market Dashboard")
    st.write("Quick access to your stock research tools.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Analysis Engine", "Ready")
    c2.metric("Model", "Loaded")
    c3.metric("Market Data", "Yahoo Finance")

    st.info(
        "Choose a section from the sidebar to analyse a stock, view market "
        "indices, compare shares, review company financials or check a price level."
    )

    st.subheader("Quick Stock Analysis")
    quick_symbol = st.text_input(
        "NSE stock symbol",
        value="RELIANCE",
        key="dashboard_symbol",
    ).strip().upper()
    if st.button("Analyse this stock", type="primary"):
        st.session_state["quick_stock_symbol"] = quick_symbol
        st.session_state["navigate_to_analysis"] = True
        st.rerun()

    st.subheader("Important")
    st.warning(
        "This is an experimental research tool. Model probabilities are not "
        "calibrated guarantees or chances of profit. Data is not a live NSE feed. "
        "Verify prices and information with your broker and official filings."
    )

# =========================================================
# PAGE: AI STOCK ANALYSIS
# =========================================================
elif page == "AI Stock Analysis":
    st.subheader("🔍 AI Stock Analysis")
    history_file = Path(__file__).parent / "stock_search_history.json"
    if "stock_history" not in st.session_state:
        try:
            st.session_state.stock_history = json.loads(
                history_file.read_text(encoding="utf-8")
            )
        except Exception:
            st.session_state.stock_history = []

    previous = ["-- Select previous stock --"] + st.session_state.stock_history
    picked = st.selectbox("Previously analysed stocks", previous)
    default_symbol = st.session_state.pop(
        "quick_stock_symbol",
        picked.replace(".NS", "") if picked != previous[0] else "RELIANCE",
    )
    symbol = st.text_input(
        "Enter NSE stock symbol",
        value=default_symbol,
        key="analysis_symbol",
    ).strip().upper()
    risk_level = st.selectbox(
        "Risk level",
        ["CONSERVATIVE", "MODERATE", "AGGRESSIVE"],
    )

    if st.button("ANALYSE STOCK", type="primary"):
        try:
            with st.spinner("Downloading historical data and analysing..."):
                ticker, raw = download_history(symbol, "5y")
                if len(raw) < 300:
                    st.error("Insufficient historical data for this analysis.")
                    st.stop()

                enriched, prediction, probabilities = get_prediction(
                    raw, model, features
                )
                enriched = detect_patterns(enriched)
                latest = enriched.iloc[-1]

                if ticker not in st.session_state.stock_history:
                    st.session_state.stock_history.insert(0, ticker)
                    st.session_state.stock_history = (
                        st.session_state.stock_history[:30]
                    )
                    history_file.write_text(
                        json.dumps(st.session_state.stock_history, indent=2),
                        encoding="utf-8",
                    )

                st.session_state["last_analysis"] = {
                    "ticker": ticker,
                    "data": enriched,
                    "prediction": str(prediction),
                    "probabilities": probabilities,
                    "risk_level": risk_level,
                }
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")

    result = st.session_state.get("last_analysis")
    if result:
        ticker = result["ticker"]
        enriched = result["data"]
        prediction = result["prediction"]
        probabilities = result["probabilities"]
        latest = enriched.iloc[-1]

        st.markdown(f"### {ticker} — AI Analysis")
        a, b, c = st.columns(3)
        a.metric("Latest Available Price", f"₹{float(latest['Close']):,.2f}")
        b.metric("AI Prediction", prediction)
        b.caption("Experimental model output")
        c.metric("RSI", f"{float(latest['RSI']):.2f}")
        st.caption(f"Latest available daily data: {enriched.index[-1].date()}")

        st.subheader("Model Probabilities")
        st.dataframe(
            probability_table(model, probabilities),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Technical Indicators")
        indicator_rows = [
            ("EMA 20", "EMA20"),
            ("EMA 50", "EMA50"),
            ("EMA 200", "EMA200"),
            ("MACD", "MACD"),
            ("ATR", "ATR"),
            ("20-Day Rolling VWAP", "VWAP_20"),
        ]
        indicators = pd.DataFrame(
            {
                "Indicator": [name for name, _ in indicator_rows],
                "Value": [
                    round(float(latest[key]), 2) for _, key in indicator_rows
                ],
            }
        )
        st.dataframe(indicators, use_container_width=True, hide_index=True)

        st.subheader("Automatic Candlestick Pattern Recognition")
        recent = enriched[
            enriched["Candlestick_Pattern"] != "No pattern"
        ][["Close", "Candlestick_Pattern"]].tail(10)
        if recent.empty:
            st.info("No recognised candlestick patterns in the available recent data.")
        else:
            st.dataframe(
                recent.sort_index(ascending=False),
                use_container_width=True,
            )

        risk_settings = {
            "CONSERVATIVE": (1.0, 1.5),
            "MODERATE": (1.5, 2.5),
            "AGGRESSIVE": (2.0, 3.5),
        }
        stop_mult, target_mult = risk_settings[result["risk_level"]]
        price = float(latest["Close"])
        atr = float(latest["ATR"])
        st.subheader("Illustrative Target & Stop Loss")
        if prediction == "BULLISH":
            stop, target = price - stop_mult * atr, price + target_mult * atr
        elif prediction == "BEARISH":
            stop, target = price + stop_mult * atr, price - target_mult * atr
        else:
            stop, target = None, None

        if target is not None:
            x, y = st.columns(2)
            x.metric("Illustrative Target", f"₹{target:,.2f}")
            y.metric("Illustrative Stop Loss", f"₹{stop:,.2f}")
        else:
            st.info("Sideways prediction: no directional levels suggested.")

        show_candlestick_chart(enriched, f"{ticker} — Candlestick Chart")
        st.warning(
            "Research prototype only. Model probabilities are not guaranteed "
            "chances of profit. This is not live NSE data or a recommendation to trade."
        )

    with st.expander("Stock Search History"):
        if st.session_state.stock_history:
            st.write(st.session_state.stock_history)
        else:
            st.caption("No stocks analysed yet.")

# =========================================================
# PAGE: MARKET OVERVIEW
# =========================================================
elif page == "Market Overview":
    st.subheader("🇮🇳 Indian Market Overview")
    st.caption("AI overview of NIFTY 50 and BANK NIFTY using available Yahoo Finance data.")
    if st.button("ANALYSE NIFTY & BANK NIFTY", type="primary"):
        index_list = {"NIFTY 50": "^NSEI", "BANK NIFTY": "^NSEBANK"}
        results = []
        for name, symbol in index_list.items():
            try:
                with st.spinner(f"Analysing {name}..."):
                    _, raw = download_history(symbol, "5y")
                    enriched, prediction, probabilities = get_prediction(
                        raw, model, features
                    )
                    latest = enriched.dropna(subset=features).iloc[-1]
                    results.append(
                        {
                            "Index": name,
                            "Latest Close": round(float(latest["Close"]), 2),
                            "AI Trend": str(prediction),
                            "Model Probability (%)": round(
                                float(np.max(probabilities)) * 100, 2
                            ),
                            "RSI": round(float(latest["RSI"]), 2),
                        }
                    )
            except Exception as exc:
                st.warning(f"{name}: {exc}")
        if results:
            st.dataframe(
                pd.DataFrame(results),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.error("Market overview could not be generated.")

# =========================================================
# PAGE: STOCK COMPARISON
# =========================================================
elif page == "Stock Comparison":
    st.subheader("📊 Multi-Stock AI Comparison")
    symbols_text = st.text_input(
        "Enter up to 5 NSE stocks separated by commas",
        value="RELIANCE,TCS,INFY",
        key="compare_symbols",
    )
    if st.button("COMPARE STOCKS", type="primary"):
        symbols = [x.strip().upper() for x in symbols_text.split(",") if x.strip()]
        if not symbols:
            st.warning("Enter at least one stock symbol.")
        elif len(symbols) > 5:
            st.warning("Maximum 5 stocks can be compared at a time.")
        else:
            results = []
            for stock in symbols:
                try:
                    with st.spinner(f"Analysing {stock}..."):
                        _, raw = download_history(stock, "5y")
                        enriched, prediction, probabilities = get_prediction(
                            raw, model, features
                        )
                        latest = enriched.dropna(subset=features).iloc[-1]
                        probability_map = dict(zip(model.classes_, probabilities))
                        results.append(
                            {
                                "Stock": stock,
                                "Price (₹)": round(float(latest["Close"]), 2),
                                "AI Prediction": str(prediction),
                                "Model Confidence (%)": round(
                                    float(max(probabilities)) * 100, 2
                                ),
                                "Bullish (%)": round(
                                    float(probability_map.get("BULLISH", 0)) * 100, 2
                                ),
                                "Bearish (%)": round(
                                    float(probability_map.get("BEARISH", 0)) * 100, 2
                                ),
                                "Sideways (%)": round(
                                    float(probability_map.get("SIDEWAYS", 0)) * 100, 2
                                ),
                                "RSI": round(float(latest["RSI"]), 2),
                                "MACD": round(float(latest["MACD"]), 2),
                            }
                        )
                except Exception as exc:
                    st.warning(f"{stock}: {exc}")

            if results:
                comparison = pd.DataFrame(results)
                st.session_state["comparison_results"] = comparison
            else:
                st.session_state.pop("comparison_results", None)

    comparison = st.session_state.get("comparison_results")
    if comparison is not None:
        st.subheader("Comparison Results")
        st.dataframe(comparison, use_container_width=True, hide_index=True)
        st.download_button(
            "📥 Download Comparison Report",
            data=comparison.to_csv(index=False).encode("utf-8"),
            file_name="stock_ai_comparison.csv",
            mime="text/csv",
        )
        st.caption(
            "Model confidence is not a guaranteed probability of profit. "
            "Predictions are experimental and based on historical data."
        )

    st.divider()
    st.subheader("📈 Relative Stock Performance")
    if st.button("SHOW 1-YEAR PERFORMANCE CHART"):
        symbols = [x.strip().upper() for x in symbols_text.split(",") if x.strip()]
        fig = go.Figure()
        for stock in symbols[:5]:
            try:
                _, hist = download_history(stock, "1y")
                close = hist["Close"].dropna()
                if len(close) > 1:
                    normalized = close / close.iloc[0] * 100
                    fig.add_trace(
                        go.Scatter(
                            x=normalized.index,
                            y=normalized,
                            mode="lines",
                            name=stock,
                        )
                    )
            except Exception as exc:
                st.warning(f"{stock}: {exc}")
        if fig.data:
            fig.update_layout(
                title="1-Year Relative Stock Performance",
                xaxis_title="Date",
                yaxis_title="Growth Index (Starting Value = 100)",
                hovermode="x unified",
                template="plotly_dark",
                paper_bgcolor="#111c2f",
                plot_bgcolor="#111c2f",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Historical relative performance only. Starting value is "
                "normalised to 100; this is not a future return forecast."
            )
        else:
            st.info("No performance data could be plotted.")

# =========================================================
# PAGE: FUNDAMENTALS
# =========================================================
elif page == "Fundamentals":
    st.subheader("🏢 Company Fundamentals & Shareholding")
    symbol = st.text_input(
        "Enter NSE company symbol",
        value="RELIANCE",
        key="fundamentals_symbol",
    ).strip().upper()
    if st.button("GET COMPANY FUNDAMENTALS", type="primary"):
        try:
            company = yf.Ticker(symbol if symbol.endswith(".NS") else symbol + ".NS")
            info = company.info
            st.markdown(f"### {info.get('longName', symbol)}")
            details = {
                "Sector": info.get("sector", "Not available"),
                "Industry": info.get("industry", "Not available"),
                "Market Cap": format_market_cap(info.get("marketCap")),
                "P/E Ratio": info.get("trailingPE", "Not available"),
                "Price to Book": info.get("priceToBook", "Not available"),
                "Dividend Yield": info.get("dividendYield", "Not available"),
                "52 Week High": info.get("fiftyTwoWeekHigh", "Not available"),
                "52 Week Low": info.get("fiftyTwoWeekLow", "Not available"),
            }
            st.subheader("Company Details")
            st.dataframe(
                pd.DataFrame(details.items(), columns=["Particulars", "Value"]),
                use_container_width=True,
                hide_index=True,
            )
            annual = company.financials
            st.subheader("Annual Financial Results")
            if annual is not None and not annual.empty:
                st.dataframe(annual.T.head(5), use_container_width=True)
            else:
                st.info("Annual financial results are not available.")
            for title, attr in [
                ("Major Shareholders", "major_holders"),
                ("Institutional Shareholders", "institutional_holders"),
            ]:
                st.subheader(title)
                holders = getattr(company, attr)
                if holders is not None and not holders.empty:
                    st.dataframe(holders, use_container_width=True)
                else:
                    st.info(f"{title} data is not available.")
            st.caption(
                "Source: Yahoo Finance. Data may be delayed, incomplete or unavailable. "
                "Verify financial information from official NSE/BSE filings."
            )
        except Exception as exc:
            st.error(f"Could not fetch company data: {exc}")

# =========================================================
# PAGE: QUARTERLY RESULTS
# =========================================================
elif page == "Quarterly Results":
    st.subheader("📅 Quarterly Financial Results")
    symbol = st.text_input(
        "Enter NSE symbol",
        value="RELIANCE",
        key="quarterly_symbol",
    ).strip().upper()
    if st.button("GET QUARTERLY RESULTS", type="primary"):
        try:
            company = yf.Ticker(symbol if symbol.endswith(".NS") else symbol + ".NS")
            for title, attr in [
                ("Quarterly Income Statement", "quarterly_income_stmt"),
                ("Quarterly Balance Sheet", "quarterly_balance_sheet"),
                ("Quarterly Cash Flow", "quarterly_cashflow"),
            ]:
                st.subheader(title)
                result = getattr(company, attr)
                if result is not None and not result.empty:
                    st.dataframe(result.T.head(8), use_container_width=True)
                else:
                    st.info(f"{title} unavailable.")
            st.caption("Source: Yahoo Finance. Quarterly data may be incomplete or delayed.")
        except Exception as exc:
            st.error(f"Quarterly data error: {exc}")

# =========================================================
# PAGE: HISTORICAL DATA
# =========================================================
elif page == "Historical Data":
    st.subheader("📥 Download Historical Stock Data")
    symbol = st.text_input(
        "Enter NSE stock symbol",
        value="RELIANCE",
        key="historical_symbol",
    ).strip().upper()
    period = st.selectbox(
        "Select historical period",
        ["1y", "2y", "5y", "10y"],
        index=2,
        key="historical_period",
    )
    if st.button("GET HISTORICAL DATA", type="primary"):
        try:
            ticker, hist = download_history(symbol, period)
            hist = hist.reset_index()
            st.success(f"Downloaded {len(hist)} daily records for {ticker}.")
            st.dataframe(hist.tail(30), use_container_width=True, hide_index=True)
            st.download_button(
                "📥 Download Historical Data CSV",
                data=hist.to_csv(index=False).encode("utf-8"),
                file_name=f"{symbol}_{period}_history.csv",
                mime="text/csv",
            )
        except Exception as exc:
            st.error(f"Historical data error: {exc}")
    st.caption("Source: Yahoo Finance. Historical data availability and accuracy may vary.")

# =========================================================
# PAGE: PRICE ALERTS
# =========================================================
elif page == "Price Alerts":
    st.subheader("🔔 Stock Price Alert Check")
    st.caption(
        "This page checks the latest available daily closing price when you press the button. "
        "It does not monitor prices continuously or send background notifications."
    )
    symbol = st.text_input(
        "NSE Stock Symbol",
        value="RELIANCE",
        key="alert_symbol",
    ).strip().upper()
    target = st.number_input(
        "Target Price (₹)",
        min_value=0.0,
        value=1600.0,
        step=10.0,
        key="alert_target",
    )
    stoploss = st.number_input(
        "Stop Loss Price (₹)",
        min_value=0.0,
        value=1300.0,
        step=10.0,
        key="alert_stoploss",
    )
    if st.button("CHECK PRICE LEVEL", type="primary"):
        try:
            _, alert_data = download_history(symbol, "5d")
            current = float(alert_data["Close"].dropna().iloc[-1])
            st.metric("Latest Available Closing Price", f"₹{current:,.2f}")
            if target > 0 and current >= target:
                st.success("🎯 Target price reached.")
            if stoploss > 0 and current <= stoploss:
                st.error("⚠️ Stop-loss level reached.")
            if target > 0 and stoploss > 0 and stoploss < current < target:
                st.info("Price is between your target and stop-loss levels.")
            st.caption(
                "Manual price check using the latest available daily data. "
                "Not a continuous live alert. Verify prices with your broker."
            )
        except Exception as exc:
            st.error(f"Price alert error: {exc}")

st.divider()
st.caption(
    "Indian Stock AI | Personal research prototype | "
    "Not investment advice. Always verify data independently."
)
