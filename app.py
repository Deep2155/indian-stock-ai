import json

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

st.set_page_config(
    page_title="Indian Stock AI",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Indian Stock AI Prediction")
st.caption("Experimental stock analysis and machine learning dashboard")

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("stock_ai_model.joblib")

try:
    package = load_model()
    model = package["model"]
    features = package["features"]
except Exception as e:
    st.error("AI model file not found. Run Step 36C first.")
    st.stop()


# Technical indicators
def calculate_indicators(df):

    d = df.copy()

    d["EMA20"] = d["Close"].ewm(span=20, adjust=False).mean()
    d["EMA50"] = d["Close"].ewm(span=50, adjust=False).mean()
    d["EMA200"] = d["Close"].ewm(span=200, adjust=False).mean()

    delta = d["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    d["RSI"] = 100 - (100 / (1 + rs))

    ema12 = d["Close"].ewm(span=12, adjust=False).mean()
    ema26 = d["Close"].ewm(span=26, adjust=False).mean()

    d["MACD"] = ema12 - ema26
    d["MACD_SIGNAL"] = d["MACD"].ewm(span=9, adjust=False).mean()

    previous_close = d["Close"].shift(1)

    tr = pd.concat([
        d["High"] - d["Low"],
        abs(d["High"] - previous_close),
        abs(d["Low"] - previous_close)
    ], axis=1).max(axis=1)

    d["ATR"] = tr.rolling(14).mean()

    typical_price = (d["High"] + d["Low"] + d["Close"]) / 3
    volume_sum = d["Volume"].rolling(20).sum()

    d["VWAP_20"] = (
        (typical_price * d["Volume"]).rolling(20).sum()
        / volume_sum.replace(0, np.nan)
    )

    d["Return_1D"] = d["Close"].pct_change(1)
    d["Return_5D"] = d["Close"].pct_change(5)
    d["Return_10D"] = d["Close"].pct_change(10)
    d["Return_20D"] = d["Close"].pct_change(20)

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

    d["Above_VWAP"] = (
        d["Close"] > d["VWAP_20"]
    ).astype(float)

    return d




# AUTOMATIC CANDLESTICK PATTERN DETECTOR
def detect_patterns(df):

    d = df.copy()

    o = d["Open"]
    h = d["High"]
    l = d["Low"]
    c = d["Close"]

    body = (c - o).abs()
    candle_range = (h - l).replace(0, np.nan)

    upper_shadow = h - pd.concat([o, c], axis=1).max(axis=1)
    lower_shadow = pd.concat([o, c], axis=1).min(axis=1) - l

    previous_open = o.shift(1)
    previous_close = c.shift(1)

    patterns = []

    for i in range(len(d)):

        found = []

        if pd.isna(candle_range.iloc[i]):
            patterns.append("No pattern")
            continue

        current_body = body.iloc[i]
        current_range = candle_range.iloc[i]

        # Doji
        if current_body <= current_range * 0.10:
            found.append("Doji")

        # Hammer
        if (
            lower_shadow.iloc[i] >= current_body * 2
            and upper_shadow.iloc[i] <= current_body
            and current_body > 0
        ):
            found.append("Hammer")

        # Shooting Star
        if (
            upper_shadow.iloc[i] >= current_body * 2
            and lower_shadow.iloc[i] <= current_body
            and current_body > 0
        ):
            found.append("Shooting Star")

        if i > 0:

            # Bullish Engulfing
            if (
                previous_close.iloc[i] < previous_open.iloc[i]
                and c.iloc[i] > o.iloc[i]
                and o.iloc[i] <= previous_close.iloc[i]
                and c.iloc[i] >= previous_open.iloc[i]
            ):
                found.append("Bullish Engulfing")

            # Bearish Engulfing
            if (
                previous_close.iloc[i] > previous_open.iloc[i]
                and c.iloc[i] < o.iloc[i]
                and o.iloc[i] >= previous_close.iloc[i]
                and c.iloc[i] <= previous_open.iloc[i]
            ):
                found.append("Bearish Engulfing")

        patterns.append(", ".join(found) if found else "No pattern")

    d["Candlestick_Pattern"] = patterns

    return d



# Refresh market data
refresh_col, refresh_info = st.columns([1, 2])

with refresh_col:
    if st.button("🔄 Refresh Market Data"):
        st.cache_data.clear()
        st.session_state["run_analysis"] = True
        st.rerun()

with refresh_info:
    st.caption("Fetch the latest available data from Yahoo Finance.")


# Stock search history
if "stock_history" not in st.session_state:
    try:
        with open("/content/stock_search_history.json", "r") as f:
            st.session_state["stock_history"] = json.load(f)
    except:
        st.session_state["stock_history"] = []

# Select stock from previous history
if "stock_symbol_input" not in st.session_state:
    st.session_state["stock_symbol_input"] = "RELIANCE"

def select_history_stock():
    selected = st.session_state["history_pick"]
    if selected != "-- Select previous stock --":
        st.session_state["stock_symbol_input"] = selected.replace(".NS", "")

history_options = ["-- Select previous stock --"] + st.session_state.get("stock_history", [])

st.selectbox(
    "Select a previously analysed stock",
    history_options,
    key="history_pick",
    on_change=select_history_stock
)

# User input
col1, col2 = st.columns([2, 1])

with col1:
    symbol = st.text_input(
        "Enter NSE stock symbol",
        key="stock_symbol_input"
    ).strip().upper()

with col2:
    risk_level = st.selectbox(
        "Risk level",
        ["CONSERVATIVE", "MODERATE", "AGGRESSIVE"]
    )


# Previously analysed stocks
with st.expander("📚 Stock Search History", expanded=False):
    history = st.session_state.get("stock_history", [])
    if history:
        for item in history:
            st.write("•", item)
    else:
        st.caption("No stocks analysed yet.")

analyse = st.button("ANALYSE STOCK", type="primary")

if analyse or st.session_state.pop("run_analysis", False):

    if not symbol:
        st.warning("Please enter a stock symbol.")
        st.stop()

    ticker = symbol if symbol.endswith(".NS") else symbol + ".NS"

    with st.spinner("Downloading stock data and analysing..."):

        try:
            df = yf.download(
                ticker,
                period="5y",
                interval="1d",
                auto_adjust=True,
                progress=False
            )

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            if df.empty or len(df) < 300:
                st.error("Stock data unavailable or insufficient.")
                st.stop()

            if ticker not in st.session_state["stock_history"]:
                st.session_state["stock_history"].insert(0, ticker)

            with open("/content/stock_search_history.json", "w") as f:
                json.dump(st.session_state["stock_history"], f)

            df = calculate_indicators(df)
            df = detect_patterns(df)

            latest_features = (
                df[features]
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
                .tail(1)
            )

            if latest_features.empty:
                st.error("Could not calculate latest indicators.")
                st.stop()

            prediction = model.predict(latest_features)[0]
            probabilities = model.predict_proba(latest_features)[0]

            latest = df.iloc[-1]

            st.subheader(ticker + " — AI Analysis")

            m1, m2, m3 = st.columns(3)

            m1.metric(
                "Latest Available Price",
                f"₹{float(latest['Close']):,.2f}"
            )

            m2.metric("AI Prediction", prediction)

            m3.metric(
                "RSI",
                f"{float(latest['RSI']):.2f}"
            )

            st.caption(
                "Latest available market data: "
                + str(df.index[-1].date())
            )

            st.subheader("Model Probabilities")

            probability_df = pd.DataFrame({
                "Market Condition": model.classes_,
                "Probability (%)": (probabilities * 100).round(2)
            })

            st.dataframe(
                probability_df,
                use_container_width=True,
                hide_index=True
            )

            st.subheader("Technical Indicators")

            indicators = pd.DataFrame({
                "Indicator": [
                    "EMA 20", "EMA 50", "EMA 200",
                    "MACD", "ATR", "20-Day Rolling VWAP"
                ],
                "Value": [
                    round(float(latest["EMA20"]), 2),
                    round(float(latest["EMA50"]), 2),
                    round(float(latest["EMA200"]), 2),
                    round(float(latest["MACD"]), 2),
                    round(float(latest["ATR"]), 2),
                    round(float(latest["VWAP_20"]), 2)
                ]
            })

            st.dataframe(
                indicators,
                use_container_width=True,
                hide_index=True
            )

            # CANDLESTICK PATTERN RESULTS
            st.subheader("Automatic Candlestick Pattern Recognition")

            recent_patterns = df[
                df["Candlestick_Pattern"] != "No pattern"
            ][["Close", "Candlestick_Pattern"]].tail(10)

            if recent_patterns.empty:
                st.info("No recognised candlestick patterns in recent data.")
            else:
                st.dataframe(
                    recent_patterns.sort_index(ascending=False),
                    use_container_width=True
                )

            # Illustrative ATR levels
            settings = {
                "CONSERVATIVE": (1.0, 1.5),
                "MODERATE": (1.5, 2.5),
                "AGGRESSIVE": (2.0, 3.5)
            }

            stop_multiple, target_multiple = settings[risk_level]

            price = float(latest["Close"])
            atr = float(latest["ATR"])

            if prediction == "BULLISH":
                stop = price - stop_multiple * atr
                target = price + target_multiple * atr

            elif prediction == "BEARISH":
                stop = price + stop_multiple * atr
                target = price - target_multiple * atr

            else:
                stop = None
                target = None

            st.subheader("Illustrative Target & Stop Loss")

            if target is not None:
                a, b = st.columns(2)
                a.metric("Illustrative Target", f"₹{target:,.2f}")
                b.metric("Illustrative Stop Loss", f"₹{stop:,.2f}")
            else:
                st.info("Sideways prediction: no directional levels suggested.")

            # Candlestick chart
            chart = df.tail(180)

            fig = go.Figure()

            fig.add_trace(go.Candlestick(
                x=chart.index,
                open=chart["Open"],
                high=chart["High"],
                low=chart["Low"],
                close=chart["Close"],
                name="Price"
            ))

            for column in ["EMA20", "EMA50", "EMA200"]:
                fig.add_trace(go.Scatter(
                    x=chart.index,
                    y=chart[column],
                    mode="lines",
                    name=column
                ))

            fig.update_layout(
                title=ticker + " — Candlestick Chart",
                height=650,
                xaxis_rangeslider_visible=False
            )

            st.plotly_chart(fig, use_container_width=True)

            st.warning(
                "Research prototype only. Model probabilities are not "
                "guaranteed chances of profit. This is not live NSE data "
                "or a recommendation to trade."
            )

        except Exception as e:
            st.error("Analysis failed: " + str(e))



# =====================================
# MULTI STOCK AI COMPARISON
# =====================================

st.divider()
st.header("📊 Multi-Stock AI Comparison")

comparison_symbols = st.text_input(
    "Enter stocks separated by commas",
    value="RELIANCE,TCS,INFY",
    key="comparison_symbols"
)

if st.button("COMPARE STOCKS", type="primary"):

    symbols_list = [
        x.strip().upper()
        for x in comparison_symbols.split(",")
        if x.strip()
    ]

    if not symbols_list:
        st.warning("Please enter stock symbols.")
    elif len(symbols_list) > 5:
        st.warning("Please compare maximum 5 stocks at a time.")
    else:

        comparison_results = []

        for stock in symbols_list:

            ticker = stock if stock.endswith(".NS") else stock + ".NS"

            try:
                with st.spinner(f"Analysing {ticker}..."):

                    data = yf.download(
                        ticker,
                        period="5y",
                        interval="1d",
                        auto_adjust=True,
                        progress=False
                    )

                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = data.columns.get_level_values(0)

                    if data.empty or len(data) < 300:
                        st.warning(f"Insufficient data for {ticker}")
                        continue

                    data = calculate_indicators(data)

                    latest_features = (
                        data[features]
                        .replace([np.inf, -np.inf], np.nan)
                        .dropna()
                        .tail(1)
                    )

                    if latest_features.empty:
                        st.warning(f"Indicators unavailable for {ticker}")
                        continue

                    prediction = model.predict(latest_features)[0]
                    probabilities = model.predict_proba(latest_features)[0]

                    probability_map = dict(
                        zip(model.classes_, probabilities)
                    )

                    latest = data.iloc[-1]

                    comparison_results.append({
                        "Stock": stock,
                        "Price": round(float(latest["Close"]), 2),
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
                        "MACD": round(float(latest["MACD"]), 2)
                    })

            except Exception as e:
                st.warning(f"{ticker}: Analysis failed - {e}")

        if comparison_results:

            comparison_df = pd.DataFrame(comparison_results)

            st.subheader("Comparison Results")
            st.dataframe(
                comparison_df,
                use_container_width=True,
                hide_index=True
            )

            csv_report = comparison_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="📥 Download Comparison Report",
                data=csv_report,
                file_name="stock_ai_comparison.csv",
                mime="text/csv"
            )

            st.caption(
                "Model confidence is not a guaranteed probability of profit. "
                "Predictions are experimental and based on historical data."
            )




# =====================================
# MULTI STOCK PERFORMANCE COMPARISON
# =====================================

st.divider()
st.header("📈 Stock Performance Comparison")

if st.button("SHOW 1-YEAR PERFORMANCE CHART"):

    symbols = [
        x.strip().upper()
        for x in comparison_symbols.split(",")
        if x.strip()
    ]

    performance_chart = go.Figure()

    for stock in symbols[:5]:

        ticker = stock if stock.endswith(".NS") else stock + ".NS"

        try:
            history_data = yf.download(
                ticker,
                period="1y",
                interval="1d",
                auto_adjust=True,
                progress=False
            )

            if isinstance(history_data.columns, pd.MultiIndex):
                history_data.columns = history_data.columns.get_level_values(0)

            if history_data.empty:
                st.warning(f"Data unavailable for {ticker}")
                continue

            close = history_data["Close"].dropna()

            if len(close) < 2:
                continue

            normalized = (close / close.iloc[0]) * 100

            performance_chart.add_trace(
                go.Scatter(
                    x=normalized.index,
                    y=normalized,
                    mode="lines",
                    name=stock
                )
            )

        except Exception as e:
            st.warning(f"{ticker}: {e}")

    if performance_chart.data:

        performance_chart.update_layout(
            title="1-Year Relative Stock Performance",
            xaxis_title="Date",
            yaxis_title="Growth Index (Starting Value = 100)",
            hovermode="x unified",
            template="plotly_white"
        )

        st.plotly_chart(
            performance_chart,
            use_container_width=True
        )

        st.caption(
            "Historical relative performance only. "
            "Starting value is normalized to 100; this is not a future return forecast."
        )



# NIFTY and BANK NIFTY AI Market Overview
st.divider()
st.header("🇮🇳 Indian Market Overview")

if st.button("🔍 ANALYSE NIFTY & BANK NIFTY"):
    index_list = {
        "NIFTY 50": "^NSEI",
        "BANK NIFTY": "^NSEBANK"
    }

    overview_results = []

    for index_name, index_symbol in index_list.items():
        try:
            index_data = yf.download(
                index_symbol,
                period="5y",
                interval="1d",
                auto_adjust=True,
                progress=False
            )

            if isinstance(index_data.columns, pd.MultiIndex):
                index_data.columns = index_data.columns.get_level_values(0)

            index_data = calculate_indicators(index_data)
            latest_row = index_data.dropna(subset=features).tail(1)

            if latest_row.empty:
                continue

            latest_features = latest_row[features]
            prediction = model.predict(latest_features)[0]
            probabilities = model.predict_proba(latest_features)[0]
            confidence = max(probabilities) * 100

            overview_results.append({
                "Index": index_name,
                "Latest Close": round(float(latest_row["Close"].iloc[0]), 2),
                "AI Trend": str(prediction),
                "Model Confidence": f"{confidence:.2f}%"
            })

        except Exception as e:
            st.warning(f"{index_name}: Data unavailable - {e}")

    if overview_results:
        st.dataframe(
            pd.DataFrame(overview_results),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.error("Market overview could not be generated.")


# Company Fundamentals and Shareholding
st.divider()
st.header("🏢 Company Fundamentals & Shareholding")

fundamental_symbol = st.text_input(
    "Enter NSE company symbol",
    value="RELIANCE",
    key="fundamental_symbol"
).strip().upper()

if st.button("📊 GET COMPANY FUNDAMENTALS"):
    try:
        company_ticker = yf.Ticker(fundamental_symbol + ".NS")
        company_info = company_ticker.info

        st.subheader(company_info.get("longName", fundamental_symbol))

        details = {
            "Sector": company_info.get("sector", "Not available"),
            "Industry": company_info.get("industry", "Not available"),
            "Market Cap": company_info.get("marketCap", "Not available"),
            "P/E Ratio": company_info.get("trailingPE", "Not available"),
            "Price to Book": company_info.get("priceToBook", "Not available"),
            "Dividend Yield": company_info.get("dividendYield", "Not available"),
            "52 Week High": company_info.get("fiftyTwoWeekHigh", "Not available"),
            "52 Week Low": company_info.get("fiftyTwoWeekLow", "Not available")
        }

        st.subheader("Company Details")
        st.dataframe(
            pd.DataFrame(details.items(), columns=["Particulars", "Value"]),
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Annual Financial Results")
        annual_results = company_ticker.financials

        if annual_results is not None and not annual_results.empty:
            st.dataframe(annual_results.T.head(5), use_container_width=True)
        else:
            st.info("Annual financial results are not available.")

        st.subheader("Major Shareholders")
        major_holders = company_ticker.major_holders

        if major_holders is not None and not major_holders.empty:
            st.dataframe(major_holders, use_container_width=True)
        else:
            st.info("Major shareholder data is not available.")

        st.subheader("Institutional Shareholders")
        institutional_holders = company_ticker.institutional_holders

        if institutional_holders is not None and not institutional_holders.empty:
            st.dataframe(institutional_holders, use_container_width=True)
        else:
            st.info("Institutional shareholder data is not available.")

        st.caption(
            "Source: Yahoo Finance. Data may be delayed, incomplete or unavailable. "
            "Verify financial information from official NSE/BSE filings before investing."
        )

    except Exception as e:
        st.error(f"Could not fetch company data: {e}")


# Quarterly Financial Results
st.divider()
st.header("📅 Quarterly Financial Results")

quarterly_symbol = st.text_input(
    "Enter NSE symbol for quarterly results",
    value="RELIANCE",
    key="quarterly_symbol"
).strip().upper()

if st.button("GET QUARTERLY RESULTS"):
    try:
        quarterly_ticker = yf.Ticker(quarterly_symbol + ".NS")

        quarterly_income = quarterly_ticker.quarterly_income_stmt

        if quarterly_income is not None and not quarterly_income.empty:
            st.subheader("Quarterly Income Statement")
            st.dataframe(
                quarterly_income.T.head(8),
                use_container_width=True
            )
        else:
            st.info("Quarterly income statement unavailable.")

        quarterly_balance = quarterly_ticker.quarterly_balance_sheet

        if quarterly_balance is not None and not quarterly_balance.empty:
            st.subheader("Quarterly Balance Sheet")
            st.dataframe(
                quarterly_balance.T.head(8),
                use_container_width=True
            )
        else:
            st.info("Quarterly balance sheet unavailable.")

        quarterly_cashflow = quarterly_ticker.quarterly_cashflow

        if quarterly_cashflow is not None and not quarterly_cashflow.empty:
            st.subheader("Quarterly Cash Flow")
            st.dataframe(
                quarterly_cashflow.T.head(8),
                use_container_width=True
            )
        else:
            st.info("Quarterly cash flow unavailable.")

    except Exception as e:
        st.error(f"Quarterly data error: {e}")

    st.caption("Source: Yahoo Finance. Quarterly data may be incomplete or delayed.")


# Historical Stock Data Download
st.divider()
st.header("📥 Download Historical Stock Data")

download_symbol = st.text_input(
    "Enter NSE stock symbol",
    value="RELIANCE",
    key="historical_download_symbol"
).strip().upper()

download_period = st.selectbox(
    "Select historical period",
    ["1y", "2y", "5y", "10y"],
    index=2,
    key="historical_download_period"
)

if st.button("GET HISTORICAL DATA"):
    try:
        historical_data = yf.download(
            download_symbol + ".NS",
            period=download_period,
            interval="1d",
            auto_adjust=True,
            progress=False
        )

        if isinstance(historical_data.columns, pd.MultiIndex):
            historical_data.columns = historical_data.columns.get_level_values(0)

        historical_data = historical_data.reset_index()

        if not historical_data.empty:
            st.success(
                f"Downloaded {len(historical_data)} daily records for {download_symbol}."
            )

            st.dataframe(
                historical_data.tail(20),
                use_container_width=True,
                hide_index=True
            )

            historical_csv = historical_data.to_csv(index=False).encode("utf-8")

            st.download_button(
                "📥 Download Historical Data CSV",
                data=historical_csv,
                file_name=f"{download_symbol}_{download_period}_history.csv",
                mime="text/csv"
            )
        else:
            st.warning("No historical data found. Check the NSE symbol.")

    except Exception as e:
        st.error(f"Historical data error: {e}")

    st.caption(
        "Source: Yahoo Finance. Historical data availability and accuracy may vary."
    )


# Stock Price Alert System
st.divider()
st.header("🔔 Stock Price Alert System")

alert_symbol = st.text_input(
    "NSE Stock Symbol",
    value="RELIANCE",
    key="alert_symbol"
).strip().upper()

alert_target = st.number_input(
    "Target Price (₹)",
    min_value=0.0,
    value=1600.0,
    step=10.0,
    key="alert_target"
)

alert_stoploss = st.number_input(
    "Stop Loss Price (₹)",
    min_value=0.0,
    value=1300.0,
    step=10.0,
    key="alert_stoploss"
)

if st.button("CHECK PRICE ALERT"):
    try:
        alert_data = yf.download(
            alert_symbol + ".NS",
            period="5d",
            interval="1d",
            auto_adjust=True,
            progress=False
        )

        if isinstance(alert_data.columns, pd.MultiIndex):
            alert_data.columns = alert_data.columns.get_level_values(0)

        if not alert_data.empty:
            current_price = float(alert_data["Close"].dropna().iloc[-1])

            st.metric("Latest Available Closing Price", f"₹{current_price:.2f}")

            if alert_target > 0 and current_price >= alert_target:
                st.success("🎯 TARGET PRICE REACHED!")

            if alert_stoploss > 0 and current_price <= alert_stoploss:
                st.error("⚠️ STOP-LOSS LEVEL REACHED!")

            if (
                alert_target > 0
                and alert_stoploss > 0
                and alert_stoploss < current_price < alert_target
            ):
                st.info("Price is between your target and stop-loss levels.")

            st.caption(
                "This is a manual price check using the latest available daily data, "
                "not a continuous live alert. Verify prices with your broker."
            )
        else:
            st.warning("Stock data unavailable. Check the NSE symbol.")

    except Exception as e:
        st.error(f"Price alert error: {e}")
