import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta

st.set_page_config(page_title="NSE Stock Technical Analysis App", layout="wide")

st.title("📈 NSE Stock Data & Technical Analysis Dashboard")
st.markdown("Upload any raw stock data file (CSV, XLS, XLSX) from the NSE website. The app will automatically parse dates, prices, and volumes.")

# Sidebar for file upload supporting multiple formats
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload NSE File", type=["csv", "xlsx", "xls"])

# Function to generate sample data if none uploaded
@st.cache_data
def load_default_data():
    dates = pd.date_range(end=pd.Timestamp.today(), periods=250, freq='B')
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(250) * 1.5)
    df = pd.DataFrame({
        'Date': dates,
        'Open': prices + np.random.randn(250) * 0.5,
        'High': prices + abs(np.random.randn(250) * 1.5),
        'Low': prices - abs(np.random.randn(250) * 1.5),
        'Close': prices + np.random.randn(250) * 0.5,
        'Volume': np.random.randint(100000, 5000000, size=250)
    })
    return df

# Universal parser for raw NSE files
def parse_raw_file(file):
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file)
        else:
            return None, "Unsupported file format."

        df.columns = df.columns.str.strip()
        
        col_mapping = {}
        for col in df.columns:
            c_upper = col.upper()
            if any(k in c_upper for k in ['DATE', 'TIMESTAMP', 'TRADDT']):
                col_mapping[col] = 'Date'
            elif c_upper in ['OPEN', 'OPEN_PRICE', 'OPENPRC']:
                col_mapping[col] = 'Open'
            elif c_upper in ['HIGH', 'HIGH_PRICE', 'HIGHPRC']:
                col_mapping[col] = 'High'
            elif c_upper in ['LOW', 'LOW_PRICE', 'LOWPRC']:
                col_mapping[col] = 'Low'
            elif c_upper in ['CLOSE', 'CLOSE_PRICE', 'CLOSEPRC']:
                col_mapping[col] = 'Close'
            elif c_upper in ['TOTTRDQTY', 'VOLUME', 'TTL_TRD_QNTY', 'QTY', 'TRADED_QTY']:
                col_mapping[col] = 'Volume'
                
        df = df.rename(columns=col_mapping)
        
        required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            return None, f"Could not automatically map columns. Found columns: {list(df.columns)}."

        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace(',', '').str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        df = df.dropna(subset=['Date', 'Close']).sort_values('Date').reset_index(drop=True)
        return df, None
    except Exception as e:
        return None, str(e)

if uploaded_file is not None:
    df, error_msg = parse_raw_file(uploaded_file)
    if error_msg:
        st.error(f"Error parsing file: {error_msg}")
        df = load_default_data()
    else:
        st.sidebar.success("NSE File successfully processed!")
else:
    st.sidebar.info("Using sample stock data. Upload an NSE file (CSV/Excel) to test.")
    df = load_default_data()

# --- CALCULATE TECHNICAL INDICATORS ---
df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
df['RSI'] = ta.momentum.rsi(df['Close'], window=14)

macd = ta.trend.MACD(df['Close'])
df['MACD'] = macd.macd()
df['MACD_signal'] = macd.macd_signal()
df['MACD_diff'] = macd.macd_diff()

bollinger = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
df['BB_High'] = bollinger.bollinger_hband()
df['BB_Low'] = bollinger.bollinger_lband()
df['BB_Mid'] = bollinger.bollinger_mavg()

# --- TABS FOR ORGANIZING CHARTS ---
tab1, tab2, tab3 = st.tabs(["📊 Financial & Technical Charts", "📉 General Statistical Plots", "📋 Raw Data"])

with tab1:
    st.subheader("Core Technical Analysis Indicators")
    
    st.markdown("### 1 & 5. Candlestick Chart with Bollinger Bands & Moving Averages")
    fig_candle = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
    fig_candle.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Candlestick'), row=1, col=1)
    fig_candle.add_trace(go.Scatter(x=df['Date'], y=df['SMA_20'], line=dict(color='orange', width=1.5), name='SMA 20'), row=1, col=1)
    fig_candle.add_trace(go.Scatter(x=df['Date'], y=df['SMA_50'], line=dict(color='blue', width=1.5), name='SMA 50'), row=1, col=1)
    fig_candle.add_trace(go.Scatter(x=df['Date'], y=df['BB_High'], line=dict(color='gray', dash='dash', width=1), name='BB High'), row=1, col=1)
    fig_candle.add_trace(go.Scatter(x=df['Date'], y=df['BB_Low'], line=dict(color='gray', dash='dash', width=1), name='BB Low', fill='tonexty'), row=1, col=1)
    fig_candle.add_trace(go.Bar(x=df['Date'], y=df['Volume'], name='Volume', marker_color='purple'), row=2, col=1)
    fig_candle.update_layout(height=600, xaxis_rangeslider_visible=False, template='plotly_dark')
    st.plotly_chart(fig_candle, use_container_width=True)

    st.markdown("### 3. Relative Strength Index (RSI)")
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], line=dict(color='magenta', width=2), name='RSI'))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
    fig_rsi.update_layout(height=300, yaxis_range=[0, 100], template='plotly_dark', margin=dict(t=20, b=20))
    st.plotly_chart(fig_rsi, use_container_width=True)

    st.markdown("### 4. Moving Average Convergence & Divergence (MACD)")
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=df['Date'], y=df['MACD'], line=dict(color='cyan', width=1.5), name='MACD'))
    fig_macd.add_trace(go.Scatter(x=df['Date'], y=df['MACD_signal'], line=dict(color='orange', width=1.5), name='Signal'))
    fig_macd.add_trace(go.Bar(x=df['Date'], y=df['MACD_diff'], name='Histogram', marker_color='gray'))
    fig_macd.update_layout(height=300, template='plotly_dark', margin=dict(t=20, b=20))
    st.plotly_chart(fig_macd, use_container_width=True)

with tab2:
    st.subheader("Statistical Plots & Alternative Views")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 6. Closing Price Line Chart")
        fig_line = go.Figure(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', line=dict(color='lime')))
        fig_line.update_layout(height=350, template='plotly_dark')
        st.plotly_chart(fig_line, use_container_width=True)

        st.markdown("### 7. Trading Volume Bar Chart")
        fig_bar = go.Figure(go.Bar(x=df['Date'], y=df['Volume'], marker_color='dodgerblue'))
        fig_bar.update_layout(height=350, template='plotly_dark')
        st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("### 9. Histogram of Close Prices")
        fig_hist = go.Figure(go.Histogram(x=df['Close'], nbinsx=20, marker_color='teal'))
        fig_hist.update_layout(height=350, template='plotly_dark')
        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        df['Cumulative_Return'] = (1 + df['Close'].pct_change()).cumprod() - 1
        st.markdown("### 12. Cumulative Return Area Chart")
        fig_area = go.Figure(go.Scatter(x=df['Date'], y=df['Cumulative_Return'], fill='tozeroy', line=dict(color='gold')))
        fig_area.update_layout(height=350, template='plotly_dark')
        st.plotly_chart(fig_area, use_container_width=True)

        st.markdown("### 10. Box Plot of OHLC Prices")
        fig_box = go.Figure()
        for col in ['Open', 'High', 'Low', 'Close']:
            fig_box.add_trace(go.Box(y=df[col], name=col))
        fig_box.update_layout(height=350, template='plotly_dark')
        st.plotly_chart(fig_box, use_container_width=True)

        st.markdown("### 11. Scatter Plot: Volume vs. Close Price")
        fig_scatter = go.Figure(go.Scatter(x=df['Volume'], y=df['Close'], mode='markers', marker=dict(color=df['Close'], colorscale='Viridis', showscale=True)))
        fig_scatter.update_layout(height=350, template='plotly_dark')
        st.plotly_chart(fig_scatter, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        df['Month'] = df['Date'].dt.strftime('%B %Y')
        monthly_vol = df.groupby('Month')['Volume'].sum().reset_index()
        st.markdown("### 8. Pie Chart: Volume Distribution by Month")
        fig_pie = go.Figure(go.Pie(labels=monthly_vol['Month'], values=monthly_vol['Volume'], hole=.3))
        fig_pie.update_layout(height=400, template='plotly_dark')
        st.plotly_chart(fig_pie, use_container_width=True)

    with col4:
        monthly_hl = df.groupby('Month')[['High', 'Low']].sum().reset_index()
        st.markdown("### 13. Stacked Bar Chart: Monthly High/Low Totals")
        fig_stacked = go.Figure()
        fig_stacked.add_trace(go.Bar(x=monthly_hl['Month'], y=monthly_hl['High'], name='Total High', marker_color='indianred'))
        fig_stacked.add_trace(go.Bar(x=monthly_hl['Month'], y=monthly_hl['Low'], name='Total Low', marker_color='royalblue'))
        fig_stacked.update_layout(barmode='stack', height=400, template='plotly_dark')
        st.plotly_chart(fig_stacked, use_container_width=True)

with tab3:
    st.subheader("Cleaned Dataset View")
    st.dataframe(df, use_container_width=True)
