import pytz
from sklearn.ensemble import IsolationForest
import streamlit as st
import pandas as pd
import yfinance as yf
import datetime
import altair as alt
import plotly.graph_objects as go
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense # type: ignore
import pickle

# Function to add a ticker bubble
def add_bubble():
    if st.session_state.new_bubble:
        st.session_state.current_ticker = st.session_state.new_bubble
        st.session_state.new_bubble = ''

# Function to remove a ticker bubble
def remove_bubble(bubble):
    if st.session_state.current_ticker == bubble:
        st.session_state.current_ticker = None
    st.session_state.bubbles.remove(bubble)  

# Function to visualize ticker bubbles
def displayTickerBubbles():
    st.markdown("""
        <style>
        .stButton button {
            margin: 2px;
            white-space: nowrap;
        }
        </style>
        """, unsafe_allow_html=True)

    bubble_container = st.container()
    cols = bubble_container.columns(10)

    for i, bubble in enumerate(st.session_state.bubbles):
        if cols[i % 10].button(bubble, key=bubble):
            st.session_state.current_ticker = bubble
            st.session_state.bubbles = []  # Clear bubbles after selection
            st.rerun()  # Refresh program upon addition of ticker bubble

# Function to display input widgets
def displayInput():
    input, refresh = st.columns([9, 1])
    with input:
        st.text_input("Enter ticker:", 
                      key='new_bubble', 
                      placeholder="Enter Stock Ticker e.g. RELIANCE.NS", 
                      on_change=add_bubble)
    with refresh:
        st.markdown("<br>", unsafe_allow_html=True)
        st.button("Refresh", key='refresh_button')

def displayStockInfo(ticker):
    try:
        stockInfo = yf.Ticker(ticker)
        info = stockInfo.info
        
        # Check if info is not empty
        if not info:
            st.error(f"No data found for ticker: {ticker}")
            return

        # Display company name
        st.subheader(str(info.get('longName', 'N/A')))

        price, vol, float_ = st.columns(3)
        
        def safe_format(value):
            try:
                return str('{:,}'.format(float(value)))
            except (ValueError, TypeError):
                return 'N/A'

        with price:
            currentPrice = info.get('currentPrice', 'N/A')
            previousClose = info.get('previousClose', 'N/A')
            try:
                currentPrice = float(currentPrice)
                previousClose = float(previousClose)
                dailyChange = round((currentPrice - previousClose) / previousClose, 4) * 100
                delta = f"{round(dailyChange, 2)} %"
            except (ValueError, TypeError):
                delta = 'N/A'
            st.metric(label="Price",
                      value=safe_format(currentPrice) + " " + info.get('currency', 'INR'), 
                      delta=delta)

        with vol:
            currentVolume = info.get('volume', 'N/A')
            averageVolume = info.get('averageVolume', 'N/A')
            try:
                currentVolume = float(currentVolume)
                averageVolume = float(averageVolume)
                dailyVolChange = round((currentVolume - averageVolume) / averageVolume, 4) * 100
                vol_delta = f"{round(dailyVolChange, 2)} %"
            except (ValueError, TypeError):
                vol_delta = 'N/A'
            st.metric(label="Volume",
                      value=safe_format(currentVolume), 
                      delta=vol_delta)

        with float_:
            sharesFloat = info.get('floatShares', 'N/A')
            st.metric(label="Float",
                      value=safe_format(sharesFloat))
    
    except Exception as e:
        st.error(f"Error fetching stock information: {e}")


# Function to generate buy/sell signals (Assumed implementation)
def generate_buy_sell_signals(data):
    data['SMA'] = data['Close'].rolling(window=20).mean()
    data['EMA'] = data['Close'].ewm(span=20, adjust=False).mean()
    data['Upper_BB'], data['Lower_BB'] = calculate_bollinger_bands(data['Close'])
    data['Buy_Signal'] = (data['Close'] > data['SMA']) & (data['Close'] < data['Upper_BB'])
    data['Sell_Signal'] = (data['Close'] < data['SMA']) & (data['Close'] > data['Lower_BB'])
    return data

# Function to calculate Bollinger Bands (Assumed implementation)
def calculate_bollinger_bands(series, window=20, num_std=2):
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, lower_band

# Function to preprocess data for LSTM
def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Close']])
    
    X, y = [], []
    for i in range(len(scaled_data) - 60):
        X.append(scaled_data[i:i + 60])
        y.append(scaled_data[i + 60])
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y, scaler

# Function to build and train LSTM model
def train_lstm_model(X, y):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=10, batch_size=32, verbose=1)
    return model

# Function to make future predictions using the trained LSTM model
def predict_future_prices(model, scaler, data, days=30):
    last_60_days = data[['Close']].tail(60)
    last_60_days_scaled = scaler.transform(last_60_days)
    X_test = np.array([last_60_days_scaled])
    
    predictions = []
    for _ in range(days):
        predicted_price = model.predict(X_test)[0][0]
        predictions.append(predicted_price)
        
        # Update the input for the next prediction
        new_input = np.append(X_test[0][1:], [[predicted_price]], axis=0)
        X_test = np.array([new_input])
    
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# Function to detect anomalies in stock prices
def detect_anomalies(data):
    model = IsolationForest(contamination=0.01, random_state=42)
    data['Anomaly'] = model.fit_predict(data[['Close']])
    return data

# Load the pre-trained trading signal model
def load_model():
    with open('trading_signal_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Function to generate trading signals
def generate_trading_signals(data, model):
    # Feature extraction (you need to define how to extract features for your model)
    features = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Predict signals
    predictions = model.predict(scaled_features)
    data['Trading_Signal'] = predictions
    return data

# Function to display stock price chart with buy/sell signals, future predictions, and anomalies
def displayPredictionChart(ticker):
    try:
        stockInfo = yf.Ticker(ticker)
        
        # Get the current time in UTC and IST
        utc_now = datetime.datetime.now(pytz.utc)
        ist = pytz.timezone('Asia/Kolkata')
        ist_now = utc_now.astimezone(ist)
        todayIST = datetime.datetime.now(ist).date()
        
        start_date = todayIST - datetime.timedelta(days=90)  # 3 months
        
        # Fetch historical data
        data = stockInfo.history(start=start_date, end=ist_now, interval='1d')
        
        # Generate buy/sell signals
        data = generate_buy_sell_signals(data)
        
        # Detect anomalies
        data = detect_anomalies(data)
        
        # Check if 'Datetime' column exists
        if 'Datetime' not in data.columns:
            data.reset_index(inplace=True)
            if 'Date' in data.columns:
                data.rename(columns={'Date': 'Datetime'}, inplace=True)
        
        if not data.empty:
            data['Datetime'] = pd.to_datetime(data['Datetime'])
            
            # Prepare data for LSTM
            X, y, scaler = preprocess_data(data)
            
            # Train LSTM model
            model = train_lstm_model(X, y)
            
            # Predict future prices
            future_prices = predict_future_prices(model, scaler, data)
            future_dates = [data['Datetime'].max() + datetime.timedelta(days=i+1) for i in range(len(future_prices))]
            
            # Plotly for buy/sell signals chart with predictions and anomalies
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data['Datetime'], y=data['Close'], mode='lines', name='Close Price'))
            fig.add_trace(go.Scatter(x=data['Datetime'], y=data['SMA'], mode='lines', name='SMA'))
            fig.add_trace(go.Scatter(x=data['Datetime'], y=data['EMA'], mode='lines', name='EMA'))
            fig.add_trace(go.Scatter(x=data['Datetime'], y=data['Upper_BB'], mode='lines', fill=None, name='Upper BB'))
            fig.add_trace(go.Scatter(x=data['Datetime'], y=data['Lower_BB'], mode='lines', fill='tonexty', name='Lower BB'))
            
            # Add Buy and Sell signals
            buy_signals = data[data['Buy_Signal']]
            sell_signals = data[data['Sell_Signal']]
            fig.add_trace(go.Scatter(x=buy_signals['Datetime'], y=buy_signals['Close'], mode='markers', marker=dict(color='green', symbol='triangle-up', size=10), name='Buy Signal'))
            fig.add_trace(go.Scatter(x=sell_signals['Datetime'], y=sell_signals['Close'], mode='markers', marker=dict(color='red', symbol='triangle-down', size=10), name='Sell Signal'))
            
            # Add future price predictions
            fig.add_trace(go.Scatter(x=future_dates, y=future_prices.flatten(), mode='lines', name='Future Price Prediction', line=dict(dash='dash')))
            
            # Add anomalies
            anomalies = data[data['Anomaly'] == -1]
            fig.add_trace(go.Scatter(x=anomalies['Datetime'], y=anomalies['Close'], mode='markers', marker=dict(color='purple', symbol='x', size=10), name='Anomaly'))
            
            fig.update_layout(title=f'{ticker} Stock Price with Indicators, Buy/Sell Signals, Future Predictions, and Anomalies',
                              xaxis_title='Date',
                              yaxis_title='Price',
                              xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.write("No data available for the selected ticker or date.")
        
        # Display recent news headlines
        st.subheader("News")
        for news in stockInfo.news:
            news_time = datetime.datetime.fromtimestamp(news['providerPublishTime'])
            st.write(f"{news_time.strftime('%Y-%m-%d %H:%M:%S')} - [{news['title']}]({news['link']})")
    
    except Exception as e:
        st.error(f"Error fetching chart data: {e}")



# Function to display stock price chart for a given ticker
def displayChart(ticker):
    try:
        stockInfo = yf.Ticker(ticker)
        
        # Get user selections for chart type and time interval
        chart_type = st.selectbox('Select chart type:', ['Line Chart', 'Candlestick Chart', 'Stock Price Prediction with Buy/Sell Indications'])
        interval = st.selectbox('Select time interval:', ['5m', '15m', '30m', '1h', '1d', '1wk', '1mo', '3mo'])

        # Get the current time in UTC and IST
        utc_now = datetime.datetime.now(pytz.utc)
        ist = pytz.timezone('Asia/Kolkata')
        ist_now = utc_now.astimezone(ist)
        todayIST = datetime.datetime.now(ist).date()
        
        # Determine the start date based on the selected time range
        if interval == '5m':
            start_date = todayIST - datetime.timedelta(days=5)
        elif interval == '15m':
            start_date = todayIST - datetime.timedelta(days=15)
        elif interval == '30m':
            start_date = todayIST - datetime.timedelta(days=30)
        elif interval == '1h':
            start_date = todayIST - datetime.timedelta(days=60)
        elif interval == '1d':
            start_date = todayIST - datetime.timedelta(days=90)
        elif interval == '1wk':
            start_date = todayIST - datetime.timedelta(weeks=13)
        elif interval == '1mo':
            start_date = todayIST - datetime.timedelta(weeks=52)
        elif interval == '3mo':
            start_date = todayIST - datetime.timedelta(weeks=156)
        else:
            start_date = todayIST - datetime.timedelta(days=180)
        
        # Fetch historical data
        data = stockInfo.history(start=start_date, end=ist_now, interval=interval)
        
        # Check if 'Datetime' column exists
        if 'Datetime' not in data.columns:
            data.reset_index(inplace=True)
            if 'Date' in data.columns:
                data.rename(columns={'Date': 'Datetime'}, inplace=True)
        
        if not data.empty:
            data['Datetime'] = pd.to_datetime(data['Datetime'])
            
            if chart_type == 'Line Chart':
                # Line Chart using Altair
                chart = alt.Chart(data).mark_line().encode(
                    x='Datetime:T',
                    y='Close:Q'
                ).properties(
                    title=f'{ticker} Stock Price',
                    width=800,
                    height=400
                )
                st.altair_chart(chart, use_container_width=True)
            
            elif chart_type == 'Candlestick Chart':
                # Candlestick Chart using Plotly
                fig = go.Figure(data=[go.Candlestick(x=data['Datetime'],
                                                     open=data['Open'],
                                                     high=data['High'],
                                                     low=data['Low'],
                                                     close=data['Close'])])
                fig.update_layout(title=f'{ticker} Candlestick Chart',
                                  xaxis_title='Date',
                                  yaxis_title='Price',
                                  xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
            
            elif chart_type == 'Stock Price Prediction with Buy/Sell Indications':
                displayPredictionChart(ticker)
            
        else:
            st.write("No data available for the selected ticker or date.")
        
        # Display recent news headlines
        st.subheader("News")
        for news in stockInfo.news:
            news_time = datetime.datetime.fromtimestamp(news['providerPublishTime'])
            st.write(f"{news_time.strftime('%Y-%m-%d %H:%M:%S')} - [{news['title']}]({news['link']})")
    
    except Exception as e:
        st.error(f"Error fetching chart data: {e}")

# Display stock information for the selected ticker 
def displayWatchlist():
    if st.session_state.current_ticker:
        displayStockInfo(st.session_state.current_ticker)
        displayChart(st.session_state.current_ticker)

def main():
    # Initialize Streamlit title
    st.title('Real-time Stock Price Prediction')

    # Initialize session state for bubbles and current_ticker
    if 'bubbles' not in st.session_state:
        st.session_state.bubbles = []
    if 'current_ticker' not in st.session_state:
        st.session_state.current_ticker = None

    displayInput()
    displayTickerBubbles()
    displayWatchlist()

if __name__ == '__main__':
    main()
