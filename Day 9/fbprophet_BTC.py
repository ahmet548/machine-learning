from flask import Flask, jsonify, request
from flask_cors import CORS
import yfinance as yf
from prophet import Prophet

app = Flask(__name__)
CORS(app)

SYMBOL_MAP = {
    'BTC': 'BTC-USD',
    'ETH': 'ETH-USD',
    'SOL': 'SOL-USD',
    'GOOG': 'GOOG',
    'BFREN': 'BFREN.IS'
}

@app.route('/api/forecast')
def forecast():
    symbol = request.args.get('symbol', 'BTC')
    days = int(request.args.get('days', 30))
    yf_symbol = SYMBOL_MAP.get(symbol, 'BTC-USD')

    df = yf.download(yf_symbol, period="1y")
    df = df[['Close']].reset_index()
    df.columns = ['ds', 'y']
    df['ds'] = df['ds'].dt.tz_localize(None)

    model = Prophet()
    model.fit(df)

    future = model.make_future_dataframe(periods=days)
    pred = model.predict(future)

    labels = [d.strftime('%Y-%m-%d') for d in pred['ds']]
    hist_len = len(df)

    historical_data = [float(v) for v in df['y']] + [None] * days
    forecast_data = [None] * (hist_len - 1) + [float(v) for v in pred['yhat'].iloc[hist_len-1:]]
    upper_data = [None] * (hist_len - 1) + [float(v) for v in pred['yhat_upper'].iloc[hist_len-1:]]
    lower_data = [None] * (hist_len - 1) + [float(v) for v in pred['yhat_lower'].iloc[hist_len-1:]]

    return jsonify({
        'labels': labels,
        'historicalData': historical_data,
        'forecastData': forecast_data,
        'upperData': upper_data,
        'lowerData': lower_data
    })

if __name__ == '__main__':
    app.run(port=8000)