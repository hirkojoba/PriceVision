# 📈 PriceVision - AI Stock Trend Predictor

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

An intelligent stock price trend prediction application powered by machine learning. PriceVision analyzes historical stock data and predicts next-day price trends (UP, DOWN, or FLAT) with confidence scores.

> ⚠️ **EDUCATIONAL PURPOSE ONLY** - This application is designed for educational and portfolio demonstration purposes. It does NOT constitute financial advice. Do not use these predictions for actual trading or investment decisions.

---

## Features

- **Trend Prediction**: Predict next-day stock price movements (UP/DOWN/FLAT)
- **Machine Learning**: Uses Random Forest or Gradient Boosting classifiers
- **Technical Indicators**: 15+ engineered features including moving averages, RSI, volatility
- **Interactive Visualizations**: Beautiful charts for price history, confusion matrices, feature importance
- **Model Explainability**: Feature importance analysis and performance metrics
- **PDF Reports**: Downloadable comprehensive prediction reports
- **Web Interface**: Easy-to-use Streamlit dashboard
- **Tested**: Comprehensive unit test coverage

---

## Tech Stack

- **Python 3.8+**
- **Machine Learning**: scikit-learn (Random Forest, Gradient Boosting)
- **Data Processing**: pandas, numpy
- **Data Source**: yfinance (Yahoo Finance API)
- **Visualization**: matplotlib, plotly, seaborn
- **Web Framework**: Streamlit
- **PDF Generation**: ReportLab
- **Testing**: pytest
- **Cloud Deployment**: Railway

---

## Usage

### Using the Application

1. **Configure Settings** (Left Sidebar):
   - Enter a stock ticker (e.g., AAPL, MSFT, TSLA)
   - Select date range (1-10 years of historical data)
   - Choose model type and parameters

2. **Train Model**:
   - Click "Train Model" button
   - Wait for data download and model training to complete
   - View model accuracy and performance metrics

3. **Make Prediction**:
   - Click "Predict Next Day" button
   - See prediction with confidence scores
   - Explore detailed analysis in different tabs

4. **Download Report**:
   - Navigate to "Report" tab
   - Generate and download comprehensive PDF report

---

## How It Works

### 1. Data Collection
- Downloads historical OHLCV (Open, High, Low, Close, Volume) data from Yahoo Finance
- Supports customizable date ranges (1-10 years)

### 2. Feature Engineering
Creates 15+ technical indicators:
- **Price Features**: Daily returns, moving averages (5, 10, 20-day)
- **Volatility**: Rolling standard deviation of returns
- **Volume**: Volume changes and ratios
- **Position**: Price relative to moving averages
- **Momentum**: RSI (Relative Strength Index)
- **Lag Features**: Previous day values

### 3. Label Creation
Defines trend based on next-day price movement:
- **UP** ⬆️: Next day close ≥ today's close × 1.005 (+0.5%)
- **DOWN** ⬇️: Next day close ≤ today's close × 0.995 (-0.5%)
- **FLAT** ➡️: Between -0.5% and +0.5%

### 4. Model Training
- **Train/Test Split**: Chronological (80/20) - NO random shuffling
- **Algorithms**: Random Forest (default) or Gradient Boosting
- **Evaluation**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix

### 5. Prediction
- Processes latest data through feature pipeline
- Outputs trend prediction with probability distribution
- Categorizes confidence level (HIGH/MEDIUM/LOW)

---

## Model Performance

Typical performance metrics on 5 years of data:

- **Accuracy**: 50-65% (significantly better than random 33%)
- **Precision**: Varies by class (UP/DOWN/FLAT)
- **Feature Importance**: Moving averages and momentum indicators typically most important

> **Note**: Stock market prediction is inherently uncertain. No model can consistently predict market movements with high accuracy.

---

## Screenshots

### Data Overview
![Data Overview Part 1](screenshots/data%20overview%20pt%201.jpeg)
*Historical price chart with moving averages*

![Data Overview Part 2](screenshots/data%20overview%20pt%202.jpeg)
*Price history and summary statistics*

![Data Overview Part 3](screenshots/data%20overview%20pt%203%20(view%20raw%20data).jpeg)
*Raw data preview and exploration*

### Next-Day Trend Prediction
![Prediction](screenshots/next%20daytrend%20prediction.jpeg)
*Trend prediction with confidence scores and probability distribution*

### Model Performance Analysis
![Performance Part 1](screenshots/model%20performance%20analysis%20pt%201.jpeg)
*Overall accuracy and confusion matrix*

![Performance Part 2](screenshots/model%20performance%20analysis%20pt%202.jpeg)
*Per-class metrics and feature importance*

![Performance Part 3](screenshots/model%20performance%20analysis%20pt%203%20(Detailed%20Classification%20Report).jpeg)
*Detailed classification report*

### PDF Report Generation
![PDF Report](screenshots/generate%20PDF%20report.jpeg)
*Downloadable comprehensive prediction report*

---

## Disclaimer

**IMPORTANT**: This application is for **educational purposes only**.

- **NOT financial advice**
- **NOT a guarantee of profitability**
- **NOT suitable for real trading**
- **FOR learning and demonstration only**

Stock market predictions are inherently uncertain. Past performance does not guarantee future results. Always consult with a qualified financial advisor before making investment decisions.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
